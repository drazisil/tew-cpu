const std = @import("std");

// ─── Register indices ────────────────────────────────────────────────────────
const EAX: u3 = 0;
const ECX: u3 = 1;
const EDX: u3 = 2;
const EBX: u3 = 3;
const ESP: u3 = 4;
const EBP: u3 = 5;
const ESI: u3 = 6;
const EDI: u3 = 7;

// ─── EFLAGS bit positions ─────────────────────────────────────────────────────
const CF_BIT: u5 = 0;
const PF_BIT: u5 = 2;
const ZF_BIT: u5 = 6;
const SF_BIT: u5 = 7;
const DF_BIT: u5 = 10;
const OF_BIT: u5 = 11;

// ─── Segment override tokens ─────────────────────────────────────────────────
const SEG_NONE: u8 = 0;
const SEG_FS: u8 = 1;
const SEG_GS: u8 = 2;

// ─── REP prefix tokens ───────────────────────────────────────────────────────
const REP_NONE: u8 = 0;
const REP_REP: u8 = 1;
const REP_REPNE: u8 = 2;

// ─── Public types ─────────────────────────────────────────────────────────────
pub const IntHandlerFn = *const fn (state: *anyopaque, int_num: u8) callconv(.C) void;
pub const OpFn = *const fn (*CpuState) void;

pub const RunResult = enum(c_int) {
    ok = 0,
    halted = 1,
    faulted = 2,
    step_limit = 3,
};

pub const CpuState = struct {
    regs: [8]u32 = .{0} ** 8,
    eip: u32 = 0,
    eflags: u32 = 0,
    fpu_stack: [8]f64 = .{0.0} ** 8,
    fpu_top: u32 = 0,
    fpu_status_word: u16 = 0,
    fpu_control_word: u16 = 0x037F,
    fpu_tag_word: u16 = 0xFFFF,
    halted: bool = false,
    faulted: bool = false,
    fs_base: u32 = 0,
    gs_base: u32 = 0,
    step_count: u64 = 0,
    last_opcode: u8 = 0,
    int_handler: ?IntHandlerFn = null,
    memory: [*]u8 = undefined,
    memory_size: usize = 0,
    seg_override: u8 = SEG_NONE,
    rep_prefix: u8 = REP_NONE,
    op_size_ovr: bool = false,
};

// ─── Internal helper structs ─────────────────────────────────────────────────
const RmInfo = struct { is_reg: bool, addr: u32 };
const Rm8Result = struct { value: u8, is_reg: bool, addr: u32 };
const Rm32Result = struct { value: u32, is_reg: bool, addr: u32 };
const ModRm = struct { mod: u8, reg: u8, rm: u8 };

// ─── Memory access ────────────────────────────────────────────────────────────
inline fn memRead8(s: *CpuState, addr: u32) u8 {
    if (addr >= @as(u32, @truncate(s.memory_size))) { s.faulted = true; s.halted = true; return 0; }
    return s.memory[addr];
}
inline fn memRead16(s: *CpuState, addr: u32) u16 {
    return @as(u16, memRead8(s, addr)) | (@as(u16, memRead8(s, addr + 1)) << 8);
}
inline fn memRead32(s: *CpuState, addr: u32) u32 {
    return @as(u32, memRead8(s, addr)) | (@as(u32, memRead8(s, addr + 1)) << 8) |
           (@as(u32, memRead8(s, addr + 2)) << 16) | (@as(u32, memRead8(s, addr + 3)) << 24);
}
inline fn memReadS32(s: *CpuState, addr: u32) i32 { return @bitCast(memRead32(s, addr)); }
inline fn memWrite8(s: *CpuState, addr: u32, v: u8) void {
    if (addr >= @as(u32, @truncate(s.memory_size))) { s.faulted = true; s.halted = true; return; }
    s.memory[addr] = v;
}
inline fn memWrite16(s: *CpuState, addr: u32, v: u16) void {
    memWrite8(s, addr, @truncate(v));
    memWrite8(s, addr + 1, @truncate(v >> 8));
}
inline fn memWrite32(s: *CpuState, addr: u32, v: u32) void {
    memWrite8(s, addr, @truncate(v));
    memWrite8(s, addr + 1, @truncate(v >> 8));
    memWrite8(s, addr + 2, @truncate(v >> 16));
    memWrite8(s, addr + 3, @truncate(v >> 24));
}

// ─── Fetch helpers ────────────────────────────────────────────────────────────
inline fn fetch8(s: *CpuState) u8 {
    const v = memRead8(s, s.eip);
    s.eip +%= 1;
    return v;
}
inline fn fetch16(s: *CpuState) u16 {
    const v = memRead16(s, s.eip);
    s.eip +%= 2;
    return v;
}
inline fn fetch32(s: *CpuState) u32 {
    const v = memRead32(s, s.eip);
    s.eip +%= 4;
    return v;
}
inline fn fetchS8(s: *CpuState) i8 { return @bitCast(fetch8(s)); }
inline fn fetchS32(s: *CpuState) i32 { return @bitCast(fetch32(s)); }
inline fn fetchImm(s: *CpuState) u32 {
    return if (s.op_size_ovr) @as(u32, fetch16(s)) else fetch32(s);
}
inline fn fetchSImm(s: *CpuState) i32 {
    if (s.op_size_ovr) {
        const v = fetch16(s);
        return @as(i32, @as(i16, @bitCast(v)));
    }
    return fetchS32(s);
}

// ─── Flag helpers ─────────────────────────────────────────────────────────────
inline fn getFlag(s: *CpuState, bit: u5) bool { return ((s.eflags >> bit) & 1) != 0; }
inline fn setFlag(s: *CpuState, bit: u5, v: bool) void {
    if (v) s.eflags |= @as(u32, 1) << bit else s.eflags &= ~(@as(u32, 1) << bit);
}
fn updateFlagsArith(s: *CpuState, result_raw: i64, op1: u32, op2: u32, is_sub: bool) void {
    const r32: u32 = @truncate(@as(u64, @bitCast(result_raw)));
    setFlag(s, ZF_BIT, r32 == 0);
    setFlag(s, SF_BIT, (r32 & 0x80000000) != 0);
    var p: u8 = @truncate(r32);
    p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
    setFlag(s, PF_BIT, (p & 1) == 0);
    if (is_sub) {
        setFlag(s, CF_BIT, op1 < op2);
    } else {
        setFlag(s, CF_BIT, r32 < op1 or r32 < op2);
    }
    const s1 = (op1 & 0x80000000) != 0;
    const s2 = (op2 & 0x80000000) != 0;
    const sr = (r32 & 0x80000000) != 0;
    if (is_sub) {
        setFlag(s, OF_BIT, s1 != s2 and sr != s1);
    } else {
        setFlag(s, OF_BIT, s1 == s2 and sr != s1);
    }
}
fn updateFlagsLogic(s: *CpuState, result: u32) void {
    setFlag(s, ZF_BIT, result == 0);
    setFlag(s, SF_BIT, (result & 0x80000000) != 0);
    setFlag(s, CF_BIT, false);
    setFlag(s, OF_BIT, false);
    var p: u8 = @truncate(result);
    p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
    setFlag(s, PF_BIT, (p & 1) == 0);
}

// ─── 8-bit register helpers ──────────────────────────────────────────────────
inline fn readReg8(s: *CpuState, idx: u8) u8 {
    return if (idx < 4) @truncate(s.regs[idx]) else @truncate(s.regs[idx - 4] >> 8);
}
inline fn writeReg8(s: *CpuState, idx: u8, v: u8) void {
    if (idx < 4) {
        s.regs[idx] = (s.regs[idx] & 0xFFFFFF00) | @as(u32, v);
    } else {
        s.regs[idx - 4] = (s.regs[idx - 4] & 0xFFFF00FF) | (@as(u32, v) << 8);
    }
}

// ─── Stack helpers ────────────────────────────────────────────────────────────
inline fn push32(s: *CpuState, v: u32) void {
    s.regs[ESP] -%= 4;
    memWrite32(s, s.regs[ESP], v);
}
inline fn pop32(s: *CpuState) u32 {
    const v = memRead32(s, s.regs[ESP]);
    s.regs[ESP] +%= 4;
    return v;
}

// ─── Segment override ─────────────────────────────────────────────────────────
inline fn applySegOvr(s: *CpuState, addr: u32) u32 {
    return switch (s.seg_override) {
        SEG_FS => s.fs_base +% addr,
        SEG_GS => s.gs_base +% addr,
        else => addr,
    };
}

// ─── ModRM decode ─────────────────────────────────────────────────────────────
fn decodeModRM(s: *CpuState) ModRm {
    const b = fetch8(s);
    return .{ .mod = (b >> 6) & 3, .reg = (b >> 3) & 7, .rm = b & 7 };
}
fn decodeSIB(s: *CpuState, mod: u8) u32 {
    const sib = fetch8(s);
    const scale: u32 = @as(u32, 1) << @as(u2, @truncate(sib >> 6));
    const index: u8 = (sib >> 3) & 7;
    const base: u8 = sib & 7;
    var addr: u32 = if (base == 5 and mod == 0) fetch32(s) else s.regs[base];
    if (index != 4) addr +%= s.regs[index] *% scale;
    return addr;
}
fn resolveRm(s: *CpuState, mod: u8, rm: u8) RmInfo {
    if (mod == 3) return .{ .is_reg = true, .addr = rm };
    const addr: u32 = switch (mod) {
        0 => switch (rm) {
            5 => fetch32(s),
            4 => decodeSIB(s, mod),
            else => s.regs[rm],
        },
        1 => blk: {
            if (rm == 4) {
                const base = decodeSIB(s, mod);
                break :blk base +% @as(u32, @bitCast(@as(i32, fetchS8(s))));
            }
            break :blk s.regs[rm] +% @as(u32, @bitCast(@as(i32, fetchS8(s))));
        },
        2 => blk: {
            if (rm == 4) {
                const base = decodeSIB(s, mod);
                break :blk base +% @as(u32, @bitCast(fetchS32(s)));
            }
            break :blk s.regs[rm] +% @as(u32, @bitCast(fetchS32(s)));
        },
        else => unreachable,
    };
    return .{ .is_reg = false, .addr = addr };
}

// ─── rm read/write helpers ────────────────────────────────────────────────────
fn readRm8(s: *CpuState, mod: u8, rm: u8) u8 {
    const r = resolveRm(s, mod, rm);
    return if (r.is_reg) readReg8(s, @truncate(r.addr)) else memRead8(s, applySegOvr(s, r.addr));
}
fn writeRm8(s: *CpuState, mod: u8, rm: u8, v: u8) void {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) writeReg8(s, @truncate(r.addr), v) else memWrite8(s, applySegOvr(s, r.addr), v);
}
fn readRm8Resolved(s: *CpuState, mod: u8, rm: u8) Rm8Result {
    const r = resolveRm(s, mod, rm);
    const addr = if (r.is_reg) r.addr else applySegOvr(s, r.addr);
    const v: u8 = if (r.is_reg) readReg8(s, @truncate(r.addr)) else memRead8(s, addr);
    return .{ .value = v, .is_reg = r.is_reg, .addr = addr };
}
fn writeRm8Resolved(s: *CpuState, is_reg: bool, addr: u32, v: u8) void {
    if (is_reg) writeReg8(s, @truncate(addr), v) else memWrite8(s, addr, v);
}
fn readRm32(s: *CpuState, mod: u8, rm: u8) u32 {
    const r = resolveRm(s, mod, rm);
    return if (r.is_reg) s.regs[r.addr] else memRead32(s, applySegOvr(s, r.addr));
}
fn writeRm32(s: *CpuState, mod: u8, rm: u8, v: u32) void {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) s.regs[r.addr] = v else memWrite32(s, applySegOvr(s, r.addr), v);
}
fn readRm32Resolved(s: *CpuState, mod: u8, rm: u8) Rm32Result {
    const r = resolveRm(s, mod, rm);
    const addr = if (r.is_reg) r.addr else applySegOvr(s, r.addr);
    const v: u32 = if (r.is_reg) s.regs[r.addr] else memRead32(s, addr);
    return .{ .value = v, .is_reg = r.is_reg, .addr = addr };
}
fn writeRm32Resolved(s: *CpuState, is_reg: bool, addr: u32, v: u32) void {
    if (is_reg) s.regs[addr] = v else memWrite32(s, addr, v);
}
fn readRmv(s: *CpuState, mod: u8, rm: u8) u32 {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) {
        return if (s.op_size_ovr) s.regs[r.addr] & 0xFFFF else s.regs[r.addr];
    }
    const addr = applySegOvr(s, r.addr);
    return if (s.op_size_ovr) @as(u32, memRead16(s, addr)) else memRead32(s, addr);
}
fn writeRmv(s: *CpuState, mod: u8, rm: u8, v: u32) void {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) {
        if (s.op_size_ovr) s.regs[r.addr] = (s.regs[r.addr] & 0xFFFF0000) | (v & 0xFFFF)
        else s.regs[r.addr] = v;
    } else {
        const addr = applySegOvr(s, r.addr);
        if (s.op_size_ovr) memWrite16(s, addr, @truncate(v)) else memWrite32(s, addr, v);
    }
}
fn readRmvResolved(s: *CpuState, mod: u8, rm: u8) Rm32Result {
    const r = resolveRm(s, mod, rm);
    const addr = if (r.is_reg) r.addr else applySegOvr(s, r.addr);
    const v: u32 = if (r.is_reg)
        (if (s.op_size_ovr) s.regs[r.addr] & 0xFFFF else s.regs[r.addr])
    else
        (if (s.op_size_ovr) @as(u32, memRead16(s, addr)) else memRead32(s, addr));
    return .{ .value = v, .is_reg = r.is_reg, .addr = addr };
}
fn writeRmvResolved(s: *CpuState, is_reg: bool, addr: u32, v: u32) void {
    if (is_reg) {
        if (s.op_size_ovr) s.regs[addr] = (s.regs[addr] & 0xFFFF0000) | (v & 0xFFFF)
        else s.regs[addr] = v;
    } else {
        if (s.op_size_ovr) memWrite16(s, addr, @truncate(v)) else memWrite32(s, addr, v);
    }
}
inline fn readEaxv(s: *CpuState) u32 {
    return if (s.op_size_ovr) s.regs[EAX] & 0xFFFF else s.regs[EAX];
}
inline fn writeEaxv(s: *CpuState, v: u32) void {
    if (s.op_size_ovr) s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | (v & 0xFFFF)
    else s.regs[EAX] = v;
}

// ─── FPU helpers ──────────────────────────────────────────────────────────────
inline fn fpuGet(s: *CpuState, i: u8) f64 {
    return s.fpu_stack[(@as(u8, @truncate(s.fpu_top)) +% i) & 7];
}
inline fn fpuSet(s: *CpuState, i: u8, v: f64) void {
    const idx: u8 = (@as(u8, @truncate(s.fpu_top)) +% i) & 7;
    s.fpu_stack[idx] = v;
    s.fpu_tag_word &= ~(@as(u16, 3) << (@as(u4, @truncate(idx)) * 2));
}
fn fpuPush(s: *CpuState, v: f64) void {
    s.fpu_top = (s.fpu_top -% 1) & 7;
    s.fpu_stack[s.fpu_top] = v;
    s.fpu_tag_word &= ~(@as(u16, 3) << (@as(u4, @truncate(s.fpu_top)) * 2));
    s.fpu_status_word = (s.fpu_status_word & ~@as(u16, 0x3800)) |
                        @as(u16, @truncate((s.fpu_top & 7) << 11));
}
fn fpuPop(s: *CpuState) f64 {
    const v = s.fpu_stack[s.fpu_top & 7];
    s.fpu_tag_word |= @as(u16, 3) << (@as(u4, @truncate(s.fpu_top & 7)) * 2);
    s.fpu_top = (s.fpu_top +% 1) & 7;
    s.fpu_status_word = (s.fpu_status_word & ~@as(u16, 0x3800)) |
                        @as(u16, @truncate((s.fpu_top & 7) << 11));
    return v;
}
fn fpuSetCC(s: *CpuState, c3: bool, c2: bool, c0: bool) void {
    s.fpu_status_word &= ~@as(u16, 0x4500);
    if (c0) s.fpu_status_word |= 0x0100;
    if (c2) s.fpu_status_word |= 0x0400;
    if (c3) s.fpu_status_word |= 0x4000;
}
fn fpuCompare(s: *CpuState, a: f64, b: f64) void {
    if (std.math.isNan(a) or std.math.isNan(b)) {
        fpuSetCC(s, true, true, true);
    } else if (a > b) {
        fpuSetCC(s, false, false, false);
    } else if (a < b) {
        fpuSetCC(s, false, false, true);
    } else {
        fpuSetCC(s, true, false, false);
    }
}
fn fpuComi(s: *CpuState, a: f64, b: f64, do_pop: bool) void {
    if (std.math.isNan(a) or std.math.isNan(b)) {
        setFlag(s, ZF_BIT, true); setFlag(s, CF_BIT, true);
    } else if (a > b) {
        setFlag(s, ZF_BIT, false); setFlag(s, CF_BIT, false);
    } else if (a < b) {
        setFlag(s, ZF_BIT, false); setFlag(s, CF_BIT, true);
    } else {
        setFlag(s, ZF_BIT, true); setFlag(s, CF_BIT, false);
    }
    setFlag(s, OF_BIT, false);
    if (do_pop) _ = fpuPop(s);
}

// ─── Float I/O ────────────────────────────────────────────────────────────────
fn readFloat(s: *CpuState, addr: u32) f32 { return @bitCast(memRead32(s, addr)); }
fn writeFloat(s: *CpuState, addr: u32, v: f32) void { memWrite32(s, addr, @bitCast(v)); }
fn readDouble(s: *CpuState, addr: u32) f64 {
    const lo = memRead32(s, addr);
    const hi = memRead32(s, addr + 4);
    const bits: u64 = (@as(u64, hi) << 32) | @as(u64, lo);
    return @bitCast(bits);
}
fn writeDouble(s: *CpuState, addr: u32, v: f64) void {
    const bits: u64 = @bitCast(v);
    memWrite32(s, addr, @truncate(bits));
    memWrite32(s, addr + 4, @truncate(bits >> 32));
}

// ─── Condition evaluation (for Jcc/SETcc/CMOVcc) ─────────────────────────────
fn evalCond(s: *CpuState, cond: u8) bool {
    const cf = getFlag(s, CF_BIT); const zf = getFlag(s, ZF_BIT);
    const sf = getFlag(s, SF_BIT); const of = getFlag(s, OF_BIT);
    const pf = getFlag(s, PF_BIT);
    return switch (cond & 0xF) {
        0x0 => of, 0x1 => !of, 0x2 => cf, 0x3 => !cf,
        0x4 => zf, 0x5 => !zf, 0x6 => cf or zf, 0x7 => !cf and !zf,
        0x8 => sf, 0x9 => !sf, 0xA => pf, 0xB => !pf,
        0xC => sf != of, 0xD => sf == of, 0xE => zf or (sf != of),
        0xF => !zf and (sf == of), else => false,
    };
}

// ─── Fault / prefix ───────────────────────────────────────────────────────────
fn opFault(s: *CpuState) void { s.faulted = true; s.halted = true; }
fn clearPrefixes(s: *CpuState) void {
    s.seg_override = SEG_NONE; s.rep_prefix = REP_NONE; s.op_size_ovr = false;
}
fn isPrefix(b: u8) bool {
    return switch (b) {
        0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3 => true,
        else => false,
    };
}

// ─── Group 1 helper (ADD/OR/ADC/SBB/AND/SUB/XOR/CMP) ────────────────────────
fn doGroup1(s: *CpuState, is_reg: bool, addr: u32, op_ext: u8, op1: u32, op2: u32) void {
    switch (op_ext) {
        0 => { const r = op1 +% op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsArith(s, @as(i64, op1) + @as(i64, op2), op1, op2, false); },
        1 => { const r = op1 | op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsLogic(s, r); },
        2 => { const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0; const r = op1 +% op2 +% c; writeRmvResolved(s, is_reg, addr, r); updateFlagsArith(s, @as(i64, op1) + @as(i64, op2) + @as(i64, c), op1, op2 +% c, false); },
        3 => { const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0; const r = op1 -% op2 -% b; writeRmvResolved(s, is_reg, addr, r); updateFlagsArith(s, @as(i64, op1) - @as(i64, op2) - @as(i64, b), op1, op2 +% b, true); },
        4 => { const r = op1 & op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsLogic(s, r); },
        5 => { const r = op1 -% op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true); },
        6 => { const r = op1 ^ op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsLogic(s, r); },
        7 => updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true),
        else => { s.faulted = true; s.halted = true; },
    }
}

// ─── Group 2 helper (shift/rotate 32-bit) ─────────────────────────────────
fn doGroup2(s: *CpuState, is_reg: bool, addr: u32, op_ext: u8, val: u32, count: u8) void {
    if (count == 0) { writeRm32Resolved(s, is_reg, addr, val); return; }
    const c5: u5 = @truncate(count);
    switch (op_ext) {
        0 => { // ROL
            const r = (val << c5) | (val >> @as(u5, @truncate(32 - @as(u8, c5))));
            const new_cf = (r & 1) != 0;
            writeRmvResolved(s, is_reg, addr, r); setFlag(s, CF_BIT, new_cf);
            if (count == 1) setFlag(s, OF_BIT, ((r & 0x80000000) != 0) != (((r >> 1) & 0x40000000) != 0));
        },
        1 => { // ROR
            const r = (val >> c5) | (val << @as(u5, @truncate(32 - @as(u8, c5))));
            const new_cf = (r & 0x80000000) != 0;
            writeRmvResolved(s, is_reg, addr, r); setFlag(s, CF_BIT, new_cf);
            if (count == 1) setFlag(s, OF_BIT, ((r & 0x80000000) != 0) != ((val >> 31) != 0));
        },
        2 => { // RCL
            var temp = val << c5;
            if (getFlag(s, CF_BIT)) temp |= @as(u32, 1) << @as(u5, @truncate(count - 1));
            const new_cf = ((val >> @as(u5, @truncate(32 - @as(u8, count)))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, temp); setFlag(s, CF_BIT, new_cf);
        },
        3 => { // RCR
            var temp = val >> c5;
            if (getFlag(s, CF_BIT)) temp |= @as(u32, 1) << @as(u5, @truncate(32 - @as(u8, count)));
            const new_cf = ((val >> @as(u5, @truncate(count - 1))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, temp); setFlag(s, CF_BIT, new_cf);
        },
        4 => { // SHL
            const r = val << c5;
            const new_cf = ((val >> @as(u5, @truncate(32 - @as(u8, count)))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogic(s, r); // NOTE: update_flags_logic clears CF (matches Python)
        },
        5 => { // SHR
            const r = val >> c5;
            const new_cf = ((val >> @as(u5, @truncate(count - 1))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogic(s, r);
        },
        7 => { // SAR
            const r: u32 = @bitCast(@as(i32, @bitCast(val)) >> c5);
            const new_cf = ((val >> @as(u5, @truncate(count - 1))) & 1) != 0;
            writeRm32Resolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogic(s, r);
        },
        else => { s.faulted = true; s.halted = true; },
    }
}

// ─── Arithmetic opcodes ───────────────────────────────────────────────────────
fn op00(s: *CpuState) void { // ADD rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg); const r = res.value +% op2;
    writeRm8Resolved(s, res.is_reg, res.addr, r);
    updateFlagsArith(s, @as(i64, res.value) + @as(i64, op2), res.value, op2, false);
}
fn op01(s: *CpuState) void { // ADD rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const op2 = s.regs[d.reg]; const r = res.value +% op2;
    writeRmvResolved(s, res.is_reg, res.addr, r);
    updateFlagsArith(s, @as(i64, res.value) + @as(i64, op2), res.value, op2, false);
}
fn op02(s: *CpuState) void { // ADD r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, op1 +% op2);
    updateFlagsArith(s, @as(i64, op1) + @as(i64, op2), op1, op2, false);
}
fn op03(s: *CpuState) void { // ADD r32, rm32
    const d = decodeModRM(s); const op1 = s.regs[d.reg]; const op2 = readRm32(s, d.mod, d.rm);
    s.regs[d.reg] = op1 +% op2;
    updateFlagsArith(s, @as(i64, op1) + @as(i64, op2), op1, op2, false);
}
fn op04(s: *CpuState) void { // ADD AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al +% imm);
    updateFlagsArith(s, @as(i64, al) + @as(i64, imm), al, imm, false);
}
fn op05(s: *CpuState) void { // ADD EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s);
    writeEaxv(s, a +% imm); updateFlagsArith(s, @as(i64, a) + @as(i64, imm), a, imm, false);
}
fn op10(s: *CpuState) void { // ADC rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg); const c: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRm8Resolved(s, res.is_reg, res.addr, res.value +% op2 +% c);
    updateFlagsArith(s, @as(i64, res.value) + @as(i64, op2) + @as(i64, c), res.value, op2 +% c, false);
}
fn op11(s: *CpuState) void { // ADC rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const op2 = s.regs[d.reg]; const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRmvResolved(s, res.is_reg, res.addr, res.value +% op2 +% c);
    updateFlagsArith(s, @as(i64, res.value) + @as(i64, op2) + @as(i64, c), res.value, op2 +% c, false);
}
fn op12(s: *CpuState) void { // ADC r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    const c: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeReg8(s, d.reg, op1 +% op2 +% c);
    updateFlagsArith(s, @as(i64, op1) + @as(i64, op2) + @as(i64, c), op1, op2 +% c, false);
}
fn op13(s: *CpuState) void { // ADC r32, rm32
    const d = decodeModRM(s); const op1 = s.regs[d.reg]; const op2 = readRm32(s, d.mod, d.rm);
    const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    s.regs[d.reg] = op1 +% op2 +% c;
    updateFlagsArith(s, @as(i64, op1) + @as(i64, op2) + @as(i64, c), op1, op2 +% c, false);
}
fn op14(s: *CpuState) void { // ADC AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]); const c: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al +% imm +% c);
    updateFlagsArith(s, @as(i64, al) + @as(i64, imm) + @as(i64, c), al, imm +% c, false);
}
fn op15(s: *CpuState) void { // ADC EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s); const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeEaxv(s, a +% imm +% c); updateFlagsArith(s, @as(i64, a) + @as(i64, imm) + @as(i64, c), a, imm +% c, false);
}
fn op18(s: *CpuState) void { // SBB rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg); const b: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRm8Resolved(s, res.is_reg, res.addr, res.value -% op2 -% b);
    updateFlagsArith(s, @as(i64, res.value) - @as(i64, op2) - @as(i64, b), res.value, op2 +% b, true);
}
fn op19(s: *CpuState) void { // SBB rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const op2 = s.regs[d.reg]; const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRmvResolved(s, res.is_reg, res.addr, res.value -% op2 -% b);
    updateFlagsArith(s, @as(i64, res.value) - @as(i64, op2) - @as(i64, b), res.value, op2 +% b, true);
}
fn op1A(s: *CpuState) void { // SBB r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    const b: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeReg8(s, d.reg, op1 -% op2 -% b);
    updateFlagsArith(s, @as(i64, op1) - @as(i64, op2) - @as(i64, b), op1, op2 +% b, true);
}
fn op1B(s: *CpuState) void { // SBB r32, rm32
    const d = decodeModRM(s); const op1 = s.regs[d.reg]; const op2 = readRm32(s, d.mod, d.rm);
    const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    s.regs[d.reg] = op1 -% op2 -% b;
    updateFlagsArith(s, @as(i64, op1) - @as(i64, op2) - @as(i64, b), op1, op2 +% b, true);
}
fn op1C(s: *CpuState) void { // SBB AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]); const b: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al -% imm -% b);
    updateFlagsArith(s, @as(i64, al) - @as(i64, imm) - @as(i64, b), al, imm +% b, true);
}
fn op1D(s: *CpuState) void { // SBB EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s); const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeEaxv(s, a -% imm -% b); updateFlagsArith(s, @as(i64, a) - @as(i64, imm) - @as(i64, b), a, imm +% b, true);
}
fn op28(s: *CpuState) void { // SUB rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg);
    writeRm8Resolved(s, res.is_reg, res.addr, res.value -% op2);
    updateFlagsArith(s, @as(i64, res.value) - @as(i64, op2), res.value, op2, true);
}
fn op29(s: *CpuState) void { // SUB rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const op2 = s.regs[d.reg];
    writeRmvResolved(s, res.is_reg, res.addr, res.value -% op2);
    updateFlagsArith(s, @as(i64, res.value) - @as(i64, op2), res.value, op2, true);
}
fn op2A(s: *CpuState) void { // SUB r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, op1 -% op2); updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true);
}
fn op2B(s: *CpuState) void { // SUB r32, rm32
    const d = decodeModRM(s); const op1 = s.regs[d.reg]; const op2 = readRm32(s, d.mod, d.rm);
    s.regs[d.reg] = op1 -% op2; updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true);
}
fn op2C(s: *CpuState) void { // SUB AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al -% imm);
    updateFlagsArith(s, @as(i64, al) - @as(i64, imm), al, imm, true);
}
fn op2D(s: *CpuState) void { // SUB EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s);
    writeEaxv(s, a -% imm); updateFlagsArith(s, @as(i64, a) - @as(i64, imm), a, imm, true);
}
fn op38(s: *CpuState) void { // CMP rm8, r8
    const d = decodeModRM(s); const op1 = readRm8(s, d.mod, d.rm); const op2 = readReg8(s, d.reg);
    updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true);
}
fn op39(s: *CpuState) void { // CMP rmv, rv
    const d = decodeModRM(s); const op1 = readRm32(s, d.mod, d.rm); const op2 = s.regs[d.reg];
    updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true);
}
fn op3A(s: *CpuState) void { // CMP r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true);
}
fn op3B(s: *CpuState) void { // CMP r32, rm32
    const d = decodeModRM(s); const op1 = s.regs[d.reg]; const op2 = readRm32(s, d.mod, d.rm);
    updateFlagsArith(s, @as(i64, op1) - @as(i64, op2), op1, op2, true);
}
fn op3C(s: *CpuState) void { // CMP AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    updateFlagsArith(s, @as(i64, al) - @as(i64, imm), al, imm, true);
}
fn op3D(s: *CpuState) void { // CMP EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s);
    updateFlagsArith(s, @as(i64, a) - @as(i64, imm), a, imm, true);
}
fn opIncR32(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    const op1 = s.regs[r]; s.regs[r] = op1 +% 1;
    const cf = getFlag(s, CF_BIT); updateFlagsArith(s, @as(i64, op1) + 1, op1, 1, false); setFlag(s, CF_BIT, cf);
}}.f; }
fn opDecR32(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    const op1 = s.regs[r]; s.regs[r] = op1 -% 1;
    const cf = getFlag(s, CF_BIT); updateFlagsArith(s, @as(i64, op1) - 1, op1, 1, true); setFlag(s, CF_BIT, cf);
}}.f; }
fn op69(s: *CpuState) void { // IMUL r32, rm32, imm32
    const d = decodeModRM(s);
    const op1: i64 = @as(i32, @bitCast(readRm32(s, d.mod, d.rm)));
    const imm: i64 = fetchS32(s);
    const r32: u32 = @truncate(@as(u64, @bitCast(op1 * imm)));
    s.regs[d.reg] = r32;
    const ov = (op1 * imm) != @as(i64, @as(i32, @bitCast(r32)));
    setFlag(s, CF_BIT, ov); setFlag(s, OF_BIT, ov);
}
fn op6B(s: *CpuState) void { // IMUL r32, rm32, imm8
    const d = decodeModRM(s);
    const op1: i64 = @as(i32, @bitCast(readRm32(s, d.mod, d.rm)));
    const imm: i64 = fetchS8(s);
    const r32: u32 = @truncate(@as(u64, @bitCast(op1 * imm)));
    s.regs[d.reg] = r32;
    const ov = (op1 * imm) != @as(i64, @as(i32, @bitCast(r32)));
    setFlag(s, CF_BIT, ov); setFlag(s, OF_BIT, ov);
}
fn op80(s: *CpuState) void { // Group 1 byte: op rm8, imm8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const imm = fetch8(s); const op1 = res.value;
    switch (d.reg) {
        0 => { const r = op1 +% imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsArith(s, @as(i64,op1)+@as(i64,imm), op1, imm, false); },
        1 => { const r = op1 | imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r); },
        2 => { const c: u8 = if(getFlag(s,CF_BIT)) 1 else 0; const r = op1+%imm+%c; writeRm8Resolved(s,res.is_reg,res.addr,r); updateFlagsArith(s,@as(i64,op1)+@as(i64,imm)+@as(i64,c),op1,imm+%c,false); },
        3 => { const b: u8 = if(getFlag(s,CF_BIT)) 1 else 0; const r = op1-%imm-%b; writeRm8Resolved(s,res.is_reg,res.addr,r); updateFlagsArith(s,@as(i64,op1)-@as(i64,imm)-@as(i64,b),op1,imm+%b,true); },
        4 => { const r = op1 & imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r); },
        5 => { const r = op1 -% imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsArith(s, @as(i64,op1)-@as(i64,imm), op1, imm, true); },
        6 => { const r = op1 ^ imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r); },
        7 => updateFlagsArith(s, @as(i64,op1)-@as(i64,imm), op1, imm, true),
        else => {},
    }
}
fn op81(s: *CpuState) void { // Group 1: op rmv, immv
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    doGroup1(s, res.is_reg, res.addr, d.reg, res.value, fetchImm(s));
}
fn op83(s: *CpuState) void { // Group 1: op rmv, imm8 sign-ext
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    doGroup1(s, res.is_reg, res.addr, d.reg, res.value, @bitCast(@as(i32, fetchS8(s))));
}
fn op86(s: *CpuState) void { // XCHG r8, rm8
    const d = decodeModRM(s); const v1 = readReg8(s, d.reg); const v2 = readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, v2); writeRm8(s, d.mod, d.rm, v1);
}
fn op87(s: *CpuState) void { // XCHG r32, rm32
    const d = decodeModRM(s); const v1 = s.regs[d.reg]; const v2 = readRm32(s, d.mod, d.rm);
    s.regs[d.reg] = v2; writeRm32(s, d.mod, d.rm, v1);
}
fn opXchgEaxR(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    const tmp = s.regs[EAX]; s.regs[EAX] = s.regs[r]; s.regs[r] = tmp;
}}.f; }
fn op99(s: *CpuState) void { // CDQ
    s.regs[EDX] = if ((s.regs[EAX] & 0x80000000) != 0) 0xFFFFFFFF else 0;
}
fn opA8(s: *CpuState) void { // TEST AL, imm8
    const imm = fetch8(s); updateFlagsLogic(s, @as(u32, s.regs[EAX] & 0xFF & imm));
}
fn opA9(s: *CpuState) void { // TEST EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s); updateFlagsLogic(s, a & imm);
}
fn opC1(s: *CpuState) void { // Group 2: shift rmv, imm8
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    doGroup2(s, res.is_reg, res.addr, d.reg, res.value, fetch8(s) & 0x1F);
}
fn opC2(s: *CpuState) void { // RET imm16
    const ret = pop32(s); const imm = fetch16(s); s.regs[ESP] +%= imm; s.eip = ret;
}
fn opC8(s: *CpuState) void { // ENTER
    const alloc: u32 = fetch16(s); const nesting = fetch8(s) & 0x1F;
    push32(s, s.regs[EBP]);
    const frame = s.regs[ESP];
    if (nesting > 0) {
        var i: u8 = 1;
        while (i < nesting) : (i += 1) {
            s.regs[EBP] -%= 4;
            push32(s, memRead32(s, s.regs[EBP]));
        }
        push32(s, frame);
    }
    s.regs[EBP] = frame; s.regs[ESP] -%= alloc;
}
fn opC9(s: *CpuState) void { // LEAVE
    s.regs[ESP] = s.regs[EBP]; s.regs[EBP] = pop32(s);
}
fn opD1(s: *CpuState) void { // Group 2: shift rmv, 1
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    doGroup2(s, res.is_reg, res.addr, d.reg, res.value, 1);
}
fn opD3(s: *CpuState) void { // Group 2: shift rmv, CL
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    doGroup2(s, res.is_reg, res.addr, d.reg, res.value, @truncate(s.regs[ECX] & 0x1F));
}
fn opF6(s: *CpuState) void { // Group 3 byte
    const d = decodeModRM(s); const val = readRm8(s, d.mod, d.rm);
    switch (d.reg) {
        0 => updateFlagsLogic(s, @as(u32, val & fetch8(s))),
        2 => writeRm8(s, d.mod, d.rm, ~val),
        3 => {
            const r = (0 -% @as(u32, val)) & 0xFF;
            writeRm8(s, d.mod, d.rm, @truncate(r));
            setFlag(s, CF_BIT, val != 0);
            updateFlagsArith(s, -@as(i64, val), 0, val, true);
        },
        4 => { // MUL AL, rm8
            const al: u32 = s.regs[EAX] & 0xFF; const r = al * val;
            s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | (r & 0xFFFF);
            setFlag(s, CF_BIT, (r & 0xFF00) != 0); setFlag(s, OF_BIT, (r & 0xFF00) != 0);
        },
        5 => { // IMUL AL, rm8
            const al: i16 = @as(i8, @bitCast(@as(u8, @truncate(s.regs[EAX]))));
            const sv: i16 = @as(i8, @bitCast(val));
            const r: i16 = al * sv;
            s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | @as(u32, @bitCast(@as(i32, r)));
            const se: i16 = @as(i8, @truncate(r));
            setFlag(s, CF_BIT, r != se); setFlag(s, OF_BIT, r != se);
        },
        6 => { // DIV AL, rm8
            if (val == 0) { s.faulted = true; s.halted = true; return; }
            const ax = s.regs[EAX] & 0xFFFF;
            s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | ((ax % val) << 8) | (ax / val);
        },
        7 => { // IDIV AL, rm8
            const sv: i16 = @as(i8, @bitCast(val));
            if (sv == 0) { s.faulted = true; s.halted = true; return; }
            const ax: i16 = @bitCast(@as(u16, @truncate(s.regs[EAX])));
            const q: i16 = @divTrunc(ax, sv); const r2: i16 = ax - q * sv;
            s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | (@as(u32, @bitCast(@as(i32, r2))) & 0xFF) << 8 | (@as(u32, @bitCast(@as(i32, q))) & 0xFF);
        },
        else => { s.faulted = true; s.halted = true; },
    }
}
fn opF7(s: *CpuState) void { // Group 3 word/dword
    const d = decodeModRM(s); const is16 = s.op_size_ovr;
    switch (d.reg) {
        0 => { // TEST rmv, immv
            if (is16) updateFlagsLogic(s, readRmv(s,d.mod,d.rm) & fetch16(s))
            else updateFlagsLogic(s, readRm32(s,d.mod,d.rm) & fetch32(s));
        },
        2 => { // NOT
            if (is16) writeRmv(s,d.mod,d.rm, ~readRmv(s,d.mod,d.rm) & 0xFFFF)
            else writeRm32(s,d.mod,d.rm, ~readRm32(s,d.mod,d.rm));
        },
        3 => { // NEG
            if (is16) {
                const v = readRmv(s,d.mod,d.rm); const r = (0 -% v) & 0xFFFF;
                writeRmv(s,d.mod,d.rm,r); setFlag(s,CF_BIT,v!=0); updateFlagsArith(s,-@as(i64,v),0,v,false);
            } else {
                const v = readRm32(s,d.mod,d.rm); const r = 0 -% v;
                writeRm32(s,d.mod,d.rm,r); setFlag(s,CF_BIT,v!=0); updateFlagsArith(s,-@as(i64,v),0,v,true);
            }
        },
        4 => { // MUL
            if (is16) {
                const op1 = s.regs[EAX] & 0xFFFF; const op2 = readRmv(s,d.mod,d.rm) & 0xFFFF;
                const r = op1 * op2; const ov = (r >> 16) != 0;
                s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | (r & 0xFFFF);
                s.regs[EDX] = (s.regs[EDX] & 0xFFFF0000) | ((r >> 16) & 0xFFFF);
                setFlag(s,CF_BIT,ov); setFlag(s,OF_BIT,ov);
            } else {
                const op1: u64 = s.regs[EAX]; const op2: u64 = readRm32(s,d.mod,d.rm);
                const r: u64 = op1 * op2;
                s.regs[EAX] = @truncate(r); s.regs[EDX] = @truncate(r >> 32);
                const ov = s.regs[EDX] != 0; setFlag(s,CF_BIT,ov); setFlag(s,OF_BIT,ov);
            }
        },
        5 => { // IMUL
            if (is16) {
                const op1: i32 = @as(i16, @bitCast(@as(u16, @truncate(s.regs[EAX]))));
                const op2_r = readRmv(s,d.mod,d.rm) & 0xFFFF;
                const op2: i32 = @as(i16, @bitCast(@as(u16, @truncate(op2_r))));
                const r: i32 = op1 * op2;
                s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | (@as(u32, @bitCast(r)) & 0xFFFF);
                s.regs[EDX] = (s.regs[EDX] & 0xFFFF0000) | ((@as(u32, @bitCast(r)) >> 16) & 0xFFFF);
                const se: u32 = if ((r & 0x8000) != 0) 0xFFFF else 0;
                setFlag(s,CF_BIT,((@as(u32,@bitCast(r))>>16)&0xFFFF)!=se); setFlag(s,OF_BIT,((@as(u32,@bitCast(r))>>16)&0xFFFF)!=se);
            } else {
                const op1: i64 = @as(i32, @bitCast(s.regs[EAX]));
                const op2: i64 = @as(i32, @bitCast(readRm32(s,d.mod,d.rm)));
                const r: i64 = op1 * op2;
                s.regs[EAX] = @truncate(@as(u64, @bitCast(r)));
                s.regs[EDX] = @truncate(@as(u64, @bitCast(r)) >> 32);
                const se: u32 = if ((s.regs[EAX] & 0x80000000) != 0) 0xFFFFFFFF else 0;
                setFlag(s,CF_BIT,s.regs[EDX]!=se); setFlag(s,OF_BIT,s.regs[EDX]!=se);
            }
        },
        6 => { // DIV
            if (is16) {
                const div = readRmv(s,d.mod,d.rm) & 0xFFFF;
                if (div == 0) { s.faulted=true; s.halted=true; return; }
                const dvd: u32 = ((s.regs[EDX] & 0xFFFF) << 16) | (s.regs[EAX] & 0xFFFF);
                s.regs[EAX] = (s.regs[EAX]&0xFFFF0000)|(dvd/div)&0xFFFF;
                s.regs[EDX] = (s.regs[EDX]&0xFFFF0000)|(dvd%div)&0xFFFF;
            } else {
                const div: u64 = readRm32(s,d.mod,d.rm);
                if (div == 0) { s.faulted=true; s.halted=true; return; }
                const dvd: u64 = (@as(u64,s.regs[EDX]) << 32) | @as(u64,s.regs[EAX]);
                const q = dvd / div;
                if (q > 0xFFFFFFFF) { s.faulted=true; s.halted=true; return; }
                s.regs[EAX] = @truncate(q); s.regs[EDX] = @truncate(dvd % div);
            }
        },
        7 => { // IDIV
            if (is16) {
                const raw = readRmv(s,d.mod,d.rm) & 0xFFFF;
                const div: i32 = @as(i16, @bitCast(@as(u16, @truncate(raw))));
                if (div == 0) { s.faulted=true; s.halted=true; return; }
                const dvd_raw: u32 = ((s.regs[EDX] & 0xFFFF) << 16) | (s.regs[EAX] & 0xFFFF);
                const dvd: i32 = @bitCast(dvd_raw);
                const q: i32 = @divTrunc(dvd, div); const r2: i32 = dvd - q * div;
                s.regs[EAX] = (s.regs[EAX]&0xFFFF0000)|(@as(u32,@bitCast(q))&0xFFFF);
                s.regs[EDX] = (s.regs[EDX]&0xFFFF0000)|(@as(u32,@bitCast(r2))&0xFFFF);
            } else {
                const div: i64 = @as(i32, @bitCast(readRm32(s,d.mod,d.rm)));
                if (div == 0) { s.faulted=true; s.halted=true; return; }
                const edx_s: i64 = @as(i32, @bitCast(s.regs[EDX]));
                const dvd: i64 = (edx_s << 32) | @as(i64, @intCast(s.regs[EAX]));
                const q: i64 = @divTrunc(dvd, div); const r2: i64 = dvd - q * div;
                s.regs[EAX] = @truncate(@as(u64, @bitCast(q)));
                s.regs[EDX] = @truncate(@as(u64, @bitCast(r2)));
            }
        },
        else => { s.faulted = true; s.halted = true; },
    }
}

// ─── Logic opcodes ────────────────────────────────────────────────────────────
fn op08(s: *CpuState) void { // OR rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const r = res.value | readReg8(s, d.reg); writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r);
}
fn op09(s: *CpuState) void { // OR rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const r = res.value | s.regs[d.reg]; writeRmvResolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r);
}
fn op0A(s: *CpuState) void { // OR r8, rm8
    const d = decodeModRM(s); const r = readReg8(s, d.reg) | readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, r); updateFlagsLogic(s, r);
}
fn op0B(s: *CpuState) void { // OR r32, rm32
    const d = decodeModRM(s); const r = s.regs[d.reg] | readRm32(s, d.mod, d.rm);
    s.regs[d.reg] = r; updateFlagsLogic(s, r);
}
fn op0C(s: *CpuState) void { // OR AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    const r = al | imm; s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | r; updateFlagsLogic(s, r);
}
fn op0D(s: *CpuState) void { // OR EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s); const r = a | imm;
    updateFlagsLogic(s, r); writeEaxv(s, r);
}
fn op20(s: *CpuState) void { // AND rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const r = res.value & readReg8(s, d.reg); writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r);
}
fn op21(s: *CpuState) void { // AND rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const r = res.value & s.regs[d.reg]; writeRmvResolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r);
}
fn op22(s: *CpuState) void { // AND r8, rm8
    const d = decodeModRM(s); const r = readReg8(s, d.reg) & readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, r); updateFlagsLogic(s, r);
}
fn op23(s: *CpuState) void { // AND r32, rm32
    const d = decodeModRM(s); const r = s.regs[d.reg] & readRm32(s, d.mod, d.rm);
    s.regs[d.reg] = r; updateFlagsLogic(s, r);
}
fn op24(s: *CpuState) void { // AND AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    const r = al & imm; s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | r; updateFlagsLogic(s, r);
}
fn op25(s: *CpuState) void { // AND EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s); const r = a & imm;
    updateFlagsLogic(s, r); writeEaxv(s, r);
}
fn op30(s: *CpuState) void { // XOR rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const r = res.value ^ readReg8(s, d.reg); writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r);
}
fn op31(s: *CpuState) void { // XOR rmv, rv
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    const r = res.value ^ s.regs[d.reg]; writeRmvResolved(s, res.is_reg, res.addr, r); updateFlagsLogic(s, r);
}
fn op32(s: *CpuState) void { // XOR r8, rm8
    const d = decodeModRM(s); const r = readReg8(s, d.reg) ^ readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, r); updateFlagsLogic(s, r);
}
fn op33(s: *CpuState) void { // XOR r32, rm32
    const d = decodeModRM(s); const r = s.regs[d.reg] ^ readRm32(s, d.mod, d.rm);
    s.regs[d.reg] = r; updateFlagsLogic(s, r);
}
fn op35(s: *CpuState) void { // XOR EAX/AX, immv
    const a = readEaxv(s); const imm = fetchImm(s); const r = a ^ imm;
    updateFlagsLogic(s, r); writeEaxv(s, r);
}
fn op84(s: *CpuState) void { // TEST rm8, r8
    const d = decodeModRM(s);
    updateFlagsLogic(s, @as(u32, readRm8(s, d.mod, d.rm) & readReg8(s, d.reg)));
}
fn op85(s: *CpuState) void { // TEST rmv, rv
    const d = decodeModRM(s); updateFlagsLogic(s, readRm32(s, d.mod, d.rm) & s.regs[d.reg]);
}

// ─── Data movement opcodes ────────────────────────────────────────────────────
fn op88(s: *CpuState) void { // MOV rm8, r8
    const d = decodeModRM(s); writeRm8(s, d.mod, d.rm, readReg8(s, d.reg));
}
fn op89(s: *CpuState) void { // MOV rmv, rv
    const d = decodeModRM(s);
    if (s.op_size_ovr) writeRmv(s, d.mod, d.rm, s.regs[d.reg] & 0xFFFF)
    else writeRm32(s, d.mod, d.rm, s.regs[d.reg]);
}
fn op8A(s: *CpuState) void { // MOV r8, rm8
    const d = decodeModRM(s); writeReg8(s, d.reg, readRm8(s, d.mod, d.rm));
}
fn op8B(s: *CpuState) void { // MOV rv, rmv
    const d = decodeModRM(s);
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (readRmv(s, d.mod, d.rm) & 0xFFFF)
    else s.regs[d.reg] = readRm32(s, d.mod, d.rm);
}
fn op8D(s: *CpuState) void { // LEA r32, rm
    const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm); s.regs[d.reg] = r.addr;
}
fn opA0(s: *CpuState) void { // MOV AL, [disp32]
    const addr = applySegOvr(s, fetch32(s));
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | memRead8(s, addr);
}
fn opA1(s: *CpuState) void { // MOV EAX, [disp32]
    s.regs[EAX] = memRead32(s, applySegOvr(s, fetch32(s)));
}
fn opA2(s: *CpuState) void { // MOV [disp32], AL
    memWrite8(s, applySegOvr(s, fetch32(s)), @truncate(s.regs[EAX]));
}
fn opA3(s: *CpuState) void { // MOV [disp32], EAX
    memWrite32(s, applySegOvr(s, fetch32(s)), s.regs[EAX]);
}
fn opMovR8Imm(comptime r: u8) OpFn { return struct { fn f(s: *CpuState) void {
    const imm = fetch8(s);
    if (r < 4) s.regs[r] = (s.regs[r] & 0xFFFFFF00) | @as(u32, imm)
    else s.regs[r - 4] = (s.regs[r - 4] & 0xFFFF00FF) | (@as(u32, imm) << 8);
}}.f; }
fn opMovR32Imm(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    s.regs[r] = fetch32(s);
}}.f; }
fn opC6(s: *CpuState) void { // MOV rm8, imm8
    const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm); const imm = fetch8(s);
    if (r.is_reg) {
        const ri = r.addr; s.regs[ri] = (s.regs[ri] & 0xFFFFFF00) | @as(u32, imm);
    } else memWrite8(s, applySegOvr(s, r.addr), imm);
}
fn opC7(s: *CpuState) void { // MOV rmv, immv
    const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
    if (s.op_size_ovr) {
        const imm = fetch16(s);
        if (r.is_reg) s.regs[r.addr] = (s.regs[r.addr] & 0xFFFF0000) | @as(u32, imm)
        else memWrite16(s, applySegOvr(s, r.addr), imm);
    } else {
        const imm = fetch32(s);
        if (r.is_reg) s.regs[r.addr] = imm else memWrite32(s, applySegOvr(s, r.addr), imm);
    }
}
fn opC4(s: *CpuState) void { // LES r32, m (flat: load offset, ignore seg)
    const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
    if (!r.is_reg) s.regs[d.reg] = memRead32(s, applySegOvr(s, r.addr));
}
fn opC5(s: *CpuState) void { // LDS r32, m (flat: load offset, ignore seg)
    const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
    if (!r.is_reg) s.regs[d.reg] = memRead32(s, applySegOvr(s, r.addr));
}
fn op0E(s: *CpuState) void { push32(s, 0x1B); }  // PUSH CS (flat CS=0x1B)
fn op06(s: *CpuState) void { push32(s, 0x23); }  // PUSH ES
fn op16(s: *CpuState) void { push32(s, 0x23); }  // PUSH SS
fn op1E(s: *CpuState) void { push32(s, 0x23); }  // PUSH DS
fn op07(s: *CpuState) void { _ = pop32(s); }      // POP ES
fn op17(s: *CpuState) void { _ = pop32(s); }      // POP SS
fn op1F(s: *CpuState) void { _ = pop32(s); }      // POP DS

// ─── Control flow opcodes ─────────────────────────────────────────────────────
fn opC3(s: *CpuState) void { s.eip = pop32(s); }  // RET
fn opE8(s: *CpuState) void { // CALL rel32
    const rel: i32 = fetchS32(s); const target = s.eip +% @as(u32, @bitCast(rel));
    push32(s, s.eip); s.eip = target;
}
fn opE9(s: *CpuState) void { // JMP rel32
    const rel: i32 = fetchS32(s); s.eip = s.eip +% @as(u32, @bitCast(rel));
}
fn opEB(s: *CpuState) void { // JMP rel8
    const rel: i8 = fetchS8(s); s.eip = s.eip +% @as(u32, @bitCast(@as(i32, rel)));
}
fn opJcc8(comptime cond: u8) OpFn { return struct { fn f(s: *CpuState) void {
    const rel: i8 = fetchS8(s);
    if (evalCond(s, cond)) s.eip = s.eip +% @as(u32, @bitCast(@as(i32, rel)));
}}.f; }
fn opJccNear(comptime cond: u8) OpFn { return struct { fn f(s: *CpuState) void {
    const rel: i32 = fetchS32(s);
    if (evalCond(s, cond)) s.eip = s.eip +% @as(u32, @bitCast(rel));
}}.f; }
fn opE0(s: *CpuState) void { // LOOPNE
    const rel: i8 = fetchS8(s); s.regs[ECX] -%= 1;
    if (s.regs[ECX] != 0 and !getFlag(s, ZF_BIT)) s.eip = s.eip +% @as(u32, @bitCast(@as(i32, rel)));
}
fn opE1(s: *CpuState) void { // LOOPE
    const rel: i8 = fetchS8(s); s.regs[ECX] -%= 1;
    if (s.regs[ECX] != 0 and getFlag(s, ZF_BIT)) s.eip = s.eip +% @as(u32, @bitCast(@as(i32, rel)));
}
fn opE2(s: *CpuState) void { // LOOP
    const rel: i8 = fetchS8(s); s.regs[ECX] -%= 1;
    if (s.regs[ECX] != 0) s.eip = s.eip +% @as(u32, @bitCast(@as(i32, rel)));
}
fn opE3(s: *CpuState) void { // JECXZ
    const rel: i8 = fetchS8(s);
    if (s.regs[ECX] == 0) s.eip = s.eip +% @as(u32, @bitCast(@as(i32, rel)));
}

// ─── Stack opcodes ────────────────────────────────────────────────────────────
fn opPushR32(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void { push32(s, s.regs[r]); }}.f; }
fn opPopR32(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void { s.regs[r] = pop32(s); }}.f; }
fn op60(s: *CpuState) void { // PUSHAD
    const orig_esp = s.regs[ESP];
    push32(s, s.regs[EAX]); push32(s, s.regs[ECX]); push32(s, s.regs[EDX]); push32(s, s.regs[EBX]);
    push32(s, orig_esp); push32(s, s.regs[EBP]); push32(s, s.regs[ESI]); push32(s, s.regs[EDI]);
}
fn op61(s: *CpuState) void { // POPAD
    s.regs[EDI] = pop32(s); s.regs[ESI] = pop32(s); s.regs[EBP] = pop32(s); _ = pop32(s);
    s.regs[EBX] = pop32(s); s.regs[EDX] = pop32(s); s.regs[ECX] = pop32(s); s.regs[EAX] = pop32(s);
}
fn op68(s: *CpuState) void { push32(s, fetch32(s)); }  // PUSH imm32
fn op6A(s: *CpuState) void { push32(s, @bitCast(@as(i32, fetchS8(s)))); }  // PUSH imm8 sign-ext

// ─── Misc opcodes ─────────────────────────────────────────────────────────────
fn opNop(_: *CpuState) void {}   // NOP
fn opF4(s: *CpuState) void { s.halted = true; }  // HLT
fn opFC(s: *CpuState) void { setFlag(s, DF_BIT, false); }  // CLD
fn opFD(s: *CpuState) void { setFlag(s, DF_BIT, true); }   // STD
fn opF8(s: *CpuState) void { setFlag(s, CF_BIT, false); }  // CLC
fn opF9(s: *CpuState) void { setFlag(s, CF_BIT, true); }   // STC
fn opF5(s: *CpuState) void { setFlag(s, CF_BIT, !getFlag(s, CF_BIT)); }  // CMC
fn op9B(_: *CpuState) void {}   // WAIT/FWAIT — no-op
fn op9C(s: *CpuState) void { push32(s, s.eflags & 0xFCFFFF); }  // PUSHFD
fn op9D(s: *CpuState) void { s.eflags = pop32(s) & 0xFCFFFF; }  // POPFD
fn op9E(s: *CpuState) void { // SAHF
    const ah: u32 = (s.regs[EAX] >> 8) & 0xFF;
    s.eflags = (s.eflags & ~@as(u32, 0xD5)) | (ah & 0xD5);
}
fn op9F(s: *CpuState) void { // LAHF
    const ah: u32 = s.eflags & 0xD5;
    s.regs[EAX] = (s.regs[EAX] & 0xFFFF00FF) | (ah << 8);
}
fn op98(s: *CpuState) void { // CWDE / CBW
    if (s.op_size_ovr) { // CBW: sign-extend AL → AX
        const al: u8 = @truncate(s.regs[EAX]);
        const ax: u16 = @bitCast(@as(i16, @as(i8, @bitCast(al))));
        s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | @as(u32, ax);
    } else { // CWDE: sign-extend AX → EAX
        const ax: u16 = @truncate(s.regs[EAX]);
        s.regs[EAX] = @bitCast(@as(i32, @as(i16, @bitCast(ax))));
    }
}
fn opCC(s: *CpuState) void { // INT3
    if (s.int_handler) |h| h(s, 3)
    else { s.faulted = true; s.halted = true; }
}
fn opCD(s: *CpuState) void { // INT imm8
    const n = fetch8(s);
    if (s.int_handler) |h| h(s, n)
    else { s.faulted = true; s.halted = true; }
}

// ─── String opcodes ───────────────────────────────────────────────────────────
inline fn strDir(s: *CpuState) i32 { return if (getFlag(s, DF_BIT)) -1 else 1; }

fn opAA(s: *CpuState) void { // STOSB
    if (s.rep_prefix == REP_REP) {
        while (s.regs[ECX] != 0) {
            memWrite8(s, s.regs[EDI], @truncate(s.regs[EAX]));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1;
        }
    } else {
        memWrite8(s, s.regs[EDI], @truncate(s.regs[EAX]));
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
    }
}
fn opAB(s: *CpuState) void { // STOSD/STOSW
    const wide = !s.op_size_ovr;
    const step: i32 = if (wide) 4 else 2;
    const d: i32 = strDir(s) * step;
    if (s.rep_prefix == REP_REP) {
        while (s.regs[ECX] != 0) {
            if (wide) memWrite32(s, s.regs[EDI], s.regs[EAX]) else memWrite16(s, s.regs[EDI], @truncate(s.regs[EAX]));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
            s.regs[ECX] -%= 1;
        }
    } else {
        if (wide) memWrite32(s, s.regs[EDI], s.regs[EAX]) else memWrite16(s, s.regs[EDI], @truncate(s.regs[EAX]));
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
    }
}
fn opA4(s: *CpuState) void { // MOVSB
    if (s.rep_prefix == REP_REP) {
        while (s.regs[ECX] != 0) {
            memWrite8(s, s.regs[EDI], memRead8(s, s.regs[ESI]));
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1;
        }
    } else {
        memWrite8(s, s.regs[EDI], memRead8(s, s.regs[ESI]));
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
    }
}
fn opA5(s: *CpuState) void { // MOVSD/MOVSW
    const wide = !s.op_size_ovr; const step: i32 = if (wide) 4 else 2; const d: i32 = strDir(s) * step;
    if (s.rep_prefix == REP_REP) {
        while (s.regs[ECX] != 0) {
            if (wide) memWrite32(s, s.regs[EDI], memRead32(s, s.regs[ESI])) else memWrite16(s, s.regs[EDI], memRead16(s, s.regs[ESI]));
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + d);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
            s.regs[ECX] -%= 1;
        }
    } else {
        if (wide) memWrite32(s, s.regs[EDI], memRead32(s, s.regs[ESI])) else memWrite16(s, s.regs[EDI], memRead16(s, s.regs[ESI]));
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + d);
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
    }
}
fn opAC(s: *CpuState) void { // LODSB
    if (s.rep_prefix == REP_REP) {
        while (s.regs[ECX] != 0) {
            s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | memRead8(s, s.regs[ESI]);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
            s.regs[ECX] -%= 1;
        }
    } else {
        s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | memRead8(s, s.regs[ESI]);
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
    }
}
fn opAD(s: *CpuState) void { // LODSD/LODSW
    const wide = !s.op_size_ovr; const step: i32 = if (wide) 4 else 2; const d: i32 = strDir(s) * step;
    if (s.rep_prefix == REP_REP) {
        while (s.regs[ECX] != 0) {
            if (wide) s.regs[EAX] = memRead32(s, s.regs[ESI]) else s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | memRead16(s, s.regs[ESI]);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + d);
            s.regs[ECX] -%= 1;
        }
    } else {
        if (wide) s.regs[EAX] = memRead32(s, s.regs[ESI]) else s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | memRead16(s, s.regs[ESI]);
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + d);
    }
}
fn opAE(s: *CpuState) void { // SCASB
    const rep = s.rep_prefix;
    if (rep == REP_REP) {
        while (s.regs[ECX] != 0) {
            const v = memRead8(s, s.regs[EDI]); const al: u8 = @truncate(s.regs[EAX]);
            updateFlagsArith(s, @as(i64, al) - @as(i64, v), al, v, true);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1;
            if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const v = memRead8(s, s.regs[EDI]); const al: u8 = @truncate(s.regs[EAX]);
            updateFlagsArith(s, @as(i64, al) - @as(i64, v), al, v, true);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1;
            if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const v = memRead8(s, s.regs[EDI]); const al: u8 = @truncate(s.regs[EAX]);
        updateFlagsArith(s, @as(i64, al) - @as(i64, v), al, v, true);
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
    }
}
fn opAF(s: *CpuState) void { // SCASD/SCASW
    const wide = !s.op_size_ovr; const step: i32 = if (wide) 4 else 2; const d: i32 = strDir(s) * step;
    const rep = s.rep_prefix;
    const acc = if (wide) s.regs[EAX] else s.regs[EAX] & 0xFFFF;
    if (rep == REP_REP) {
        while (s.regs[ECX] != 0) {
            const v: u32 = if (wide) memRead32(s, s.regs[EDI]) else @as(u32, memRead16(s, s.regs[EDI]));
            updateFlagsArith(s, @as(i64, acc) - @as(i64, v), acc, v, true);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
            s.regs[ECX] -%= 1;
            if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const v: u32 = if (wide) memRead32(s, s.regs[EDI]) else @as(u32, memRead16(s, s.regs[EDI]));
            updateFlagsArith(s, @as(i64, acc) - @as(i64, v), acc, v, true);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
            s.regs[ECX] -%= 1;
            if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const v: u32 = if (wide) memRead32(s, s.regs[EDI]) else @as(u32, memRead16(s, s.regs[EDI]));
        updateFlagsArith(s, @as(i64, acc) - @as(i64, v), acc, v, true);
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
    }
}
fn opA6(s: *CpuState) void { // CMPSB
    const rep = s.rep_prefix;
    if (rep == REP_REP) {
        while (s.regs[ECX] != 0) {
            const src = memRead8(s, s.regs[ESI]); const dst = memRead8(s, s.regs[EDI]);
            updateFlagsArith(s, @as(i64, src) - @as(i64, dst), src, dst, true);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1; if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const src = memRead8(s, s.regs[ESI]); const dst = memRead8(s, s.regs[EDI]);
            updateFlagsArith(s, @as(i64, src) - @as(i64, dst), src, dst, true);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1; if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const src = memRead8(s, s.regs[ESI]); const dst = memRead8(s, s.regs[EDI]);
        updateFlagsArith(s, @as(i64, src) - @as(i64, dst), src, dst, true);
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
    }
}
fn opA7(s: *CpuState) void { // CMPSD
    const di: i32 = strDir(s) * 4; const rep = s.rep_prefix;
    if (rep == REP_REP) {
        while (s.regs[ECX] != 0) {
            const src = memRead32(s, s.regs[ESI]); const dst = memRead32(s, s.regs[EDI]);
            updateFlagsArith(s, @as(i64, src) - @as(i64, dst), src, dst, true);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + di);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + di);
            s.regs[ECX] -%= 1; if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const src = memRead32(s, s.regs[ESI]); const dst = memRead32(s, s.regs[EDI]);
            updateFlagsArith(s, @as(i64, src) - @as(i64, dst), src, dst, true);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + di);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + di);
            s.regs[ECX] -%= 1; if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const src = memRead32(s, s.regs[ESI]); const dst = memRead32(s, s.regs[EDI]);
        updateFlagsArith(s, @as(i64, src) - @as(i64, dst), src, dst, true);
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + di);
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + di);
    }
}

// ─── Group 4/5 opcodes ────────────────────────────────────────────────────────
fn opFE(s: *CpuState) void { // Group 4: INC/DEC rm8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    if (d.reg == 0) {
        writeRm8Resolved(s, res.is_reg, res.addr, res.value +% 1);
        const cf = getFlag(s, CF_BIT); updateFlagsArith(s, @as(i64, res.value) + 1, res.value, 1, false); setFlag(s, CF_BIT, cf);
    } else if (d.reg == 1) {
        writeRm8Resolved(s, res.is_reg, res.addr, res.value -% 1);
        const cf = getFlag(s, CF_BIT); updateFlagsArith(s, @as(i64, res.value) - 1, res.value, 1, true); setFlag(s, CF_BIT, cf);
    } else { s.faulted = true; s.halted = true; }
}
fn opFF(s: *CpuState) void { // Group 5: INC/DEC/CALL/JMP/PUSH rm32
    const d = decodeModRM(s); const res = readRm32Resolved(s, d.mod, d.rm);
    switch (d.reg) {
        0 => { writeRmvResolved(s, res.is_reg, res.addr, res.value +% 1); const cf = getFlag(s, CF_BIT); updateFlagsArith(s, @as(i64, res.value) + 1, res.value, 1, false); setFlag(s, CF_BIT, cf); },
        1 => { writeRmvResolved(s, res.is_reg, res.addr, res.value -% 1); const cf = getFlag(s, CF_BIT); updateFlagsArith(s, @as(i64, res.value) - 1, res.value, 1, true); setFlag(s, CF_BIT, cf); },
        2 => { push32(s, s.eip); s.eip = res.value; },  // CALL rm32
        4 => { s.eip = res.value; },                     // JMP rm32
        6 => { push32(s, res.value); },                  // PUSH rm32
        3, 5 => {},  // CALL/JMP far — not needed in flat model
        else => {},
    }
}

// ─── Two-byte opcodes (0x0F prefix) ──────────────────────────────────────────
fn op0F(s: *CpuState) void {
    const op2 = fetch8(s);
    switch (op2) {
        0xB6 => { // MOVZX r32, rm8
            const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
            s.regs[d.reg] = if (r.is_reg) s.regs[r.addr] & 0xFF else memRead8(s, applySegOvr(s, r.addr));
        },
        0xB7 => { // MOVZX r32, rm16
            const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
            s.regs[d.reg] = if (r.is_reg) s.regs[r.addr] & 0xFFFF else @as(u32, memRead16(s, applySegOvr(s, r.addr)));
        },
        0xBE => { // MOVSX r32, rm8
            const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
            const v: u8 = if (r.is_reg) @truncate(s.regs[r.addr]) else memRead8(s, applySegOvr(s, r.addr));
            s.regs[d.reg] = @bitCast(@as(i32, @as(i8, @bitCast(v))));
        },
        0xBF => { // MOVSX r32, rm16
            const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
            const v: u16 = if (r.is_reg) @truncate(s.regs[r.addr]) else memRead16(s, applySegOvr(s, r.addr));
            s.regs[d.reg] = @bitCast(@as(i32, @as(i16, @bitCast(v))));
        },
        0xAF => { // IMUL r32, rm32
            const d = decodeModRM(s);
            const imul_op1: i64 = @as(i32, @bitCast(s.regs[d.reg]));
            const imul_op2: i64 = @as(i32, @bitCast(readRm32(s, d.mod, d.rm)));
            const r32: u32 = @truncate(@as(u64, @bitCast(imul_op1 * imul_op2)));
            s.regs[d.reg] = r32;
            const ov = (imul_op1 * imul_op2) != @as(i64, @as(i32, @bitCast(r32)));
            setFlag(s, CF_BIT, ov); setFlag(s, OF_BIT, ov);
        },
        0x90...0x9F => { // SETcc rm8
            const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm);
            const v: u8 = if (evalCond(s, op2 & 0xF)) 1 else 0;
            if (r.is_reg) s.regs[r.addr] = (s.regs[r.addr] & 0xFFFFFF00) | v
            else memWrite8(s, applySegOvr(s, r.addr), v);
        },
        0x80...0x8F => { // Jcc rel32 (near)
            const rel = fetchS32(s);
            if (evalCond(s, op2 & 0xF)) s.eip = s.eip +% @as(u32, @bitCast(rel));
        },
        0xC1 => { // XADD rm32, r32
            const d = decodeModRM(s); const dst = readRm32(s, d.mod, d.rm); const src = s.regs[d.reg];
            s.regs[d.reg] = dst; writeRm32(s, d.mod, d.rm, dst +% src);
            updateFlagsArith(s, @as(i64, dst) + @as(i64, src), dst, src, false);
        },
        0xBD => { // BSR r32, rm32
            const d = decodeModRM(s); const v = readRm32(s, d.mod, d.rm);
            if (v == 0) setFlag(s, ZF_BIT, true)
            else { setFlag(s, ZF_BIT, false); s.regs[d.reg] = 31 - @clz(v); }
        },
        0xBC => { // BSF r32, rm32
            const d = decodeModRM(s); const v = readRm32(s, d.mod, d.rm);
            if (v == 0) setFlag(s, ZF_BIT, true)
            else { setFlag(s, ZF_BIT, false); s.regs[d.reg] = @ctz(v); }
        },
        0x40...0x4F => { // CMOVcc r32, rm32
            const d = decodeModRM(s); const v = readRm32(s, d.mod, d.rm);
            if (evalCond(s, op2 & 0xF)) s.regs[d.reg] = v;
        },
        0xC8...0xCF => { // BSWAP r32
            const r: u8 = op2 & 7; const v = s.regs[r];
            s.regs[r] = ((v & 0xFF) << 24) | (((v >> 8) & 0xFF) << 16) | (((v >> 16) & 0xFF) << 8) | (v >> 24);
        },
        0xA3 => { // BT rm32, r32
            const d = decodeModRM(s); const bit: u5 = @truncate(s.regs[d.reg] & 0x1F);
            setFlag(s, CF_BIT, ((readRm32(s, d.mod, d.rm) >> bit) & 1) != 0);
        },
        0xBA => { // Group 8: BT/BTS/BTR/BTC rm32, imm8
            const d = decodeModRM(s); const bit: u5 = @truncate(fetch8(s) & 0x1F);
            const v = readRm32(s, d.mod, d.rm); setFlag(s, CF_BIT, ((v >> bit) & 1) != 0);
            switch (d.reg) {
                5 => writeRm32(s, d.mod, d.rm, v | (@as(u32, 1) << bit)),
                6 => writeRm32(s, d.mod, d.rm, v & ~(@as(u32, 1) << bit)),
                7 => writeRm32(s, d.mod, d.rm, (v ^ (@as(u32, 1) << bit))),
                else => {},
            }
        },
        0xA2 => { // CPUID
            const leaf = s.regs[EAX];
            switch (leaf) {
                0 => { s.regs[EAX] = 1; s.regs[EBX] = 0x756E6547; s.regs[EDX] = 0x49656E69; s.regs[ECX] = 0x6C65746E; },
                1 => { s.regs[EAX] = 0x00000600; s.regs[EBX] = 0; s.regs[ECX] = 0; s.regs[EDX] = 0x00008001; },
                else => { s.regs[EAX] = 0; s.regs[EBX] = 0; s.regs[ECX] = 0; s.regs[EDX] = 0; },
            }
        },
        else => { s.faulted = true; s.halted = true; },
    }
}

// ─── FPU opcodes (0xD8–0xDF) ─────────────────────────────────────────────────
const FPU_CONSTS = [7]f64{ 1.0, 3.3219280948873626, 1.4426950408889634,
    std.math.pi, 0.3010299956639812, std.math.ln2, 0.0 };

fn opD8(s: *CpuState) void { // float32 ops
    const d = decodeModRM(s);
    if (d.mod == 3) {
        const st0 = fpuGet(s, 0); const sti = fpuGet(s, d.rm);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + sti), 1 => fpuSet(s, 0, st0 * sti),
            2 => fpuCompare(s, st0, sti), 3 => { fpuCompare(s, st0, sti); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - sti), 5 => fpuSet(s, 0, sti - st0),
            6 => fpuSet(s, 0, st0 / sti), 7 => fpuSet(s, 0, sti / st0),
            else => {},
        }
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        const val: f64 = readFloat(s, addr); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}
fn opD9(s: *CpuState) void { // FLD/FST/FSTP/constants/misc
    const d = decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => fpuPush(s, fpuGet(s, d.rm)),  // FLD ST(i)
            1 => { const t = fpuGet(s, 0); fpuSet(s, 0, fpuGet(s, d.rm)); fpuSet(s, d.rm, t); },  // FXCH
            2 => {},  // FNOP
            3 => { fpuSet(s, d.rm, fpuGet(s, 0)); _ = fpuPop(s); },  // FSTP ST(i)
            4 => switch (d.rm) {
                0 => fpuSet(s, 0, -fpuGet(s, 0)),  // FCHS
                1 => fpuSet(s, 0, @abs(fpuGet(s, 0))),  // FABS
                4 => fpuCompare(s, fpuGet(s, 0), 0.0),  // FTST
                5 => { s.fpu_status_word &= ~@as(u16, 0x4700); if (fpuGet(s, 0) < 0) s.fpu_status_word |= 0x0200; },  // FXAM
                else => {},
            },
            5 => if (d.rm < 7) fpuPush(s, FPU_CONSTS[d.rm]),  // FLD constants
            6 => switch (d.rm) {
                0 => fpuSet(s, 0, std.math.exp2(fpuGet(s, 0)) - 1.0),  // F2XM1
                1 => { const x = fpuGet(s, 0); const y = fpuGet(s, 1); _ = fpuPop(s); fpuSet(s, 0, y * std.math.log2(x)); },  // FYL2X
                5 => { fpuSet(s, 0, @rem(fpuGet(s, 0), fpuGet(s, 1))); s.fpu_status_word &= ~@as(u16, 0x0400); },  // FPREM1
                6 => { s.fpu_top = (s.fpu_top -% 1) & 7; s.fpu_status_word = (s.fpu_status_word & ~@as(u16,0x3800)) | @as(u16, @truncate((s.fpu_top & 7) << 11)); },  // FDECSTP
                7 => { s.fpu_top = (s.fpu_top +% 1) & 7; s.fpu_status_word = (s.fpu_status_word & ~@as(u16,0x3800)) | @as(u16, @truncate((s.fpu_top & 7) << 11)); },  // FINCSTP
                else => {},
            },
            7 => switch (d.rm) {
                0 => fpuSet(s, 0, @rem(fpuGet(s, 0), fpuGet(s, 1))),  // FPREM
                2 => fpuSet(s, 0, @sqrt(fpuGet(s, 0))),  // FSQRT
                3 => { const v = fpuGet(s, 0); fpuSet(s, 0, @sin(v)); fpuPush(s, @cos(v)); },  // FSINCOS
                4 => fpuSet(s, 0, @round(fpuGet(s, 0))),  // FRNDINT
                5 => { const sc: f64 = @floatFromInt(@as(i64, @intFromFloat(@trunc(fpuGet(s, 1))))); fpuSet(s, 0, fpuGet(s, 0) * std.math.exp2(sc)); },  // FSCALE
                6 => fpuSet(s, 0, @sin(fpuGet(s, 0))),  // FSIN
                7 => fpuSet(s, 0, @cos(fpuGet(s, 0))),  // FCOS
                else => {},
            },
            else => {},
        }
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, readFloat(s, addr)),  // FLD m32
            2 => writeFloat(s, addr, @floatCast(fpuGet(s, 0))),  // FST m32
            3 => { writeFloat(s, addr, @floatCast(fpuGet(s, 0))); _ = fpuPop(s); },  // FSTP m32
            4 => {},  // FLDENV NOP
            5 => s.fpu_control_word = memRead16(s, addr),  // FLDCW
            6 => {},  // FNSTENV NOP
            7 => memWrite16(s, addr, s.fpu_control_word),  // FNSTCW
            else => {},
        }
    }
}
fn opDA(s: *CpuState) void { // int32 ops / FCMOV
    const d = decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => { if (getFlag(s, CF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVB
            1 => { if (getFlag(s, ZF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVE
            2 => { if (getFlag(s, CF_BIT) or getFlag(s, ZF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVBE
            3 => fpuSet(s, 0, fpuGet(s, d.rm)),  // FCMOVU
            5 => if (d.rm == 1) { fpuCompare(s, fpuGet(s, 0), fpuGet(s, 1)); _ = fpuPop(s); _ = fpuPop(s); },  // FUCOMPP
            else => {},
        }
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        const val: f64 = @floatFromInt(memReadS32(s, addr)); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}
fn opDB(s: *CpuState) void { // FILD/FISTP int32, FCLEX/FINIT, FUCOMI
    const d = decodeModRM(s);
    if (d.mod == 3) {
        if (d.reg == 4) {
            if (d.rm == 2) { s.fpu_status_word &= 0x7F00; }  // FCLEX
            else if (d.rm == 3) { s.fpu_control_word = 0x037F; s.fpu_status_word = 0; s.fpu_tag_word = 0xFFFF; s.fpu_top = 0; }  // FINIT
        } else if (d.reg == 5) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), false)  // FUCOMI
        else if (d.reg == 6) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), false);  // FCOMI
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, @floatFromInt(memReadS32(s, addr))),  // FILD m32
            1 => { const i: i32 = @intFromFloat(@trunc(fpuGet(s, 0))); memWrite32(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTTP
            2 => { const i: i32 = @intFromFloat(@round(fpuGet(s, 0))); memWrite32(s, addr, @bitCast(i)); },  // FIST
            3 => { const i: i32 = @intFromFloat(@round(fpuGet(s, 0))); memWrite32(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTP
            5 => { // FLD m80real (simplified)
                const lo = memRead32(s, addr); const hi = memRead32(s, addr + 4); const exp = memRead16(s, addr + 8);
                const sign: f64 = if ((exp & 0x8000) != 0) -1.0 else 1.0;
                const e: i32 = @as(i32, exp & 0x7FFF) - 16383;
                const mant: f64 = (@as(f64, @floatFromInt(@as(u64, hi))) * 4294967296.0 + @as(f64, @floatFromInt(lo))) / 9223372036854775808.0;
                if (e == -16383 and lo == 0 and hi == 0) fpuPush(s, sign * 0.0)
                else fpuPush(s, sign * std.math.pow(f64, 2.0, @as(f64, @floatFromInt(e))) * mant);
            },
            7 => { writeDouble(s, addr, fpuGet(s, 0)); memWrite16(s, addr + 8, 0); _ = fpuPop(s); },  // FSTP m80
            else => {},
        }
    }
}
fn opDC(s: *CpuState) void { // float64 ops (reversed)
    const d = decodeModRM(s);
    if (d.mod == 3) {
        const st0 = fpuGet(s, 0); const sti = fpuGet(s, d.rm);
        switch (d.reg) {
            0 => fpuSet(s, d.rm, sti + st0), 1 => fpuSet(s, d.rm, sti * st0),
            2 => fpuCompare(s, st0, sti), 3 => { fpuCompare(s, st0, sti); _ = fpuPop(s); },
            4 => fpuSet(s, d.rm, sti - st0), 5 => fpuSet(s, d.rm, st0 - sti),
            6 => fpuSet(s, d.rm, sti / st0), 7 => fpuSet(s, d.rm, st0 / sti),
            else => {},
        }
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        const val = readDouble(s, addr); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}
fn opDD(s: *CpuState) void { // FLD/FST/FSTP float64, FUCOM
    const d = decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => { const idx = ((@as(u8, @truncate(s.fpu_top)) +% d.rm) & 7); s.fpu_tag_word |= @as(u16, 3) << (@as(u4, @truncate(idx)) * 2); },  // FFREE
            2 => fpuSet(s, d.rm, fpuGet(s, 0)),  // FST
            3 => { fpuSet(s, d.rm, fpuGet(s, 0)); _ = fpuPop(s); },  // FSTP
            4 => fpuCompare(s, fpuGet(s, 0), fpuGet(s, d.rm)),  // FUCOM
            5 => { fpuCompare(s, fpuGet(s, 0), fpuGet(s, d.rm)); _ = fpuPop(s); },  // FUCOMP
            else => {},
        }
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, readDouble(s, addr)),  // FLD m64
            1 => { writeDouble(s, addr, @trunc(fpuGet(s, 0))); _ = fpuPop(s); },  // FISTTP m64
            2 => writeDouble(s, addr, fpuGet(s, 0)),  // FST m64
            3 => { writeDouble(s, addr, fpuGet(s, 0)); _ = fpuPop(s); },  // FSTP m64
            4, 6 => {},  // FRSTOR/FNSAVE NOP
            7 => memWrite16(s, addr, s.fpu_status_word),  // FNSTSW m16
            else => {},
        }
    }
}
fn opDE(s: *CpuState) void { // FADDP/FMULP/etc / int16
    const d = decodeModRM(s);
    if (d.mod == 3) {
        const st0 = fpuGet(s, 0); const sti = fpuGet(s, d.rm);
        switch (d.reg) {
            0 => { fpuSet(s, d.rm, sti + st0); _ = fpuPop(s); },  // FADDP
            1 => { fpuSet(s, d.rm, sti * st0); _ = fpuPop(s); },  // FMULP
            2 => { fpuCompare(s, st0, sti); _ = fpuPop(s); },
            3 => if (d.rm == 1) { fpuCompare(s, st0, fpuGet(s, 1)); _ = fpuPop(s); _ = fpuPop(s); },  // FCOMPP
            4 => { fpuSet(s, d.rm, st0 - sti); _ = fpuPop(s); },  // FSUBRP
            5 => { fpuSet(s, d.rm, sti - st0); _ = fpuPop(s); },  // FSUBP
            6 => { fpuSet(s, d.rm, st0 / sti); _ = fpuPop(s); },  // FDIVRP
            7 => { fpuSet(s, d.rm, sti / st0); _ = fpuPop(s); },  // FDIVP
            else => {},
        }
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        const raw = memRead16(s, addr); const val: f64 = @floatFromInt(@as(i16, @bitCast(raw)));
        const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}
fn opDF(s: *CpuState) void { // FILD/FISTP int16/int64, FNSTSW AX, FUCOMIP
    const d = decodeModRM(s);
    if (d.mod == 3) {
        if (d.reg == 4 and d.rm == 0) {  // FNSTSW AX
            s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | @as(u32, s.fpu_status_word);
        } else if (d.reg == 5) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), true)   // FUCOMIP
        else if (d.reg == 6) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), true);   // FCOMIP
    } else {
        const r = resolveRm(s, d.mod, d.rm); const addr = applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => { const raw = memRead16(s, addr); fpuPush(s, @floatFromInt(@as(i16, @bitCast(raw)))); },  // FILD m16
            1 => { const i: i16 = @intFromFloat(@trunc(fpuGet(s, 0))); memWrite16(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTTP m16
            2 => { const i: i16 = @intFromFloat(@round(fpuGet(s, 0))); memWrite16(s, addr, @bitCast(i)); },  // FIST m16
            3 => { const i: i16 = @intFromFloat(@round(fpuGet(s, 0))); memWrite16(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTP m16
            5 => { const lo = memRead32(s, addr); const hi: i32 = memReadS32(s, addr + 4); fpuPush(s, @as(f64, @floatFromInt(@as(i64, hi) * 0x100000000 + @as(i64, lo)))); },  // FILD m64
            7 => { const val = fpuGet(s, 0); const iv: i64 = @intFromFloat(@trunc(val)); const bits: u64 = @bitCast(iv); memWrite32(s, addr, @truncate(bits)); memWrite32(s, addr + 4, @truncate(bits >> 32)); _ = fpuPop(s); },  // FISTP m64
            else => {},
        }
    }
}

// ─── Dispatch table ───────────────────────────────────────────────────────────
const dispatch_table: [256]OpFn = dt: {
    @setEvalBranchQuota(20000);
    var t = [_]OpFn{opFault} ** 256;
    t[0x00] = op00; t[0x01] = op01; t[0x02] = op02; t[0x03] = op03; t[0x04] = op04; t[0x05] = op05;
    t[0x06] = op06; t[0x07] = op07; t[0x08] = op08; t[0x09] = op09; t[0x0A] = op0A; t[0x0B] = op0B;
    t[0x0C] = op0C; t[0x0D] = op0D; t[0x0E] = op0E; t[0x0F] = op0F;
    t[0x10] = op10; t[0x11] = op11; t[0x12] = op12; t[0x13] = op13; t[0x14] = op14; t[0x15] = op15;
    t[0x16] = op16; t[0x17] = op17; t[0x18] = op18; t[0x19] = op19; t[0x1A] = op1A; t[0x1B] = op1B;
    t[0x1C] = op1C; t[0x1D] = op1D; t[0x1E] = op1E; t[0x1F] = op1F;
    t[0x20] = op20; t[0x21] = op21; t[0x22] = op22; t[0x23] = op23; t[0x24] = op24; t[0x25] = op25;
    t[0x28] = op28; t[0x29] = op29; t[0x2A] = op2A; t[0x2B] = op2B; t[0x2C] = op2C; t[0x2D] = op2D;
    t[0x30] = op30; t[0x31] = op31; t[0x32] = op32; t[0x33] = op33; t[0x35] = op35;
    t[0x38] = op38; t[0x39] = op39; t[0x3A] = op3A; t[0x3B] = op3B; t[0x3C] = op3C; t[0x3D] = op3D;
    var r: u8 = 0;
    while (r < 8) : (r += 1) {
        const rr: u3 = @truncate(r);
        t[0x40 + r] = opIncR32(rr);
        t[0x48 + r] = opDecR32(rr);
        t[0x50 + r] = opPushR32(rr);
        t[0x58 + r] = opPopR32(rr);
        t[0x70 + r] = opJcc8(rr);
        t[0x78 + r] = opJcc8(@as(u8, rr) + 8);
        t[0xB8 + r] = opMovR32Imm(rr);
    }
    var rb: u8 = 0;
    while (rb < 8) : (rb += 1) {
        t[0xB0 + rb] = opMovR8Imm(rb);
    }
    t[0x60] = op60; t[0x61] = op61; t[0x68] = op68; t[0x69] = op69; t[0x6A] = op6A; t[0x6B] = op6B;
    t[0x80] = op80; t[0x81] = op81; t[0x83] = op83; t[0x84] = op84; t[0x85] = op85;
    t[0x86] = op86; t[0x87] = op87;
    t[0x88] = op88; t[0x89] = op89; t[0x8A] = op8A; t[0x8B] = op8B; t[0x8D] = op8D;
    t[0x90] = opNop;
    var rx: u8 = 1;
    while (rx < 8) : (rx += 1) {
        t[0x90 + rx] = opXchgEaxR(@truncate(rx));
    }
    t[0x98] = op98; t[0x99] = op99; t[0x9B] = op9B; t[0x9C] = op9C; t[0x9D] = op9D;
    t[0x9E] = op9E; t[0x9F] = op9F;
    t[0xA0] = opA0; t[0xA1] = opA1; t[0xA2] = opA2; t[0xA3] = opA3;
    t[0xA4] = opA4; t[0xA5] = opA5; t[0xA6] = opA6; t[0xA7] = opA7;
    t[0xA8] = opA8; t[0xA9] = opA9;
    t[0xAA] = opAA; t[0xAB] = opAB; t[0xAC] = opAC; t[0xAD] = opAD; t[0xAE] = opAE; t[0xAF] = opAF;
    t[0xC1] = opC1; t[0xC2] = opC2; t[0xC3] = opC3; t[0xC4] = opC4; t[0xC5] = opC5;
    t[0xC6] = opC6; t[0xC7] = opC7; t[0xC8] = opC8; t[0xC9] = opC9; t[0xCC] = opCC; t[0xCD] = opCD;
    t[0xD1] = opD1; t[0xD3] = opD3;
    t[0xD8] = opD8; t[0xD9] = opD9; t[0xDA] = opDA; t[0xDB] = opDB;
    t[0xDC] = opDC; t[0xDD] = opDD; t[0xDE] = opDE; t[0xDF] = opDF;
    t[0xE0] = opE0; t[0xE1] = opE1; t[0xE2] = opE2; t[0xE3] = opE3;
    t[0xE8] = opE8; t[0xE9] = opE9; t[0xEB] = opEB;
    t[0xF4] = opF4; t[0xF5] = opF5; t[0xF6] = opF6; t[0xF7] = opF7;
    t[0xF8] = opF8; t[0xF9] = opF9; t[0xFC] = opFC; t[0xFD] = opFD;
    t[0xFE] = opFE; t[0xFF] = opFF;
    break :dt t;
};

// ─── Execution engine ─────────────────────────────────────────────────────────
fn cpuStep(s: *CpuState) void {
    var opcode = fetch8(s);
    while (isPrefix(opcode)) {
        switch (opcode) {
            0x64 => s.seg_override = SEG_FS,
            0x65 => s.seg_override = SEG_GS,
            0xF3 => s.rep_prefix = REP_REP,
            0xF2 => s.rep_prefix = REP_REPNE,
            0x66 => s.op_size_ovr = true,
            else => {},
        }
        opcode = fetch8(s);
    }
    s.last_opcode = opcode;
    dispatch_table[opcode](s);
    clearPrefixes(s);
    if (!s.faulted) s.step_count += 1;
}

// ─── C API ────────────────────────────────────────────────────────────────────
export fn cpu_create(memory: [*]u8, memory_size: usize) ?*CpuState {
    const s = std.heap.c_allocator.create(CpuState) catch return null;
    s.* = CpuState{ .memory = memory, .memory_size = memory_size };
    return s;
}
export fn cpu_destroy(s: *CpuState) void { std.heap.c_allocator.destroy(s); }
export fn cpu_set_int_handler(s: *CpuState, handler: IntHandlerFn) void { s.int_handler = handler; }
export fn cpu_run(s: *CpuState, max_steps: u64) RunResult {
    var i: u64 = 0;
    while (!s.halted and i < max_steps) : (i += 1) cpuStep(s);
    if (s.faulted) return .faulted;
    if (s.halted) return .halted;
    return .step_limit;
}
export fn cpu_get_reg(s: *CpuState, idx: u32) u32 { return if (idx < 8) s.regs[idx] else 0; }
export fn cpu_set_reg(s: *CpuState, idx: u32, val: u32) void { if (idx < 8) s.regs[idx] = val; }
export fn cpu_get_eip(s: *CpuState) u32 { return s.eip; }
export fn cpu_set_eip(s: *CpuState, val: u32) void { s.eip = val; }
export fn cpu_get_eflags(s: *CpuState) u32 { return s.eflags; }
export fn cpu_set_eflags(s: *CpuState, val: u32) void { s.eflags = val; }
export fn cpu_is_halted(s: *CpuState) bool { return s.halted; }
export fn cpu_is_faulted(s: *CpuState) bool { return s.faulted; }
export fn cpu_clear_halted(s: *CpuState) void { s.halted = false; s.faulted = false; }
export fn cpu_get_step_count(s: *CpuState) u64 { return s.step_count; }
export fn cpu_get_last_opcode(s: *CpuState) u8 { return s.last_opcode; }
export fn cpu_set_fs_base(s: *CpuState, val: u32) void { s.fs_base = val; }
export fn cpu_set_gs_base(s: *CpuState, val: u32) void { s.gs_base = val; }
export fn cpu_get_fs_base(s: *CpuState) u32 { return s.fs_base; }
export fn cpu_get_gs_base(s: *CpuState) u32 { return s.gs_base; }
export fn cpu_fpu_get(s: *CpuState, i: u32) f64 { return if (i < 8) s.fpu_stack[i] else 0.0; }
export fn cpu_fpu_set(s: *CpuState, i: u32, val: f64) void { if (i < 8) s.fpu_stack[i] = val; }
export fn cpu_fpu_get_top(s: *CpuState) u32 { return s.fpu_top; }
export fn cpu_fpu_set_top(s: *CpuState, val: u32) void { s.fpu_top = val & 7; }
export fn cpu_fpu_get_status(s: *CpuState) u16 { return s.fpu_status_word; }
export fn cpu_fpu_set_status(s: *CpuState, val: u16) void { s.fpu_status_word = val; }
export fn cpu_fpu_get_control(s: *CpuState) u16 { return s.fpu_control_word; }
export fn cpu_fpu_set_control(s: *CpuState, val: u16) void { s.fpu_control_word = val; }
export fn cpu_fpu_get_tag(s: *CpuState) u16 { return s.fpu_tag_word; }
export fn cpu_fpu_set_tag(s: *CpuState, val: u16) void { s.fpu_tag_word = val; }

// ─── Tests ────────────────────────────────────────────────────────────────────
const testing = std.testing;

test "NOP advances EIP" {
    var mem = [_]u8{0x90} ++ [_]u8{0xF4} ++ [_]u8{0} ** 62;
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 1), s.eip);
    try testing.expect(!s.halted);
}

test "HLT stops execution" {
    var mem = [_]u8{0xF4} ++ [_]u8{0} ** 63;
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    cpuStep(&s);
    try testing.expect(s.halted);
    try testing.expect(!s.faulted);
}

test "ADD EAX imm32" {
    var mem = [_]u8{0x05, 0x05, 0x00, 0x00, 0x00} ++ [_]u8{0} ** 59; // ADD EAX, 5
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 10;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 15), s.regs[EAX]);
}

test "PUSH/POP round-trip" {
    var combined = [_]u8{0x50, 0x58} ++ [_]u8{0} ** 1022;
    var cs = CpuState{ .memory = &combined, .memory_size = combined.len };
    cs.regs[EAX] = 0xDEADBEEF;
    cs.regs[ESP] = 0x200;
    cpuStep(&cs); // PUSH EAX
    try testing.expectEqual(@as(u32, 0x1FC), cs.regs[ESP]);
    cs.regs[EAX] = 0; // clobber
    cpuStep(&cs); // POP EAX
    try testing.expectEqual(@as(u32, 0xDEADBEEF), cs.regs[EAX]);
    try testing.expectEqual(@as(u32, 0x200), cs.regs[ESP]);
}

test "XOR EAX, EAX zeroes register" {
    var mem = [_]u8{0x33, 0xC0} ++ [_]u8{0} ** 62; // XOR EAX, EAX
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x12345678;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0), s.regs[EAX]);
    try testing.expect(getFlag(&s, ZF_BIT));
}
