// SPDX-License-Identifier: LGPL-3.0-or-later
// core.zig — CpuState definition and all shared helpers.
// Imported by every opcode module; no imports from our own modules here.
const std = @import("std");

// ─── Register indices ────────────────────────────────────────────────────────
pub const EAX: u3 = 0;
pub const ECX: u3 = 1;
pub const EDX: u3 = 2;
pub const EBX: u3 = 3;
pub const ESP: u3 = 4;
pub const EBP: u3 = 5;
pub const ESI: u3 = 6;
pub const EDI: u3 = 7;

// ─── EFLAGS bit positions ─────────────────────────────────────────────────────
pub const CF_BIT: u5 = 0;
pub const PF_BIT: u5 = 2;
pub const ZF_BIT: u5 = 6;
pub const SF_BIT: u5 = 7;
pub const DF_BIT: u5 = 10;
pub const OF_BIT: u5 = 11;

// ─── Segment override tokens ─────────────────────────────────────────────────
pub const SEG_NONE: u8 = 0;
pub const SEG_FS: u8 = 1;
pub const SEG_GS: u8 = 2;

// ─── REP prefix tokens ───────────────────────────────────────────────────────
pub const REP_NONE: u8 = 0;
pub const REP_REP: u8 = 1;
pub const REP_REPNE: u8 = 2;

// ─── Public types ─────────────────────────────────────────────────────────────
pub const IntHandlerFn = *const fn (state: *anyopaque, int_num: u8) callconv(.C) void;
pub const LogpointFn  = *const fn (eip: u32, regs: [*]u32, memory: [*]u8, memory_size: usize) callconv(.C) void;
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
    // f80 gives 64-bit mantissa — exact round-trip for all 64-bit integers (FILD/FISTP m64).
    fpu_stack: [8]f80 = .{0.0} ** 8,
    fpu_top: u32 = 0,
    fpu_status_word: u16 = 0,
    fpu_control_word: u16 = 0x037F,
    fpu_tag_word: u16 = 0xFFFF,
    // MMX registers mm0–mm7 (shared silicon with x87, separate storage here).
    mmx_regs: [8]u64 = .{0} ** 8,
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
    watchpoint: u32 = 0,
    watchpoint_eip: u32 = 0,
    watchpoint_val: u32 = 0,
    watchpoint_hit: bool = false,
    // EIP breakpoints (halt): up to 8 slots; 0 = empty.
    bp_table:   [8]u32 = .{0} ** 8,
    bp_hit:     bool   = false,
    bp_hit_eip: u32    = 0,
    // EIP logpoints (fire C callback inline, no halt): up to 8 slots.
    lp_eip:  [8]u32     = .{0} ** 8,
    lp_cb:   [8]?LogpointFn = .{null} ** 8,
};

// ─── Internal helper structs ─────────────────────────────────────────────────
pub const RmInfo = struct { is_reg: bool, addr: u32 };
pub const Rm8Result = struct { value: u8, is_reg: bool, addr: u32 };
pub const Rm32Result = struct { value: u32, is_reg: bool, addr: u32 };
pub const ModRm = struct { mod: u8, reg: u8, rm: u8 };

// ─── Memory access ────────────────────────────────────────────────────────────
pub inline fn memRead8(s: *CpuState, addr: u32) u8 {
    if (addr >= @as(u32, @truncate(s.memory_size))) { s.faulted = true; s.halted = true; return 0; }
    return s.memory[addr];
}
pub inline fn memRead16(s: *CpuState, addr: u32) u16 {
    return @as(u16, memRead8(s, addr)) | (@as(u16, memRead8(s, addr + 1)) << 8);
}
pub inline fn memRead32(s: *CpuState, addr: u32) u32 {
    return @as(u32, memRead8(s, addr)) | (@as(u32, memRead8(s, addr + 1)) << 8) |
           (@as(u32, memRead8(s, addr + 2)) << 16) | (@as(u32, memRead8(s, addr + 3)) << 24);
}
pub inline fn memReadS32(s: *CpuState, addr: u32) i32 { return @bitCast(memRead32(s, addr)); }
pub inline fn memWrite8(s: *CpuState, addr: u32, v: u8) void {
    if (addr >= @as(u32, @truncate(s.memory_size))) { s.faulted = true; s.halted = true; return; }
    if (s.watchpoint != 0 and addr == s.watchpoint) {
        s.watchpoint_eip = s.eip;
        s.watchpoint_val = v;
        s.watchpoint_hit = true;
        s.memory[addr] = v;
        s.halted = true;
        return;
    }
    s.memory[addr] = v;
}
pub inline fn memWrite16(s: *CpuState, addr: u32, v: u16) void {
    memWrite8(s, addr, @truncate(v));
    memWrite8(s, addr + 1, @truncate(v >> 8));
}
pub inline fn memWrite32(s: *CpuState, addr: u32, v: u32) void {
    memWrite8(s, addr, @truncate(v));
    memWrite8(s, addr + 1, @truncate(v >> 8));
    memWrite8(s, addr + 2, @truncate(v >> 16));
    memWrite8(s, addr + 3, @truncate(v >> 24));
}

// ─── Fetch helpers ────────────────────────────────────────────────────────────
pub inline fn fetch8(s: *CpuState) u8 {
    const v = memRead8(s, s.eip);
    s.eip +%= 1;
    return v;
}
pub inline fn fetch16(s: *CpuState) u16 {
    const v = memRead16(s, s.eip);
    s.eip +%= 2;
    return v;
}
pub inline fn fetch32(s: *CpuState) u32 {
    const v = memRead32(s, s.eip);
    s.eip +%= 4;
    return v;
}
pub inline fn fetchS8(s: *CpuState) i8 { return @bitCast(fetch8(s)); }
pub inline fn fetchS32(s: *CpuState) i32 { return @bitCast(fetch32(s)); }
pub inline fn fetchImm(s: *CpuState) u32 {
    return if (s.op_size_ovr) @as(u32, fetch16(s)) else fetch32(s);
}
pub inline fn fetchSImm(s: *CpuState) i32 {
    if (s.op_size_ovr) {
        const v = fetch16(s);
        return @as(i32, @as(i16, @bitCast(v)));
    }
    return fetchS32(s);
}

// ─── Flag helpers ─────────────────────────────────────────────────────────────
pub inline fn getFlag(s: *CpuState, bit: u5) bool { return ((s.eflags >> bit) & 1) != 0; }
pub inline fn setFlag(s: *CpuState, bit: u5, v: bool) void {
    if (v) s.eflags |= @as(u32, 1) << bit else s.eflags &= ~(@as(u32, 1) << bit);
}
pub fn updateFlagsArith(s: *CpuState, result_raw: i64, op1: u32, op2: u32, is_sub: bool) void {
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
pub fn updateFlagsLogic(s: *CpuState, result: u32) void {
    setFlag(s, ZF_BIT, result == 0);
    setFlag(s, SF_BIT, (result & 0x80000000) != 0);
    setFlag(s, CF_BIT, false);
    setFlag(s, OF_BIT, false);
    var p: u8 = @truncate(result);
    p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
    setFlag(s, PF_BIT, (p & 1) == 0);
}
pub fn updateFlagsLogic8(s: *CpuState, result: u8) void {
    setFlag(s, ZF_BIT, result == 0);
    setFlag(s, SF_BIT, (result & 0x80) != 0);
    setFlag(s, CF_BIT, false);
    setFlag(s, OF_BIT, false);
    var p: u8 = result;
    p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
    setFlag(s, PF_BIT, (p & 1) == 0);
}

// ─── 8-bit register helpers ──────────────────────────────────────────────────
pub inline fn readReg8(s: *CpuState, idx: u8) u8 {
    return if (idx < 4) @truncate(s.regs[idx]) else @truncate(s.regs[idx - 4] >> 8);
}
pub inline fn writeReg8(s: *CpuState, idx: u8, v: u8) void {
    if (idx < 4) {
        s.regs[idx] = (s.regs[idx] & 0xFFFFFF00) | @as(u32, v);
    } else {
        s.regs[idx - 4] = (s.regs[idx - 4] & 0xFFFF00FF) | (@as(u32, v) << 8);
    }
}

// ─── Stack helpers ────────────────────────────────────────────────────────────
pub inline fn push32(s: *CpuState, v: u32) void {
    s.regs[ESP] -%= 4;
    memWrite32(s, s.regs[ESP], v);
}
pub inline fn pop32(s: *CpuState) u32 {
    const v = memRead32(s, s.regs[ESP]);
    s.regs[ESP] +%= 4;
    return v;
}

// ─── Segment override ─────────────────────────────────────────────────────────
pub inline fn applySegOvr(s: *CpuState, addr: u32) u32 {
    return switch (s.seg_override) {
        SEG_FS => s.fs_base +% addr,
        SEG_GS => s.gs_base +% addr,
        else => addr,
    };
}

// ─── ModRM decode ─────────────────────────────────────────────────────────────
pub fn decodeModRM(s: *CpuState) ModRm {
    const b = fetch8(s);
    return .{ .mod = (b >> 6) & 3, .reg = (b >> 3) & 7, .rm = b & 7 };
}
pub fn decodeSIB(s: *CpuState, mod: u8) u32 {
    const sib = fetch8(s);
    const scale: u32 = @as(u32, 1) << @as(u2, @truncate(sib >> 6));
    const index: u8 = (sib >> 3) & 7;
    const base: u8 = sib & 7;
    var addr: u32 = if (base == 5 and mod == 0) fetch32(s) else s.regs[base];
    if (index != 4) addr +%= s.regs[index] *% scale;
    return addr;
}
pub fn resolveRm(s: *CpuState, mod: u8, rm: u8) RmInfo {
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
pub fn readRm8(s: *CpuState, mod: u8, rm: u8) u8 {
    const r = resolveRm(s, mod, rm);
    return if (r.is_reg) readReg8(s, @truncate(r.addr)) else memRead8(s, applySegOvr(s, r.addr));
}
pub fn writeRm8(s: *CpuState, mod: u8, rm: u8, v: u8) void {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) writeReg8(s, @truncate(r.addr), v) else memWrite8(s, applySegOvr(s, r.addr), v);
}
pub fn readRm8Resolved(s: *CpuState, mod: u8, rm: u8) Rm8Result {
    const r = resolveRm(s, mod, rm);
    const addr = if (r.is_reg) r.addr else applySegOvr(s, r.addr);
    const v: u8 = if (r.is_reg) readReg8(s, @truncate(r.addr)) else memRead8(s, addr);
    return .{ .value = v, .is_reg = r.is_reg, .addr = addr };
}
pub fn writeRm8Resolved(s: *CpuState, is_reg: bool, addr: u32, v: u8) void {
    if (is_reg) writeReg8(s, @truncate(addr), v) else memWrite8(s, addr, v);
}
pub fn readRm32(s: *CpuState, mod: u8, rm: u8) u32 {
    const r = resolveRm(s, mod, rm);
    return if (r.is_reg) s.regs[r.addr] else memRead32(s, applySegOvr(s, r.addr));
}
pub fn writeRm32(s: *CpuState, mod: u8, rm: u8, v: u32) void {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) s.regs[r.addr] = v else memWrite32(s, applySegOvr(s, r.addr), v);
}
pub fn readRm32Resolved(s: *CpuState, mod: u8, rm: u8) Rm32Result {
    const r = resolveRm(s, mod, rm);
    const addr = if (r.is_reg) r.addr else applySegOvr(s, r.addr);
    const v: u32 = if (r.is_reg) s.regs[r.addr] else memRead32(s, addr);
    return .{ .value = v, .is_reg = r.is_reg, .addr = addr };
}
pub fn writeRm32Resolved(s: *CpuState, is_reg: bool, addr: u32, v: u32) void {
    if (is_reg) s.regs[addr] = v else memWrite32(s, addr, v);
}
pub fn readRmv(s: *CpuState, mod: u8, rm: u8) u32 {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) {
        return if (s.op_size_ovr) s.regs[r.addr] & 0xFFFF else s.regs[r.addr];
    }
    const addr = applySegOvr(s, r.addr);
    return if (s.op_size_ovr) @as(u32, memRead16(s, addr)) else memRead32(s, addr);
}
pub fn writeRmv(s: *CpuState, mod: u8, rm: u8, v: u32) void {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) {
        if (s.op_size_ovr) s.regs[r.addr] = (s.regs[r.addr] & 0xFFFF0000) | (v & 0xFFFF)
        else s.regs[r.addr] = v;
    } else {
        const addr = applySegOvr(s, r.addr);
        if (s.op_size_ovr) memWrite16(s, addr, @truncate(v)) else memWrite32(s, addr, v);
    }
}
pub fn readRmvResolved(s: *CpuState, mod: u8, rm: u8) Rm32Result {
    const r = resolveRm(s, mod, rm);
    const addr = if (r.is_reg) r.addr else applySegOvr(s, r.addr);
    const v: u32 = if (r.is_reg)
        (if (s.op_size_ovr) s.regs[r.addr] & 0xFFFF else s.regs[r.addr])
    else
        (if (s.op_size_ovr) @as(u32, memRead16(s, addr)) else memRead32(s, addr));
    return .{ .value = v, .is_reg = r.is_reg, .addr = addr };
}
pub fn writeRmvResolved(s: *CpuState, is_reg: bool, addr: u32, v: u32) void {
    if (is_reg) {
        if (s.op_size_ovr) s.regs[addr] = (s.regs[addr] & 0xFFFF0000) | (v & 0xFFFF)
        else s.regs[addr] = v;
    } else {
        if (s.op_size_ovr) memWrite16(s, addr, @truncate(v)) else memWrite32(s, addr, v);
    }
}
pub inline fn readEaxv(s: *CpuState) u32 {
    return if (s.op_size_ovr) s.regs[EAX] & 0xFFFF else s.regs[EAX];
}
pub inline fn writeEaxv(s: *CpuState, v: u32) void {
    if (s.op_size_ovr) s.regs[EAX] = (s.regs[EAX] & 0xFFFF0000) | (v & 0xFFFF)
    else s.regs[EAX] = v;
}

// ─── Condition evaluation ─────────────────────────────────────────────────────
pub fn evalCond(s: *CpuState, cond: u8) bool {
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

// ─── Fault handler ────────────────────────────────────────────────────────────
pub fn opFault(s: *CpuState) void { s.faulted = true; s.halted = true; }
