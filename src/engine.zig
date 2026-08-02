// SPDX-License-Identifier: LGPL-3.0-or-later
// engine.zig — One-byte opcode handlers, dispatch table, execution engine.
// Internal only: kernel.zig (the Python-facing C ABI) drives this via
// cpuStep, nothing here is `export`. Shared types/helpers live in core.zig.
// FPU ops in fpu.zig. 0x0F ops in two_byte.zig.
const std = @import("std");
const core = @import("core.zig");
const fpu = @import("fpu.zig");
const two_byte = @import("two_byte.zig");

// ─── Type and constant aliases from core ────────────────────────────────────
// CpuState/RunResult are `pub` -- unlike the rest of these aliases -- so an
// external Zig consumer that @imports this file as a module (e.g.
// pe-walker's emulator.zig, vendoring this source rather than linking a
// compiled library) can reference them directly, the same way cpu_run below
// is `pub export fn` rather than just `export fn`.
pub const CpuState = core.CpuState;
pub const RunResult = core.RunResult;
pub const WriteHookFn = core.WriteHookFn;
pub const StepHookFn = core.StepHookFn;
const EAX = core.EAX; const ECX = core.ECX; const EDX = core.EDX; const EBX = core.EBX;
const ESP = core.ESP; const EBP = core.EBP; const ESI = core.ESI; const EDI = core.EDI;
const CF_BIT = core.CF_BIT; const PF_BIT = core.PF_BIT; const ZF_BIT = core.ZF_BIT;
const SF_BIT = core.SF_BIT; const DF_BIT = core.DF_BIT; const OF_BIT = core.OF_BIT;
const SEG_NONE = core.SEG_NONE; const SEG_FS = core.SEG_FS; const SEG_GS = core.SEG_GS;
const REP_NONE = core.REP_NONE; const REP_REP = core.REP_REP; const REP_REPNE = core.REP_REPNE;
const IntHandlerFn = core.IntHandlerFn;
const OpFn = core.OpFn;
const RmInfo = core.RmInfo;
const Rm8Result = core.Rm8Result;
const Rm32Result = core.Rm32Result;
const ModRm = core.ModRm;

// ─── Helper aliases from core ────────────────────────────────────────────────
const memRead8 = core.memRead8;
const memRead16 = core.memRead16;
const memRead32 = core.memRead32;
const memWrite8 = core.memWrite8;
const memWrite16 = core.memWrite16;
const memWrite32 = core.memWrite32;
const fetch8 = core.fetch8;
const fetch32 = core.fetch32;
const fetchS8 = core.fetchS8;
const fetchS32 = core.fetchS32;
const getFlag = core.getFlag;
const setFlag = core.setFlag;
const Width = core.Width;
const updateFlagsArithW = core.updateFlagsArithW;
const updateFlagsLogicW = core.updateFlagsLogicW;
const readReg8 = core.readReg8;
const writeReg8 = core.writeReg8;
const push32 = core.push32;
const pop32 = core.pop32;
const applySegOvr = core.applySegOvr;
const decodeModRM = core.decodeModRM;
const resolveRm = core.resolveRm;
const readRm8 = core.readRm8;
const writeRm8 = core.writeRm8;
const readRmFixed32 = core.readRmFixed32;
const writeRmFixed32 = core.writeRmFixed32;
const readRm8Resolved = core.readRm8Resolved;
const writeRm8Resolved = core.writeRm8Resolved;
const readRmFixed32Resolved = core.readRmFixed32Resolved;
const writeRmFixed32Resolved = core.writeRmFixed32Resolved;
const evalCond = core.evalCond;
const opFault = core.opFault;
// ─── Local helpers not in core ───────────────────────────────────────────────
inline fn fetch16(s: *CpuState) u16 {
    const v = memRead16(s, s.eip); s.eip +%= 2; return v;
}
inline fn fetchImm(s: *CpuState) u32 {
    return if (s.op_size_ovr) @as(u32, fetch16(s)) else fetch32(s);
}
inline fn fetchSImm(s: *CpuState) i32 {
    if (s.op_size_ovr) { const v = fetch16(s); return @as(i32, @as(i16, @bitCast(v))); }
    return fetchS32(s);
}
fn readRmv(s: *CpuState, mod: u8, rm: u8) u32 {
    const r = resolveRm(s, mod, rm);
    if (r.is_reg) return if (s.op_size_ovr) s.regs[r.addr] & 0xFFFF else s.regs[r.addr];
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
fn clearPrefixes(s: *CpuState) void {
    s.seg_override = SEG_NONE; s.rep_prefix = REP_NONE; s.op_size_ovr = false;
}
fn isPrefix(b: u8) bool {
    return switch (b) {
        0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3 => true,
        else => false,
    };
}
inline fn strDir(s: *CpuState) i32 { return if (getFlag(s, DF_BIT)) -1 else 1; }

// ─── Group 1 helper (ADD/OR/ADC/SBB/AND/SUB/XOR/CMP) ────────────────────────
fn doGroup1(s: *CpuState, is_reg: bool, addr: u32, op_ext: u8, op1: u32, op2: u32) void {
    switch (op_ext) {
        0 => { const r = op1 +% op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsArithW(s, @as(i64, op1) + @as(i64, op2), op1, op2, false, .w32); },
        1 => { const r = op1 | op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsLogicW(s, r, .w32); },
        2 => { const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0; const r = op1 +% op2 +% c; writeRmvResolved(s, is_reg, addr, r); updateFlagsArithW(s, @as(i64, op1) + @as(i64, op2) + @as(i64, c), op1, op2 +% c, false, .w32); },
        3 => { const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0; const r = op1 -% op2 -% b; writeRmvResolved(s, is_reg, addr, r); updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2) - @as(i64, b), op1, op2 +% b, true, .w32); },
        4 => { const r = op1 & op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsLogicW(s, r, .w32); },
        5 => { const r = op1 -% op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, .w32); },
        6 => { const r = op1 ^ op2; writeRmvResolved(s, is_reg, addr, r); updateFlagsLogicW(s, r, .w32); },
        7 => updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, .w32),
        else => { s.faulted = true; s.halted = true; },
    }
}

// ─── Group 2 helper (shift/rotate 32-bit) ─────────────────────────────────
fn doGroup2(s: *CpuState, is_reg: bool, addr: u32, op_ext: u8, val: u32, count: u8, width: Width) void {
    // count==0 must still perform the write (real hardware still does the
    // read-modify-write cycle for a shift/rotate by zero) -- this was
    // hardcoded 32-bit (writeRmFixed32Resolved) regardless of op_size_ovr,
    // a real memory-corruption bug: a 16-bit shift-by-zero on a memory
    // operand wrote 4 bytes instead of 2.
    if (count == 0) { writeRmvResolved(s, is_reg, addr, val); return; }
    const bits: u6 = if (width == .w16) 16 else 32;
    const mask: u32 = if (width == .w16) 0xFFFF else 0xFFFFFFFF;
    const sign_bit: u32 = if (width == .w16) 0x8000 else 0x80000000;
    const c5: u5 = @truncate(count);
    switch (op_ext) {
        0 => { // ROL -- rotate amount/OF check generalized to `bits` (was
            // hardcoded to 32, so a 16-bit ROL wrapped around a 32-bit
            // boundary instead of a 16-bit one).
            const c: u8 = count % @as(u8, @intCast(bits));
            const r: u32 = if (c == 0) val else (((val << @as(u5, @truncate(c))) | (val >> @as(u5, @truncate(bits - @as(u6, @intCast(c)))))) & mask);
            const new_cf = (r & 1) != 0;
            writeRmvResolved(s, is_reg, addr, r); setFlag(s, CF_BIT, new_cf);
            if (count == 1) setFlag(s, OF_BIT, ((r & sign_bit) != 0) != ((r & (sign_bit >> 1)) != 0));
        },
        1 => { // ROR -- see ROL's comment
            const c: u8 = count % @as(u8, @intCast(bits));
            const r: u32 = if (c == 0) val else (((val >> @as(u5, @truncate(c))) | (val << @as(u5, @truncate(bits - @as(u6, @intCast(c)))))) & mask);
            const new_cf = (r & sign_bit) != 0;
            writeRmvResolved(s, is_reg, addr, r); setFlag(s, CF_BIT, new_cf);
            if (count == 1) setFlag(s, OF_BIT, ((r & sign_bit) != 0) != ((val & sign_bit) != 0));
        },
        2 => { // RCL -- write is already width-aware; the rotate-through-CF
            // math itself still assumes 32-bit width internally (not
            // generalized in this pass -- doGroup2_8's proper 9-bit-modular
            // version is the pattern a future fix should follow for 16-bit;
            // left as a known gap, same as the Phase 2+ consolidation this
            // was scoped out of).
            var temp = val << c5;
            if (getFlag(s, CF_BIT)) temp |= @as(u32, 1) << @as(u5, @truncate(count - 1));
            const new_cf = ((val >> @as(u5, @truncate(32 - @as(u8, count)))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, temp); setFlag(s, CF_BIT, new_cf);
        },
        3 => { // RCR -- see RCL's comment
            var temp = val >> c5;
            if (getFlag(s, CF_BIT)) temp |= @as(u32, 1) << @as(u5, @truncate(32 - @as(u8, count)));
            const new_cf = ((val >> @as(u5, @truncate(count - 1))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, temp); setFlag(s, CF_BIT, new_cf);
        },
        4 => { // SHL -- CF extraction and flags width generalized to `bits`
            const r = (val << c5) & mask;
            const shift_for_cf: u6 = if (count > bits) bits else @as(u6, @intCast(count));
            const new_cf = shift_for_cf != 0 and ((val >> @as(u5, @truncate(bits - shift_for_cf))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogicW(s, r, width);
        },
        5 => { // SHR -- see SHL's comment
            const r = (val & mask) >> c5;
            const shift_for_cf: u6 = if (count > bits) bits else @as(u6, @intCast(count));
            const new_cf = shift_for_cf != 0 and ((val >> @as(u5, @truncate(shift_for_cf - 1))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogicW(s, r, width);
        },
        7 => { // SAR -- was hardcoded-32-bit write (writeRmFixed32Resolved,
            // same memory-corruption shape as count==0 above) AND assumed a
            // 32-bit sign bit unconditionally: once the caller's read is
            // width-aware, a genuinely-16-bit operand's upper 16 bits are
            // correctly zero (not sign-extended), so shifting `val` as a
            // plain i32 would treat every 16-bit operand as non-negative.
            // Sign-extend from the width-correct bit first instead.
            const shifted: u32 = switch (width) {
                .w16 => blk: {
                    const signed16: i16 = @bitCast(@as(u16, @truncate(val)));
                    const wide: i32 = signed16;
                    break :blk @as(u32, @bitCast(wide >> c5)) & mask;
                },
                else => @bitCast(@as(i32, @bitCast(val)) >> c5),
            };
            const shift_for_cf: u6 = if (count > bits) bits else @as(u6, @intCast(count));
            const new_cf = shift_for_cf != 0 and ((val >> @as(u5, @truncate(shift_for_cf - 1))) & 1) != 0;
            writeRmvResolved(s, is_reg, addr, shifted);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogicW(s, shifted, width);
        },
        else => { s.faulted = true; s.halted = true; },
    }
}

fn doGroup2_8(s: *CpuState, is_reg: bool, addr: u32, op_ext: u8, val: u8, count: u8) void {
    if (count == 0) { writeRm8Resolved(s, is_reg, addr, val); return; }
    switch (op_ext) {
        0 => { // ROL r/m8
            const c8: u3 = @truncate(count & 7);
            const r: u8 = if (c8 == 0) val else (val << c8) | (val >> @as(u3, @truncate(8 - @as(u8, c8))));
            writeRm8Resolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, (r & 1) != 0);
            if ((count & 0x1F) == 1) setFlag(s, OF_BIT, ((r & 0x80) != 0) != ((r & 1) != 0));
        },
        1 => { // ROR r/m8
            const c8: u3 = @truncate(count & 7);
            const r: u8 = if (c8 == 0) val else (val >> c8) | (val << @as(u3, @truncate(8 - @as(u8, c8))));
            writeRm8Resolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, (r & 0x80) != 0);
            if ((count & 0x1F) == 1) setFlag(s, OF_BIT, ((r & 0x80) != 0) != ((r & 0x40) != 0));
        },
        2 => { // RCL r/m8 — 9-bit rotation through CF
            const c9: u8 = count % 9;
            var temp: u8 = val; var cf = getFlag(s, CF_BIT);
            var i: u8 = 0;
            while (i < c9) : (i += 1) {
                const out = (temp & 0x80) != 0;
                temp = (temp << 1) | @as(u8, if (cf) 1 else 0);
                cf = out;
            }
            writeRm8Resolved(s, is_reg, addr, temp); setFlag(s, CF_BIT, cf);
        },
        3 => { // RCR r/m8 — 9-bit rotation through CF
            const c9: u8 = count % 9;
            var temp: u8 = val; var cf = getFlag(s, CF_BIT);
            var i: u8 = 0;
            while (i < c9) : (i += 1) {
                const out = (temp & 1) != 0;
                temp = (temp >> 1) | (@as(u8, if (cf) 1 else 0) << 7);
                cf = out;
            }
            writeRm8Resolved(s, is_reg, addr, temp); setFlag(s, CF_BIT, cf);
        },
        4 => { // SHL r/m8
            const c8: u3 = @truncate(count & 7);
            const r: u8 = if (c8 == 0) val else val << c8;
            const new_cf = if (count <= 8) ((val >> @as(u3, @truncate(8 - @as(u8, count)))) & 1) != 0 else false;
            writeRm8Resolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogicW(s, r, .w8);
        },
        5 => { // SHR r/m8
            const c8: u3 = @truncate(count & 7);
            const r: u8 = if (c8 == 0) val else val >> c8;
            const new_cf = ((val >> @as(u3, @truncate(count - 1))) & 1) != 0;
            writeRm8Resolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogicW(s, r, .w8);
        },
        7 => { // SAR r/m8
            const c8: u3 = @truncate(count & 7);
            const r: u8 = @bitCast(@as(i8, @bitCast(val)) >> c8);
            const new_cf = ((val >> @as(u3, @truncate(count - 1))) & 1) != 0;
            writeRm8Resolved(s, is_reg, addr, r);
            setFlag(s, CF_BIT, new_cf); updateFlagsLogicW(s, r, .w8);
        },
        else => { s.faulted = true; s.halted = true; },
    }
}

// ─── Arithmetic opcodes ───────────────────────────────────────────────────────
fn op00(s: *CpuState) void { // ADD rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg); const r = res.value +% op2;
    writeRm8Resolved(s, res.is_reg, res.addr, r);
    updateFlagsArithW(s, @as(i64, res.value) + @as(i64, op2), res.value, op2, false, .w8);
}
fn op01(s: *CpuState) void { // ADD rmv, rv -- was hardcoded-32-bit read
    // (readRmFixed32Resolved) paired with a width-aware write (writeRmvResolved);
    // now width-aware throughout, including masking the register operand
    // under 0x66, which the fixed-32 read never did either.
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const r = res.value +% op2;
    writeRmvResolved(s, res.is_reg, res.addr, r);
    updateFlagsArithW(s, @as(i64, res.value) + @as(i64, op2), res.value, op2, false, width);
}
fn op02(s: *CpuState) void { // ADD r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, op1 +% op2);
    updateFlagsArithW(s, @as(i64, op1) + @as(i64, op2), op1, op2, false, .w8);
}
fn op03(s: *CpuState) void { // ADD rv, rmv -- op_size_ovr was never consulted
    // at all here (unlike its MOV counterpart op8B); now mirrors op8B's pattern.
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op1 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const op2 = readRmv(s, d.mod, d.rm);
    const r = op1 +% op2;
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsArithW(s, @as(i64, op1) + @as(i64, op2), op1, op2, false, width);
}
fn op04(s: *CpuState) void { // ADD AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al +% imm);
    updateFlagsArithW(s, @as(i64, al) + @as(i64, imm), al, imm, false, .w8);
}
fn op05(s: *CpuState) void { // ADD EAX/AX, immv -- see op85's comment: read width
    // alone isn't enough, flags width must match or SF is wrong for 16-bit results.
    const a = readEaxv(s); const imm = fetchImm(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    writeEaxv(s, a +% imm); updateFlagsArithW(s, @as(i64, a) + @as(i64, imm), a, imm, false, width);
}
fn op10(s: *CpuState) void { // ADC rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg); const c: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRm8Resolved(s, res.is_reg, res.addr, res.value +% op2 +% c);
    updateFlagsArithW(s, @as(i64, res.value) + @as(i64, op2) + @as(i64, c), res.value, op2 +% c, false, .w8);
}
fn op11(s: *CpuState) void { // ADC rmv, rv -- see op01's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRmvResolved(s, res.is_reg, res.addr, res.value +% op2 +% c);
    updateFlagsArithW(s, @as(i64, res.value) + @as(i64, op2) + @as(i64, c), res.value, op2 +% c, false, width);
}
fn op12(s: *CpuState) void { // ADC r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    const c: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeReg8(s, d.reg, op1 +% op2 +% c);
    updateFlagsArithW(s, @as(i64, op1) + @as(i64, op2) + @as(i64, c), op1, op2 +% c, false, .w8);
}
fn op13(s: *CpuState) void { // ADC rv, rmv -- see op03's comment
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op1 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const op2 = readRmv(s, d.mod, d.rm);
    const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    const r = op1 +% op2 +% c;
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsArithW(s, @as(i64, op1) + @as(i64, op2) + @as(i64, c), op1, op2 +% c, false, width);
}
fn op14(s: *CpuState) void { // ADC AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]); const c: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al +% imm +% c);
    updateFlagsArithW(s, @as(i64, al) + @as(i64, imm) + @as(i64, c), al, imm +% c, false, .w8);
}
fn op15(s: *CpuState) void { // ADC EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s); const c: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    writeEaxv(s, a +% imm +% c); updateFlagsArithW(s, @as(i64, a) + @as(i64, imm) + @as(i64, c), a, imm +% c, false, width);
}
fn op18(s: *CpuState) void { // SBB rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg); const b: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRm8Resolved(s, res.is_reg, res.addr, res.value -% op2 -% b);
    updateFlagsArithW(s, @as(i64, res.value) - @as(i64, op2) - @as(i64, b), res.value, op2 +% b, true, .w8);
}
fn op19(s: *CpuState) void { // SBB rmv, rv -- see op01's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeRmvResolved(s, res.is_reg, res.addr, res.value -% op2 -% b);
    updateFlagsArithW(s, @as(i64, res.value) - @as(i64, op2) - @as(i64, b), res.value, op2 +% b, true, width);
}
fn op1A(s: *CpuState) void { // SBB r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    const b: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    writeReg8(s, d.reg, op1 -% op2 -% b);
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2) - @as(i64, b), op1, op2 +% b, true, .w8);
}
fn op1B(s: *CpuState) void { // SBB rv, rmv -- see op03's comment
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op1 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const op2 = readRmv(s, d.mod, d.rm);
    const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    const r = op1 -% op2 -% b;
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2) - @as(i64, b), op1, op2 +% b, true, width);
}
fn op1C(s: *CpuState) void { // SBB AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]); const b: u8 = if (getFlag(s, CF_BIT)) 1 else 0;
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al -% imm -% b);
    updateFlagsArithW(s, @as(i64, al) - @as(i64, imm) - @as(i64, b), al, imm +% b, true, .w8);
}
fn op1D(s: *CpuState) void { // SBB EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s); const b: u32 = if (getFlag(s, CF_BIT)) 1 else 0;
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    writeEaxv(s, a -% imm -% b); updateFlagsArithW(s, @as(i64, a) - @as(i64, imm) - @as(i64, b), a, imm +% b, true, width);
}
fn op28(s: *CpuState) void { // SUB rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const op2 = readReg8(s, d.reg);
    writeRm8Resolved(s, res.is_reg, res.addr, res.value -% op2);
    updateFlagsArithW(s, @as(i64, res.value) - @as(i64, op2), res.value, op2, true, .w8);
}
fn op29(s: *CpuState) void { // SUB rmv, rv -- see op01's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    writeRmvResolved(s, res.is_reg, res.addr, res.value -% op2);
    updateFlagsArithW(s, @as(i64, res.value) - @as(i64, op2), res.value, op2, true, width);
}
fn op2A(s: *CpuState) void { // SUB r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, op1 -% op2); updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, .w8);
}
fn op2B(s: *CpuState) void { // SUB rv, rmv -- see op03's comment
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op1 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const op2 = readRmv(s, d.mod, d.rm);
    const r = op1 -% op2;
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, width);
}
fn op2C(s: *CpuState) void { // SUB AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | (al -% imm);
    updateFlagsArithW(s, @as(i64, al) - @as(i64, imm), al, imm, true, .w8);
}
fn op2D(s: *CpuState) void { // SUB EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    writeEaxv(s, a -% imm); updateFlagsArithW(s, @as(i64, a) - @as(i64, imm), a, imm, true, width);
}
fn op38(s: *CpuState) void { // CMP rm8, r8
    const d = decodeModRM(s); const op1 = readRm8(s, d.mod, d.rm); const op2 = readReg8(s, d.reg);
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, .w8);
}
fn op39(s: *CpuState) void { // CMP rmv, rv -- see op01's comment (no write here, flags only)
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op1 = readRmv(s, d.mod, d.rm);
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, width);
}
fn op3A(s: *CpuState) void { // CMP r8, rm8
    const d = decodeModRM(s); const op1 = readReg8(s, d.reg); const op2 = readRm8(s, d.mod, d.rm);
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, .w8);
}
fn op3B(s: *CpuState) void { // CMP rv, rmv -- see op03's comment (no write)
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op1 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const op2 = readRmv(s, d.mod, d.rm);
    updateFlagsArithW(s, @as(i64, op1) - @as(i64, op2), op1, op2, true, width);
}
fn op3C(s: *CpuState) void { // CMP AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    updateFlagsArithW(s, @as(i64, al) - @as(i64, imm), al, imm, true, .w8);
}
fn op3D(s: *CpuState) void { // CMP EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    updateFlagsArithW(s, @as(i64, a) - @as(i64, imm), a, imm, true, width);
}
fn opIncR32(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    const op1 = s.regs[r]; s.regs[r] = op1 +% 1;
    const cf = getFlag(s, CF_BIT); updateFlagsArithW(s, @as(i64, op1) + 1, op1, 1, false, .w32); setFlag(s, CF_BIT, cf);
}}.f; }
fn opDecR32(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    const op1 = s.regs[r]; s.regs[r] = op1 -% 1;
    const cf = getFlag(s, CF_BIT); updateFlagsArithW(s, @as(i64, op1) - 1, op1, 1, true, .w32); setFlag(s, CF_BIT, cf);
}}.f; }
fn op69(s: *CpuState) void { // IMUL r32, rm32, imm32
    const d = decodeModRM(s);
    const op1: i64 = @as(i32, @bitCast(readRmFixed32(s, d.mod, d.rm)));
    const imm: i64 = fetchS32(s);
    const r32: u32 = @truncate(@as(u64, @bitCast(op1 * imm)));
    s.regs[d.reg] = r32;
    const ov = (op1 * imm) != @as(i64, @as(i32, @bitCast(r32)));
    setFlag(s, CF_BIT, ov); setFlag(s, OF_BIT, ov);
}
fn op6B(s: *CpuState) void { // IMUL r32, rm32, imm8
    const d = decodeModRM(s);
    const op1: i64 = @as(i32, @bitCast(readRmFixed32(s, d.mod, d.rm)));
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
        0 => { const r = op1 +% imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsArithW(s, @as(i64,op1)+@as(i64,imm), op1, imm, false, .w8); },
        1 => { const r = op1 | imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, .w8); },
        2 => { const c: u8 = if(getFlag(s,CF_BIT)) 1 else 0; const r = op1+%imm+%c; writeRm8Resolved(s,res.is_reg,res.addr,r); updateFlagsArithW(s,@as(i64,op1)+@as(i64,imm)+@as(i64,c),op1,imm+%c,false, .w8); },
        3 => { const b: u8 = if(getFlag(s,CF_BIT)) 1 else 0; const r = op1-%imm-%b; writeRm8Resolved(s,res.is_reg,res.addr,r); updateFlagsArithW(s,@as(i64,op1)-@as(i64,imm)-@as(i64,b),op1,imm+%b,true, .w8); },
        4 => { const r = op1 & imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, .w8); },
        5 => { const r = op1 -% imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsArithW(s, @as(i64,op1)-@as(i64,imm), op1, imm, true, .w8); },
        6 => { const r = op1 ^ imm; writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, .w8); },
        7 => updateFlagsArithW(s, @as(i64,op1)-@as(i64,imm), op1, imm, true, .w8),
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
    const d = decodeModRM(s); const v1 = s.regs[d.reg]; const v2 = readRmFixed32(s, d.mod, d.rm);
    s.regs[d.reg] = v2; writeRmFixed32(s, d.mod, d.rm, v1);
}
fn opXchgEaxR(comptime r: u3) OpFn { return struct { fn f(s: *CpuState) void {
    const tmp = s.regs[EAX]; s.regs[EAX] = s.regs[r]; s.regs[r] = tmp;
}}.f; }
fn op99(s: *CpuState) void { // CDQ
    s.regs[EDX] = if ((s.regs[EAX] & 0x80000000) != 0) 0xFFFFFFFF else 0;
}
fn opA8(s: *CpuState) void { // TEST AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    updateFlagsLogicW(s, al & imm, .w8);
}
fn opA9(s: *CpuState) void { // TEST EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    updateFlagsLogicW(s, a & imm, width);
}
fn opC1(s: *CpuState) void { // Group 2: shift rmv, imm8 -- read was hardcoded
    // 32-bit (readRmFixed32Resolved), ignoring op_size_ovr; see doGroup2's
    // own comment for the layered bugs this + doGroup2 together had.
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    doGroup2(s, res.is_reg, res.addr, d.reg, res.value, fetch8(s) & 0x1F, width);
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
fn opC0(s: *CpuState) void { // Group 2: shift rm8, imm8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    doGroup2_8(s, res.is_reg, res.addr, d.reg, res.value, fetch8(s) & 0x1F);
}
fn opD1(s: *CpuState) void { // Group 2: shift rmv, 1 -- see opC1's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    doGroup2(s, res.is_reg, res.addr, d.reg, res.value, 1, width);
}
fn opD2(s: *CpuState) void { // Group 2: shift rm8, CL
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    doGroup2_8(s, res.is_reg, res.addr, d.reg, res.value, @truncate(s.regs[ECX] & 0x1F));
}
fn opD3(s: *CpuState) void { // Group 2: shift rmv, CL -- see opC1's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    doGroup2(s, res.is_reg, res.addr, d.reg, res.value, @truncate(s.regs[ECX] & 0x1F), width);
}
fn opF6(s: *CpuState) void { // Group 3 byte
    const d = decodeModRM(s); const val = readRm8(s, d.mod, d.rm);
    switch (d.reg) {
        0 => updateFlagsLogicW(s, val & fetch8(s), .w8),
        2 => writeRm8(s, d.mod, d.rm, ~val),
        3 => {
            const r = (0 -% @as(u32, val)) & 0xFF;
            writeRm8(s, d.mod, d.rm, @truncate(r));
            setFlag(s, CF_BIT, val != 0);
            updateFlagsArithW(s, -@as(i64, val), 0, val, true, .w8);
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
            if (is16) updateFlagsLogicW(s, readRmv(s,d.mod,d.rm) & fetch16(s), .w32)
            else updateFlagsLogicW(s, readRmFixed32(s,d.mod,d.rm) & fetch32(s), .w32);
        },
        2 => { // NOT
            if (is16) writeRmv(s,d.mod,d.rm, ~readRmv(s,d.mod,d.rm) & 0xFFFF)
            else writeRmFixed32(s,d.mod,d.rm, ~readRmFixed32(s,d.mod,d.rm));
        },
        3 => { // NEG
            if (is16) {
                const v = readRmv(s,d.mod,d.rm); const r = (0 -% v) & 0xFFFF;
                writeRmv(s,d.mod,d.rm,r); setFlag(s,CF_BIT,v!=0); updateFlagsArithW(s,-@as(i64,v),0,v,false, .w32);
            } else {
                const v = readRmFixed32(s,d.mod,d.rm); const r = 0 -% v;
                writeRmFixed32(s,d.mod,d.rm,r); setFlag(s,CF_BIT,v!=0); updateFlagsArithW(s,-@as(i64,v),0,v,true, .w32);
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
                const op1: u64 = s.regs[EAX]; const op2: u64 = readRmFixed32(s,d.mod,d.rm);
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
                const op2: i64 = @as(i32, @bitCast(readRmFixed32(s,d.mod,d.rm)));
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
                const div: u64 = readRmFixed32(s,d.mod,d.rm);
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
                const div: i64 = @as(i32, @bitCast(readRmFixed32(s,d.mod,d.rm)));
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
    const r = res.value | readReg8(s, d.reg); writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, .w8);
}
fn op09(s: *CpuState) void { // OR rmv, rv -- see op01's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const r = res.value | op2; writeRmvResolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, width);
}
fn op0A(s: *CpuState) void { // OR r8, rm8
    const d = decodeModRM(s); const r = readReg8(s, d.reg) | readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, r); updateFlagsLogicW(s, r, .w8);
}
fn op0B(s: *CpuState) void { // OR rv, rmv -- see op03's comment
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const r = s.regs[d.reg] | readRmv(s, d.mod, d.rm);
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsLogicW(s, r, width);
}
fn op0C(s: *CpuState) void { // OR AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    const r = al | imm; s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | r; updateFlagsLogicW(s, r, .w8);
}
fn op0D(s: *CpuState) void { // OR EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s); const r = a | imm;
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    updateFlagsLogicW(s, r, width); writeEaxv(s, r);
}
fn op20(s: *CpuState) void { // AND rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const r = res.value & readReg8(s, d.reg); writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, .w8);
}
fn op21(s: *CpuState) void { // AND rmv, rv -- see op01's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const r = res.value & op2; writeRmvResolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, width);
}
fn op22(s: *CpuState) void { // AND r8, rm8
    const d = decodeModRM(s); const r = readReg8(s, d.reg) & readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, r); updateFlagsLogicW(s, r, .w8);
}
fn op23(s: *CpuState) void { // AND rv, rmv -- see op03's comment
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const r = s.regs[d.reg] & readRmv(s, d.mod, d.rm);
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsLogicW(s, r, width);
}
fn op24(s: *CpuState) void { // AND AL, imm8
    const imm = fetch8(s); const al: u8 = @truncate(s.regs[EAX]);
    const r = al & imm; s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | r; updateFlagsLogicW(s, r, .w8);
}
fn op25(s: *CpuState) void { // AND EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s); const r = a & imm;
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    updateFlagsLogicW(s, r, width); writeEaxv(s, r);
}
fn op30(s: *CpuState) void { // XOR rm8, r8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    const r = res.value ^ readReg8(s, d.reg); writeRm8Resolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, .w8);
}
fn op31(s: *CpuState) void { // XOR rmv, rv -- see op01's comment
    const d = decodeModRM(s); const res = readRmvResolved(s, d.mod, d.rm);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    const r = res.value ^ op2; writeRmvResolved(s, res.is_reg, res.addr, r); updateFlagsLogicW(s, r, width);
}
fn op32(s: *CpuState) void { // XOR r8, rm8
    const d = decodeModRM(s); const r = readReg8(s, d.reg) ^ readRm8(s, d.mod, d.rm);
    writeReg8(s, d.reg, r); updateFlagsLogicW(s, r, .w8);
}
fn op33(s: *CpuState) void { // XOR rv, rmv -- see op03's comment
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const r = s.regs[d.reg] ^ readRmv(s, d.mod, d.rm);
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (r & 0xFFFF)
    else s.regs[d.reg] = r;
    updateFlagsLogicW(s, r, width);
}
fn op35(s: *CpuState) void { // XOR EAX/AX, immv -- see op85's comment
    const a = readEaxv(s); const imm = fetchImm(s); const r = a ^ imm;
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    updateFlagsLogicW(s, r, width); writeEaxv(s, r);
}
fn op84(s: *CpuState) void { // TEST rm8, r8
    const d = decodeModRM(s);
    updateFlagsLogicW(s, readRm8(s, d.mod, d.rm) & readReg8(s, d.reg), .w8);
}
fn op85(s: *CpuState) void { // TEST rmv, rv -- was hardcoded-32-bit read AND
    // a hardcoded-32-bit flag width; fixing only the read would still always
    // report SF=0 for a correctly-16-bit-masked value (bit 31 of a masked
    // 16-bit result is never set), so both halves need the width parameter.
    const d = decodeModRM(s);
    const width: Width = if (s.op_size_ovr) .w16 else .w32;
    const op2 = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
    updateFlagsLogicW(s, readRmv(s, d.mod, d.rm) & op2, width);
}

// ─── Data movement opcodes ────────────────────────────────────────────────────
fn op88(s: *CpuState) void { // MOV rm8, r8
    const d = decodeModRM(s); writeRm8(s, d.mod, d.rm, readReg8(s, d.reg));
}
fn op89(s: *CpuState) void { // MOV rmv, rv
    const d = decodeModRM(s);
    if (s.op_size_ovr) writeRmv(s, d.mod, d.rm, s.regs[d.reg] & 0xFFFF)
    else writeRmFixed32(s, d.mod, d.rm, s.regs[d.reg]);
}
fn op8A(s: *CpuState) void { // MOV r8, rm8
    const d = decodeModRM(s); writeReg8(s, d.reg, readRm8(s, d.mod, d.rm));
}
fn op8B(s: *CpuState) void { // MOV rv, rmv
    const d = decodeModRM(s);
    if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (readRmv(s, d.mod, d.rm) & 0xFFFF)
    else s.regs[d.reg] = readRmFixed32(s, d.mod, d.rm);
}
fn op8D(s: *CpuState) void { // LEA r32, rm
    const d = decodeModRM(s); const r = resolveRm(s, d.mod, d.rm); s.regs[d.reg] = r.addr;
}
fn opA0(s: *CpuState) void { // MOV AL, [disp32]
    const addr = applySegOvr(s, fetch32(s));
    s.regs[EAX] = (s.regs[EAX] & 0xFFFFFF00) | memRead8(s, addr);
}
fn opA1(s: *CpuState) void { // MOV EAX, [disp32]  (66-prefix: MOV AX, [disp32])
    const addr = applySegOvr(s, fetch32(s));
    if (s.op_size_ovr) s.regs[EAX] = @as(u32, memRead16(s, addr))
    else s.regs[EAX] = memRead32(s, addr);
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
    // Must honor 0x66: real MOV r16, imm16 reads only 2 bytes and writes only
    // the low 16 bits (upper 16 untouched) -- ignoring op_size_ovr here reads
    // 2 bytes too many as a bogus imm32 and desyncs EIP from the real
    // instruction stream (found live via MSJET35.DLL's 0x15035655 fault).
    if (s.op_size_ovr) {
        const imm = fetch16(s);
        s.regs[r] = (s.regs[r] & 0xFFFF0000) | @as(u32, imm);
    } else {
        s.regs[r] = fetch32(s);
    }
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
fn op0E(s: *CpuState) void { push32(s, 0x1B); }  // PUSH CS
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
fn opNop(_: *CpuState) void {}
fn opF4(s: *CpuState) void { s.halted = true; }  // HLT
fn opFC(s: *CpuState) void { setFlag(s, DF_BIT, false); }  // CLD
fn opFD(s: *CpuState) void { setFlag(s, DF_BIT, true); }   // STD
fn opF8(s: *CpuState) void { setFlag(s, CF_BIT, false); }  // CLC
fn opF9(s: *CpuState) void { setFlag(s, CF_BIT, true); }   // STC
fn opF5(s: *CpuState) void { setFlag(s, CF_BIT, !getFlag(s, CF_BIT)); }  // CMC
fn op9B(_: *CpuState) void {}   // WAIT/FWAIT
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
            updateFlagsArithW(s, @as(i64, al) - @as(i64, v), al, v, true, .w8);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1;
            if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const v = memRead8(s, s.regs[EDI]); const al: u8 = @truncate(s.regs[EAX]);
            updateFlagsArithW(s, @as(i64, al) - @as(i64, v), al, v, true, .w8);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1;
            if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const v = memRead8(s, s.regs[EDI]); const al: u8 = @truncate(s.regs[EAX]);
        updateFlagsArithW(s, @as(i64, al) - @as(i64, v), al, v, true, .w8);
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
            updateFlagsArithW(s, @as(i64, acc) - @as(i64, v), acc, v, true, .w32);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
            s.regs[ECX] -%= 1;
            if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const v: u32 = if (wide) memRead32(s, s.regs[EDI]) else @as(u32, memRead16(s, s.regs[EDI]));
            updateFlagsArithW(s, @as(i64, acc) - @as(i64, v), acc, v, true, .w32);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
            s.regs[ECX] -%= 1;
            if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const v: u32 = if (wide) memRead32(s, s.regs[EDI]) else @as(u32, memRead16(s, s.regs[EDI]));
        updateFlagsArithW(s, @as(i64, acc) - @as(i64, v), acc, v, true, .w32);
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + d);
    }
}
fn opA6(s: *CpuState) void { // CMPSB
    const rep = s.rep_prefix;
    if (rep == REP_REP) {
        while (s.regs[ECX] != 0) {
            const src = memRead8(s, s.regs[ESI]); const dst = memRead8(s, s.regs[EDI]);
            updateFlagsArithW(s, @as(i64, src) - @as(i64, dst), src, dst, true, .w8);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1; if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const src = memRead8(s, s.regs[ESI]); const dst = memRead8(s, s.regs[EDI]);
            updateFlagsArithW(s, @as(i64, src) - @as(i64, dst), src, dst, true, .w8);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
            s.regs[ECX] -%= 1; if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const src = memRead8(s, s.regs[ESI]); const dst = memRead8(s, s.regs[EDI]);
        updateFlagsArithW(s, @as(i64, src) - @as(i64, dst), src, dst, true, .w8);
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + strDir(s));
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + strDir(s));
    }
}
fn opA7(s: *CpuState) void { // CMPSD
    const di: i32 = strDir(s) * 4; const rep = s.rep_prefix;
    if (rep == REP_REP) {
        while (s.regs[ECX] != 0) {
            const src = memRead32(s, s.regs[ESI]); const dst = memRead32(s, s.regs[EDI]);
            updateFlagsArithW(s, @as(i64, src) - @as(i64, dst), src, dst, true, .w32);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + di);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + di);
            s.regs[ECX] -%= 1; if (!getFlag(s, ZF_BIT)) break;
        }
    } else if (rep == REP_REPNE) {
        while (s.regs[ECX] != 0) {
            const src = memRead32(s, s.regs[ESI]); const dst = memRead32(s, s.regs[EDI]);
            updateFlagsArithW(s, @as(i64, src) - @as(i64, dst), src, dst, true, .w32);
            s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + di);
            s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + di);
            s.regs[ECX] -%= 1; if (getFlag(s, ZF_BIT)) break;
        }
    } else {
        const src = memRead32(s, s.regs[ESI]); const dst = memRead32(s, s.regs[EDI]);
        updateFlagsArithW(s, @as(i64, src) - @as(i64, dst), src, dst, true, .w32);
        s.regs[ESI] = @bitCast(@as(i32, @bitCast(s.regs[ESI])) + di);
        s.regs[EDI] = @bitCast(@as(i32, @bitCast(s.regs[EDI])) + di);
    }
}

// ─── Group 4/5 opcodes ────────────────────────────────────────────────────────
fn opFE(s: *CpuState) void { // Group 4: INC/DEC rm8
    const d = decodeModRM(s); const res = readRm8Resolved(s, d.mod, d.rm);
    if (d.reg == 0) {
        writeRm8Resolved(s, res.is_reg, res.addr, res.value +% 1);
        const cf = getFlag(s, CF_BIT); updateFlagsArithW(s, @as(i64, res.value) + 1, res.value, 1, false, .w8); setFlag(s, CF_BIT, cf);
    } else if (d.reg == 1) {
        writeRm8Resolved(s, res.is_reg, res.addr, res.value -% 1);
        const cf = getFlag(s, CF_BIT); updateFlagsArithW(s, @as(i64, res.value) - 1, res.value, 1, true, .w8); setFlag(s, CF_BIT, cf);
    } else { s.faulted = true; s.halted = true; }
}
fn opFF(s: *CpuState) void { // Group 5: INC/DEC/CALL/JMP/PUSH rm32
    // decodeModRM+readRmFixed32Resolved must run exactly ONCE per instruction:
    // for a memory operand with a displacement/SIB, resolveRm() itself calls
    // fetch32/fetch8 to consume those trailing bytes and advances s.eip past
    // them. Calling a second Resolved helper here (as INC/DEC used to, to get
    // a width-aware read) re-fetches the same trailing bytes a second time,
    // overshooting EIP by the displacement's size -- confirmed live via
    // MCity_d.exe/authlogin.dll: `ff 05 <disp32>` (INC dword ptr [disp32])
    // landed EIP 4 bytes past the correct next instruction and faulted.
    // CALL/JMP/PUSH want the fixed-32 value/address as-is; INC/DEC derive
    // their width-aware value by masking `res.value` instead of re-resolving
    // -- resolveRm never consults op_size_ovr, so res.is_reg/res.addr are
    // identical to what a second Rmv-resolve would have produced anyway.
    const d = decodeModRM(s); const res = readRmFixed32Resolved(s, d.mod, d.rm);
    switch (d.reg) {
        0 => {
            const width: Width = if (s.op_size_ovr) .w16 else .w32;
            const rv_value: u32 = if (s.op_size_ovr) res.value & 0xFFFF else res.value;
            writeRmvResolved(s, res.is_reg, res.addr, rv_value +% 1);
            const cf = getFlag(s, CF_BIT); updateFlagsArithW(s, @as(i64, rv_value) + 1, rv_value, 1, false, width); setFlag(s, CF_BIT, cf);
        },
        1 => {
            const width: Width = if (s.op_size_ovr) .w16 else .w32;
            const rv_value: u32 = if (s.op_size_ovr) res.value & 0xFFFF else res.value;
            writeRmvResolved(s, res.is_reg, res.addr, rv_value -% 1);
            const cf = getFlag(s, CF_BIT); updateFlagsArithW(s, @as(i64, rv_value) - 1, rv_value, 1, true, width); setFlag(s, CF_BIT, cf);
        },
        2 => { push32(s, s.eip); s.eip = res.value; },  // CALL rm32
        4 => { s.eip = res.value; },                     // JMP rm32
        6 => { push32(s, res.value); },                  // PUSH rm32
        3, 5 => {},  // CALL/JMP far — not needed in flat model
        else => {},
    }
}

// ─── Dispatch table ───────────────────────────────────────────────────────────
const dispatch_table: [256]OpFn = dt: {
    @setEvalBranchQuota(20000);
    var t = [_]OpFn{opFault} ** 256;
    t[0x00] = op00; t[0x01] = op01; t[0x02] = op02; t[0x03] = op03; t[0x04] = op04; t[0x05] = op05;
    t[0x06] = op06; t[0x07] = op07; t[0x08] = op08; t[0x09] = op09; t[0x0A] = op0A; t[0x0B] = op0B;
    t[0x0C] = op0C; t[0x0D] = op0D; t[0x0E] = op0E; t[0x0F] = two_byte.op0F;
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
    t[0xC0] = opC0; t[0xC1] = opC1; t[0xC2] = opC2; t[0xC3] = opC3; t[0xC4] = opC4; t[0xC5] = opC5;
    t[0xC6] = opC6; t[0xC7] = opC7; t[0xC8] = opC8; t[0xC9] = opC9; t[0xCC] = opCC; t[0xCD] = opCD;
    t[0xD1] = opD1; t[0xD2] = opD2; t[0xD3] = opD3;
    t[0xD8] = fpu.opD8; t[0xD9] = fpu.opD9; t[0xDA] = fpu.opDA; t[0xDB] = fpu.opDB;
    t[0xDC] = fpu.opDC; t[0xDD] = fpu.opDD; t[0xDE] = fpu.opDE; t[0xDF] = fpu.opDF;
    t[0xE0] = opE0; t[0xE1] = opE1; t[0xE2] = opE2; t[0xE3] = opE3;
    t[0xE8] = opE8; t[0xE9] = opE9; t[0xEB] = opEB;
    t[0xF4] = opF4; t[0xF5] = opF5; t[0xF6] = opF6; t[0xF7] = opF7;
    t[0xF8] = opF8; t[0xF9] = opF9; t[0xFC] = opFC; t[0xFD] = opFD;
    t[0xFE] = opFE; t[0xFF] = opFF;
    break :dt t;
};

// ─── Execution engine ─────────────────────────────────────────────────────────
pub fn cpuStep(s: *CpuState) void {
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
    var mem = [_]u8{0x05, 0x05, 0x00, 0x00, 0x00} ++ [_]u8{0} ** 59;
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
    cpuStep(&cs);
    try testing.expectEqual(@as(u32, 0x1FC), cs.regs[ESP]);
    cs.regs[EAX] = 0;
    cpuStep(&cs);
    try testing.expectEqual(@as(u32, 0xDEADBEEF), cs.regs[EAX]);
    try testing.expectEqual(@as(u32, 0x200), cs.regs[ESP]);
}

test "XOR EAX, EAX zeroes register" {
    var mem = [_]u8{0x33, 0xC0} ++ [_]u8{0} ** 62;
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x12345678;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0), s.regs[EAX]);
    try testing.expect(getFlag(&s, ZF_BIT));
}

// ─── 8-bit logic-op SF regression tests ───────────────────────────────────────
// Found live: TEST AL,AL with AL=0xf8 (SHAPE_convertdepth's $f8 investigation,
// tew-fake-kernel-gaps memory) reported SF=false, when bit 7 of 0xf8 is
// clearly set. Root cause: op84/opA8/opF6-case0 (TEST) and the 8-bit OR/AND/
// XOR opcodes (op08/0A/0C, op20/22/24, op30/32, and op80's byte-immediate
// group1 cases 1/4/6) all called the 32-bit updateFlagsLogicW(, .w32) on a
// zero-extended 8-bit result -- (result & 0x80000000) can never be true for
// a byte 0x00-0xFF, so SF was always wrong whenever bit 7 of the real 8-bit
// result was set. Fixed by routing all of them through updateFlagsLogicW(, .w8)
// (already existed, already used correctly by the 8-bit shift group) instead.
// These tests exercise one opcode per mnemonic family, not all twelve call
// sites -- the fix is mechanical/identical across all of them.

test "TEST AL,AL sets SF when AL has bit 7 set (regression: was always false)" {
    var mem = [_]u8{0x84, 0xC0} ++ [_]u8{0} ** 62; // test al,al
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x000000f8; // AL = 0xf8, the exact byte from the live bug
    cpuStep(&s);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(!getFlag(&s, ZF_BIT));
    try testing.expectEqual(@as(u32, 0x000000f8), s.regs[EAX]); // TEST never writes back
}

test "TEST AL,imm8 sets SF when the masked result has bit 7 set" {
    var mem = [_]u8{0xA8, 0xFF} ++ [_]u8{0} ** 62; // test al,0xff
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000080;
    cpuStep(&s);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "AND rm8,r8 sets SF when the 8-bit result has bit 7 set" {
    var mem = [_]u8{0x20, 0xD8} ++ [_]u8{0} ** 62; // and al,bl
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x000000ff;
    s.regs[EBX] = 0x00000080;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "OR rm8,r8 sets SF when the 8-bit result has bit 7 set" {
    var mem = [_]u8{0x08, 0xD8} ++ [_]u8{0} ** 62; // or al,bl
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000000;
    s.regs[EBX] = 0x00000080;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "XOR rm8,r8 sets SF when the 8-bit result has bit 7 set" {
    var mem = [_]u8{0x30, 0xD8} ++ [_]u8{0} ** 62; // xor al,bl
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x0000000f;
    s.regs[EBX] = 0x0000008f;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "Group3 TEST rm8,imm8 sets SF when the masked result has bit 7 set" {
    var mem = [_]u8{0xF6, 0xC0, 0xFF} ++ [_]u8{0} ** 61; // test al,0xff
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000080;
    cpuStep(&s);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "8-bit logic ops still clear SF when bit 7 is not set (no regression the other way)" {
    var mem = [_]u8{0x84, 0xC0} ++ [_]u8{0} ** 62; // test al,al
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x0000007f;
    cpuStep(&s);
    try testing.expect(!getFlag(&s, SF_BIT));
    try testing.expect(!getFlag(&s, ZF_BIT));
}

// ─── 8-bit arithmetic-op SF/OF regression tests ───────────────────────────────
// Same discovery, same fix, wider blast radius: updateFlagsArith had no
// 8-bit sibling at all (unlike updateFlagsLogic8, which existed but wasn't
// always called) -- every 8-bit ADD/ADC/SBB/SUB/CMP/INC/DEC/NEG/CMPSB/SCASB
// zero-extended its result into the 32-bit-width SF/OF check, so e.g.
// ADD AL,1 with AL=0x7f (-> 0x80, a classic INT8 overflow) reported SF=0,
// OF=0 -- both wrong. One representative test per affected mnemonic, plus
// a no-overcorrection sanity check, not all 29 call sites individually --
// the fix is mechanical/identical across all of them (see updateFlagsArith8
// just above updateFlagsLogic8).

test "ADD AL,imm8 sets SF and OF on the classic 0x7f+1 INT8 overflow" {
    var mem = [_]u8{0x04, 0x01} ++ [_]u8{0} ** 62; // add al,1
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x0000007f;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
    try testing.expect(!getFlag(&s, CF_BIT));
    try testing.expect(!getFlag(&s, ZF_BIT));
}

test "SUB AL,imm8 sets SF and CF on 0x00-1 (borrow, result 0xff)" {
    var mem = [_]u8{0x2C, 0x01} ++ [_]u8{0} ** 62; // sub al,1
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000000;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x000000ff), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, CF_BIT));
    try testing.expect(!getFlag(&s, OF_BIT));
}

test "CMP AL,imm8 sets SF without modifying AL" {
    var mem = [_]u8{0x3C, 0x01} ++ [_]u8{0} ** 62; // cmp al,1
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000000;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000000), s.regs[EAX] & 0xFF); // unchanged
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, CF_BIT));
}

test "ADC AL,imm8 sets SF when the carry-in pushes the result past 0x7f" {
    var mem = [_]u8{0x14, 0x00} ++ [_]u8{0} ** 62; // adc al,0
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x0000007f;
    setFlag(&s, CF_BIT, true);
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "SBB AL,imm8 sets SF when the borrow-in pushes the result below 0" {
    var mem = [_]u8{0x1C, 0x00} ++ [_]u8{0} ** 62; // sbb al,0
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000000;
    setFlag(&s, CF_BIT, true);
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x000000ff), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
}

test "INC rm8 sets SF and OF on 0x7f -> 0x80, leaves CF untouched" {
    var mem = [_]u8{0xFE, 0xC0} ++ [_]u8{0} ** 62; // inc al
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x0000007f;
    setFlag(&s, CF_BIT, true); // INC must not touch CF -- verify it's preserved
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
    try testing.expect(getFlag(&s, CF_BIT)); // preserved, not recomputed
}

test "DEC rm8 sets OF on 0x80 -> 0x7f (INT8_MIN decrement overflow)" {
    var mem = [_]u8{0xFE, 0xC8} ++ [_]u8{0} ** 62; // dec al
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000080;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x0000007f), s.regs[EAX] & 0xFF);
    try testing.expect(!getFlag(&s, SF_BIT)); // 0x7f: bit 7 clear
    try testing.expect(getFlag(&s, OF_BIT));  // INT8_MIN -> INT8_MAX is the overflow case
}

test "NEG rm8 sets SF, CF, and OF on the INT8_MIN special case (0x80 negates to itself)" {
    var mem = [_]u8{0xF6, 0xD8} ++ [_]u8{0} ** 62; // neg al
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000080;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000080), s.regs[EAX] & 0xFF); // -128 wraps to itself
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, CF_BIT)); // NEG of a nonzero value always sets CF
    try testing.expect(getFlag(&s, OF_BIT));
}

test "SCASB sets SF when AL-[EDI] has bit 7 set" {
    var mem = [_]u8{0xAE} ++ [_]u8{0} ** 63; // scasb
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000000;
    s.regs[EDI] = 10;
    mem[10] = 0x01; // AL(0) - [EDI](1) = 0xff
    cpuStep(&s);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, CF_BIT));
}

test "8-bit arithmetic ops still clear SF/OF when there's no overflow (no regression the other way)" {
    var mem = [_]u8{0x04, 0x01} ++ [_]u8{0} ** 62; // add al,1
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000010;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000011), s.regs[EAX] & 0xFF);
    try testing.expect(!getFlag(&s, SF_BIT));
    try testing.expect(!getFlag(&s, OF_BIT));
    try testing.expect(!getFlag(&s, CF_BIT));
}

// ─── 16-bit (0x66-prefixed) operand-width regression tests ────────────────────
// Found while planning a full-ISA test suite: several opcodes read an operand
// via a hardcoded-32-bit helper while writing (or not writing) via a
// width-aware one that correctly checks op_size_ovr, or in the "r32,rm32"
// direction family didn't consult op_size_ovr at all. doGroup2 (shift/rotate)
// had a real memory-corruption risk on top: a 16-bit shift-by-zero on a
// memory operand wrote 4 bytes instead of 2. All fixed by routing through the
// already-width-aware readRmv/readRmvResolved/writeRmvResolved family and the
// new updateFlagsArithW/LogicW width parameter. One representative test per
// family/direction, not all ~10 call sites individually -- the fix is
// mechanical/identical across each family (ModRM 0xD8 = mod=11,reg=BX,rm=AX
// throughout, reused across the ADD/CMP/OR cases below).

test "ADD rmv,rv (0x01) is width-aware: 16-bit AX+BX overflow, upper 16 of EAX preserved" {
    var mem = [_]u8{0x66, 0x01, 0xD8} ++ [_]u8{0} ** 61; // add ax,bx
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x12347FFF; // AX=0x7FFF, upper 16 = sentinel 0x1234
    s.regs[EBX] = 0x00000001; // BX=1
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x12348000), s.regs[EAX]); // AX=0x8000, upper preserved
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
}

test "ADD rv,rmv (0x03) is width-aware: op_size_ovr was never consulted at all before" {
    var mem = [_]u8{0x66, 0x03, 0xD8} ++ [_]u8{0} ** 61; // add bx,ax
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EBX] = 0x12347FFF; // BX=0x7FFF, upper 16 = sentinel 0x1234
    s.regs[EAX] = 0x00000001; // AX=1
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x12348000), s.regs[EBX]); // BX=0x8000, upper preserved
    try testing.expectEqual(@as(u32, 0x00000001), s.regs[EAX]); // source untouched
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
}

test "CMP rmv,rv (0x39) is width-aware and does not write back" {
    var mem = [_]u8{0x66, 0x39, 0xD8} ++ [_]u8{0} ** 61; // cmp ax,bx
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00000000;
    s.regs[EBX] = 0x00000001;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000000), s.regs[EAX]); // unchanged
    try testing.expectEqual(@as(u32, 0x00000001), s.regs[EBX]); // unchanged
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, CF_BIT));
}

test "OR rmv,rv (0x09) is width-aware for both the read and the flag check" {
    var mem = [_]u8{0x66, 0x09, 0xD8} ++ [_]u8{0} ** 61; // or ax,bx
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x12340000; // AX=0, upper 16 = sentinel 0x1234
    s.regs[EBX] = 0x00008000; // BX=0x8000
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x12348000), s.regs[EAX]); // AX=0x8000, upper preserved
    try testing.expect(getFlag(&s, SF_BIT));
}

test "TEST rmv,rv (0x85) is width-aware for both the read and the flag check" {
    var mem = [_]u8{0x66, 0x85, 0xC0} ++ [_]u8{0} ** 61; // test ax,ax
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00008000; // AX=0x8000
    cpuStep(&s);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(!getFlag(&s, ZF_BIT));
}

test "XADD (0x0F 0xC1) is width-aware on both operands, not just the write" {
    var mem = [_]u8{0x66, 0x0F, 0xC1, 0xD8} ++ [_]u8{0} ** 60; // xadd ax,bx
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x12347FFF; // AX=0x7FFF, upper 16 = sentinel 0x1234
    s.regs[EBX] = 0x00000001; // BX=1
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x12348000), s.regs[EAX]); // AX = 0x7FFF+1, upper preserved
    try testing.expectEqual(@as(u32, 0x00007FFF), s.regs[EBX]); // BX = old AX
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
}

test "INC rm32 Group5 (0xFF /0) is width-aware: 16-bit 0x7FFF+1 overflow" {
    var mem = [_]u8{0x66, 0xFF, 0xC0} ++ [_]u8{0} ** 61; // inc ax
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x12347FFF; // AX=0x7FFF, upper 16 = sentinel 0x1234
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x12348000), s.regs[EAX]);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
}

test "INC dword ptr [disp32] (Group5 0xFF /0) advances EIP by 6, not 10 (regression: resolveRm's fetch32 consumed the disp32 twice)" {
    // Live-confirmed regression: authlogin.dll's SBH allocator does
    // `inc dword ptr ds:0x1000dbb8` (ff 05 <disp32>, no 0x66 prefix -- this
    // bug is NOT width-gated). opFF used to call a *second* Resolved helper
    // for INC/DEC after the upfront readRmFixed32Resolved, and resolveRm()
    // itself does fetch32() for mod=00,rm=101 (disp32-only) addressing --
    // so the displacement got fetched twice, landing EIP 4 bytes past the
    // real next instruction and decoding garbage there.
    var mem = [_]u8{ 0xFF, 0x05, 0x10, 0x00, 0x00, 0x00 } ++ [_]u8{0} ** 58; // inc dword ptr [0x10]
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    mem[0x10] = 5; mem[0x11] = 0; mem[0x12] = 0; mem[0x13] = 0;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 6), s.eip);
    try testing.expectEqual(@as(u8, 6), mem[0x10]);
}

test "DEC dword ptr [disp32] (Group5 0xFF /1) advances EIP by 6, not 10" {
    var mem = [_]u8{ 0xFF, 0x0D, 0x10, 0x00, 0x00, 0x00 } ++ [_]u8{0} ** 58; // dec dword ptr [0x10]
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    mem[0x10] = 5; mem[0x11] = 0; mem[0x12] = 0; mem[0x13] = 0;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 6), s.eip);
    try testing.expectEqual(@as(u8, 4), mem[0x10]);
}

test "INC word ptr [disp32] under 0x66 advances EIP by 7 and only writes 2 bytes" {
    var mem = [_]u8{ 0x66, 0xFF, 0x05, 0x10, 0x00, 0x00, 0x00 } ++ [_]u8{0} ** 57; // inc word ptr [0x10]
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    mem[0x10] = 0xFF; mem[0x11] = 0x7F; mem[0x12] = 0xAA; // sentinel byte after the 16-bit operand
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 7), s.eip);
    try testing.expectEqual(@as(u8, 0x00), mem[0x10]);
    try testing.expectEqual(@as(u8, 0x80), mem[0x11]);
    try testing.expectEqual(@as(u8, 0xAA), mem[0x12]); // must be untouched
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
}

test "INC dword ptr [disp8] (Group5 0xFF /0) advances EIP by 3, not 7" {
    var mem = [_]u8{ 0xFF, 0x45, 0x10 } ++ [_]u8{0} ** 61; // inc dword ptr [ebp+0x10]
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EBP] = 0;
    mem[0x10] = 5; mem[0x11] = 0; mem[0x12] = 0; mem[0x13] = 0;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 3), s.eip);
    try testing.expectEqual(@as(u8, 6), mem[0x10]);
}

test "INC ECX (Group5 0xFF /0, register-direct) still works after the resolveRm double-fetch fix" {
    var mem = [_]u8{0xFF, 0xC1} ++ [_]u8{0} ** 62; // inc ecx
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[ECX] = 0x7FFFFFFF;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 2), s.eip);
    try testing.expectEqual(@as(u32, 0x80000000), s.regs[ECX]);
    try testing.expect(getFlag(&s, SF_BIT));
    try testing.expect(getFlag(&s, OF_BIT));
}

test "doGroup2 memory-corruption regression: 16-bit shift-by-zero writes only 2 bytes, not 4" {
    // shl word ptr [0x00000020], cl -- with CL=0, hits the count==0
    // early-return path, which used to write via the hardcoded-32-bit
    // writeRmFixed32Resolved regardless of op_size_ovr.
    var mem = [_]u8{ 0x66, 0xD3, 0x25, 0x20, 0x00, 0x00, 0x00 } ++ [_]u8{0} ** 57;
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[ECX] = 0; // shift count 0
    mem[0x20] = 0x34;
    mem[0x21] = 0x12; // target 16-bit word = 0x1234
    mem[0x22] = 0xAA; // sentinel byte immediately after the 2-byte operand
    cpuStep(&s);
    try testing.expectEqual(@as(u8, 0x34), mem[0x20]);
    try testing.expectEqual(@as(u8, 0x12), mem[0x21]);
    try testing.expectEqual(@as(u8, 0xAA), mem[0x22]); // must be untouched
}

test "doGroup2 ROL AX,1 wraps within 16 bits, not 32" {
    var mem = [_]u8{0x66, 0xD1, 0xC0} ++ [_]u8{0} ** 61; // rol ax,1
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00008001;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x00000003), s.regs[EAX] & 0xFFFF);
}

test "doGroup2 SAR AX,1 sign-extends within 16 bits, not 32" {
    var mem = [_]u8{0x66, 0xD1, 0xF8} ++ [_]u8{0} ** 61; // sar ax,1
    var s = CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[EAX] = 0x00008000;
    cpuStep(&s);
    try testing.expectEqual(@as(u32, 0x0000C000), s.regs[EAX] & 0xFFFF);
}

