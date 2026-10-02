// SPDX-License-Identifier: LGPL-3.0-or-later
// fpu.zig — x87 FPU helpers and opcode handlers (D8–DF).
//
// The FPU stack is [8]f80 (80-bit extended precision).  This is intentional:
// x87 hardware uses 64-bit mantissa internally, which means all 64-bit integers
// are representable exactly.  f64 only has 53 bits of mantissa, which caused
// FILD m64 / FISTP m64 to silently corrupt data for values > 2^53.
const std = @import("std");
const core = @import("core.zig");
const CpuState = core.CpuState;

// ─── x87 float -> integer stores (FIST/FISTP/FISTTP) ─────────────────────────
// Found live 2026-09-18: a guest FISTP whose source was NaN/Inf/out of range hit
// `@intFromFloat`, which PANICS the entire host process (Zig safety check) --
// the emulator itself aborted (SIGABRT, exit 134) instead of the guest seeing
// anything. Real x87 never traps here: with the invalid-operation exception
// masked (the state every user-mode program runs in) it stores the "integer
// indefinite" -- the most negative value of the destination width -- and sets
// IE in the status word. Also honors the control word's rounding-control bits
// (default round-to-nearest-EVEN; previously FIST/FISTP m32/m16 rounded
// half-away-from-zero and FISTP m64 always truncated, ignoring FLDCW).
const RoundMode = enum { nearest_even, down, up, truncate };

fn roundModeFromCw(cw: u16) RoundMode {
    return switch ((cw >> 10) & 3) {
        0 => .nearest_even,
        1 => .down,
        2 => .up,
        else => .truncate,
    };
}

fn roundToIntegral(x: f80, mode: RoundMode) f80 {
    return switch (mode) {
        .truncate => @trunc(x),
        .down => @floor(x),
        .up => @ceil(x),
        .nearest_even => blk: {
            const f = @floor(x);
            const diff = x - f;
            if (diff < 0.5) break :blk f;
            if (diff > 0.5) break :blk f + 1;
            break :blk if (@floor(f / 2.0) * 2.0 == f) f else f + 1; // exact tie: even neighbour
        },
    };
}

fn noteFistInvalid(s: *CpuState, x: f80) void {
    s.fpu_status_word |= 0x0001; // IE
    s.fist_invalid_count +%= 1;
    s.fist_invalid_eip = s.last_instr_eip;
    s.fist_invalid_val = x;
    // Snapshot up to three caller return addresses via the guest's EBP chain.
    // Raw, bounds-checked reads only: this runs inside an instruction handler and
    // must never fault or panic on a garbage EBP.
    s.fist_invalid_ret = .{ 0, 0, 0 };
    var ebp: u32 = s.regs[core.EBP];
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        if (ebp == 0 or @as(u64, ebp) + 8 > s.memory_size) break;
        const e: usize = ebp;
        s.fist_invalid_ret[i] = std.mem.readInt(u32, s.memory[e + 4 ..][0..4], .little);
        ebp = std.mem.readInt(u32, s.memory[e ..][0..4], .little);
    }
}

fn fistConvert(comptime T: type, s: *CpuState, x: f80, mode: RoundMode) T {
    const indefinite: T = std.math.minInt(T);
    if (std.math.isNan(x) or std.math.isInf(x)) {
        noteFistInvalid(s, x);
        return indefinite;
    }
    const r = roundToIntegral(x, mode);
    const lo: f80 = @floatFromInt(std.math.minInt(T));
    const hi: f80 = @floatFromInt(std.math.maxInt(T));
    if (r < lo or r > hi) {
        noteFistInvalid(s, x);
        return indefinite;
    }
    return @intFromFloat(r);
}

// ─── FPU stack helpers ────────────────────────────────────────────────────────
inline fn fpuGet(s: *CpuState, i: u8) f80 {
    return s.fpu_stack[(@as(u8, @truncate(s.fpu_top)) +% i) & 7];
}
inline fn fpuSet(s: *CpuState, i: u8, v: f80) void {
    const idx: u8 = (@as(u8, @truncate(s.fpu_top)) +% i) & 7;
    s.fpu_stack[idx] = v;
    s.fpu_tag_word &= ~(@as(u16, 3) << (@as(u4, @truncate(idx)) * 2));
}
fn fpuPush(s: *CpuState, v: f80) void {
    s.fpu_top = (s.fpu_top -% 1) & 7;
    s.fpu_stack[s.fpu_top] = v;
    s.fpu_tag_word &= ~(@as(u16, 3) << (@as(u4, @truncate(s.fpu_top)) * 2));
    s.fpu_status_word = (s.fpu_status_word & ~@as(u16, 0x3800)) |
                        @as(u16, @truncate((s.fpu_top & 7) << 11));
}
// Pop the emulated FPU stack. Deliberately returns void: this used to be
// `fn fpuPop(s) f80` called as `_ = fpuPop(s);`, and on x86/x86-64 an f80 return
// value lives in ST(0) of the HOST x87 stack -- Zig does not pop a DISCARDED x87
// return, so every popping handler (FSTP/FISTP/FCOMP/FADDP/...) leaked one host
// x87 entry. After 8 leaks the next host `fld` overflows the stack into a QNaN:
// the 2026-09 "FMUL returns NaN on one call in many" / screen.c(475) mystery.
// Never write `_ = <f80-returning fn>()` in this file; if a popped value is ever
// needed, read it with fpuGet(s, 0) BEFORE calling fpuDrop.
fn fpuDrop(s: *CpuState) void {
    s.fpu_tag_word |= @as(u16, 3) << (@as(u4, @truncate(s.fpu_top & 7)) * 2);
    s.fpu_top = (s.fpu_top +% 1) & 7;
    s.fpu_status_word = (s.fpu_status_word & ~@as(u16, 0x3800)) |
                        @as(u16, @truncate((s.fpu_top & 7) << 11));
}
fn fpuSetCC(s: *CpuState, c3: bool, c2: bool, c0: bool) void {
    s.fpu_status_word &= ~@as(u16, 0x4500);
    if (c0) s.fpu_status_word |= 0x0100;
    if (c2) s.fpu_status_word |= 0x0400;
    if (c3) s.fpu_status_word |= 0x4000;
}
fn fpuCompare(s: *CpuState, a: f80, b: f80) void {
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
fn fpuComi(s: *CpuState, a: f80, b: f80, do_pop: bool) void {
    // Unordered sets ZF=PF=CF=1; otherwise PF=0 (so `jp` after FCOMI works).
    if (std.math.isNan(a) or std.math.isNan(b)) {
        core.setFlag(s, core.ZF_BIT, true); core.setFlag(s, core.PF_BIT, true); core.setFlag(s, core.CF_BIT, true);
    } else if (a > b) {
        core.setFlag(s, core.ZF_BIT, false); core.setFlag(s, core.PF_BIT, false); core.setFlag(s, core.CF_BIT, false);
    } else if (a < b) {
        core.setFlag(s, core.ZF_BIT, false); core.setFlag(s, core.PF_BIT, false); core.setFlag(s, core.CF_BIT, true);
    } else {
        core.setFlag(s, core.ZF_BIT, true); core.setFlag(s, core.PF_BIT, false); core.setFlag(s, core.CF_BIT, false);
    }
    core.setFlag(s, core.OF_BIT, false);
    if (do_pop) fpuDrop(s);
}

// ─── D9 register-form helpers: FXAM, FPREM/FPREM1, range-checked trig ────────
// Found 2026-09-30 (MCity HOME avatar never drawing): FPTAN/FPATAN/FXTRACT/
// FYL2XP1 were silent no-ops (FPTAN also skipped its push, unbalancing the
// stack), FXAM cleared C3/C2/C0 so every value classified as "unsupported",
// and FPREM/FPREM1 never touched C2 or the quotient bits. The CRT's math
// dispatcher classifies its arguments with FXAM.

// Zig's std has no f80 atan2/log1p; the library links libc, whose long-double
// versions are the full 80-bit x87 format on x86-64.
extern fn atan2l(y: c_longdouble, x: c_longdouble) c_longdouble;
extern fn log1pl(x: c_longdouble) c_longdouble;

inline fn stTagEmpty(s: *CpuState, i: u8) bool {
    const idx: u8 = (@as(u8, @truncate(s.fpu_top)) +% i) & 7;
    return ((s.fpu_tag_word >> @as(u4, @truncate(idx * 2))) & 3) == 3;
}

// FXAM class codes in C3/C2/C0; C1 = sign.
fn fxam(s: *CpuState) void {
    const x = fpuGet(s, 0);
    s.fpu_status_word &= ~@as(u16, 0x0200);
    if (std.math.signbit(x)) s.fpu_status_word |= 0x0200;
    if (stTagEmpty(s, 0)) return fpuSetCC(s, true, false, true); // empty: 101
    if (std.math.isNan(x)) return fpuSetCC(s, false, false, true); // NaN: 001
    if (std.math.isInf(x)) return fpuSetCC(s, false, true, true); // infinity: 011
    if (x == 0) return fpuSetCC(s, true, false, false); // zero: 100
    if (!std.math.isNormal(x)) return fpuSetCC(s, true, true, false); // denormal: 110
    fpuSetCC(s, false, true, false); // normal: 010
}

// FPREM (truncating quotient, like C fmod) and FPREM1 (IEEE round-to-nearest-
// even quotient). The reduction is always completed in one step, so C2 = 0;
// C0/C3/C1 get the quotient's low three bits (Q2/Q1/Q0). The remainder itself
// is exact (@rem); the quotient bits are exact while |ST0/ST1| < 2^64, which is
// the range where hardware also completes the reduction in one step.
fn fprem(s: *CpuState, ieee: bool) void {
    const x = fpuGet(s, 0);
    const y = fpuGet(s, 1);
    if (std.math.isNan(x) or std.math.isNan(y) or std.math.isInf(x) or y == 0) {
        s.fpu_status_word |= 0x0001; // IE; result is the default NaN
        fpuSet(s, 0, std.math.nan(f80));
        return fpuSetCC(s, false, false, false);
    }
    if (std.math.isInf(y)) { // x unchanged, quotient 0
        s.fpu_status_word &= ~@as(u16, 0x0200);
        return fpuSetCC(s, false, false, false);
    }
    var r = @rem(x, y);
    const qf = @abs(@round((x - r) / y));
    var q: u64 = if (qf < 18446744073709551616.0) @intFromFloat(qf) else 0;
    if (ieee) {
        const ay = @abs(y);
        const ar = @abs(r);
        const half = ay / 2;
        if (ar > half or (ar == half and (q & 1) == 1)) {
            r = if (r > 0) r - ay else r + ay;
            q +%= 1;
        }
    }
    fpuSet(s, 0, r);
    s.fpu_status_word &= ~@as(u16, 0x0200);
    if ((q & 1) != 0) s.fpu_status_word |= 0x0200; // C1 = Q0
    fpuSetCC(s, (q & 2) != 0, false, (q & 4) != 0); // C3 = Q1, C0 = Q2
}

// FSIN/FCOS/FSINCOS/FPTAN accept |x| < 2^63. Out of range: C2 = 1 and the
// operand is left unchanged (software then reduces with FPREM and retries).
fn trigInRange(s: *CpuState, x: f80) bool {
    if (!std.math.isNan(x) and @abs(x) >= 9223372036854775808.0) {
        s.fpu_status_word |= 0x0400;
        return false;
    }
    s.fpu_status_word &= ~@as(u16, 0x0400);
    return true;
}

// FXTRACT: ST0 = x -> ST1 = unbiased exponent, ST0 = significand in [1,2).
fn fxtract(s: *CpuState) void {
    const x = fpuGet(s, 0);
    if (x == 0) {
        s.fpu_status_word |= 0x0004; // ZE
        fpuSet(s, 0, -std.math.inf(f80));
        fpuPush(s, x);
        return;
    }
    if (std.math.isNan(x) or std.math.isInf(x)) {
        fpuSet(s, 0, if (std.math.isNan(x)) x else std.math.inf(f80));
        fpuPush(s, x);
        return;
    }
    const fr = std.math.frexp(x); // x = m * 2^e, |m| in [0.5, 1)
    fpuSet(s, 0, @as(f80, @floatFromInt(fr.exponent - 1)));
    fpuPush(s, fr.significand * 2);
}

// ─── Float memory I/O ─────────────────────────────────────────────────────────
fn readFloat(s: *CpuState, addr: u32) f32 { return @bitCast(core.memRead32(s, addr)); }
fn writeFloat(s: *CpuState, addr: u32, v: f32) void { core.memWrite32(s, addr, @bitCast(v)); }
fn readDouble(s: *CpuState, addr: u32) f64 {
    const lo = core.memRead32(s, addr);
    const hi = core.memRead32(s, addr + 4);
    const bits: u64 = (@as(u64, hi) << 32) | @as(u64, lo);
    return @bitCast(bits);
}
fn writeDouble(s: *CpuState, addr: u32, v: f64) void {
    const bits: u64 = @bitCast(v);
    core.memWrite32(s, addr, @truncate(bits));
    core.memWrite32(s, addr + 4, @truncate(bits >> 32));
}

// ─── FPU constants (FLDL2T, FLDL2E, FLDPI, FLDLG2, FLDLN2, FLDZ) ───────────
const FPU_CONSTS = [7]f80{ 1.0, 3.3219280948873626, 1.4426950408889634,
    std.math.pi, 0.3010299956639812, std.math.ln2, 0.0 };

// ─── FPU opcode handlers ──────────────────────────────────────────────────────

pub fn opD8(s: *CpuState) void { // float32 ops
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        const st0 = fpuGet(s, 0); const sti = fpuGet(s, d.rm);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + sti), 1 => fpuSet(s, 0, st0 * sti),
            2 => fpuCompare(s, st0, sti), 3 => { fpuCompare(s, st0, sti); fpuDrop(s); },
            4 => fpuSet(s, 0, st0 - sti), 5 => fpuSet(s, 0, sti - st0),
            6 => fpuSet(s, 0, st0 / sti), 7 => fpuSet(s, 0, sti / st0),
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const val: f80 = readFloat(s, addr); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val),
            // The 2026-09-03 captureHostFpuState debug hook that used to run here
            // (inline-asm fstpt on the HOST x87 stack, every FMUL m32) was removed
            // 2026-09-18: it popped an empty host stack (its own earlier analysis
            // said so) and sat inside the very instruction that intermittently
            // returned NaN -- suspected of causing, not observing, the NaN.
            1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); fpuDrop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opD9(s: *CpuState) void { // FLD/FST/FSTP/constants/misc
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => fpuPush(s, fpuGet(s, d.rm)),  // FLD ST(i)
            1 => { const t = fpuGet(s, 0); fpuSet(s, 0, fpuGet(s, d.rm)); fpuSet(s, d.rm, t); },  // FXCH
            2 => if (d.rm == 0) {} else core.opFault(s),  // FNOP; D9 D1-D7 reserved
            3 => { fpuSet(s, d.rm, fpuGet(s, 0)); fpuDrop(s); },  // FSTP ST(i)
            4 => switch (d.rm) {
                0 => fpuSet(s, 0, -fpuGet(s, 0)),  // FCHS
                1 => fpuSet(s, 0, @abs(fpuGet(s, 0))),  // FABS
                4 => fpuCompare(s, fpuGet(s, 0), 0.0),  // FTST
                5 => fxam(s),  // FXAM
                else => core.opFault(s),  // D9 E2/E3/E6/E7 reserved
            },
            5 => if (d.rm < 7) fpuPush(s, FPU_CONSTS[d.rm]) else core.opFault(s),  // FLD constants; D9 EF reserved
            6 => switch (d.rm) {
                0 => fpuSet(s, 0, std.math.exp2(fpuGet(s, 0)) - 1.0),  // F2XM1
                1 => { const x = fpuGet(s, 0); const y = fpuGet(s, 1); fpuDrop(s); fpuSet(s, 0, y * std.math.log2(x)); },  // FYL2X
                2 => { const v = fpuGet(s, 0); if (trigInRange(s, v)) { fpuSet(s, 0, @tan(v)); fpuPush(s, 1.0); } },  // FPTAN
                3 => { const x = fpuGet(s, 0); const y = fpuGet(s, 1); fpuDrop(s); fpuSet(s, 0, atan2l(y, x)); },  // FPATAN: ST1 = atan2(ST1, ST0), pop
                4 => fxtract(s),  // FXTRACT
                5 => fprem(s, true),  // FPREM1
                6 => { s.fpu_top = (s.fpu_top -% 1) & 7; s.fpu_status_word = (s.fpu_status_word & ~@as(u16,0x3800)) | @as(u16, @truncate((s.fpu_top & 7) << 11)); },  // FDECSTP
                7 => { s.fpu_top = (s.fpu_top +% 1) & 7; s.fpu_status_word = (s.fpu_status_word & ~@as(u16,0x3800)) | @as(u16, @truncate((s.fpu_top & 7) << 11)); },  // FINCSTP
                else => unreachable, // rm is 3 bits
            },
            7 => switch (d.rm) {
                0 => fprem(s, false),  // FPREM
                1 => { const x = fpuGet(s, 0); const y = fpuGet(s, 1); fpuDrop(s); fpuSet(s, 0, y * (log1pl(x) / std.math.ln2)); },  // FYL2XP1: ST1 = ST1 * log2(ST0 + 1), pop
                2 => fpuSet(s, 0, @sqrt(fpuGet(s, 0))),  // FSQRT
                3 => { const v = fpuGet(s, 0); if (trigInRange(s, v)) { fpuSet(s, 0, @sin(v)); fpuPush(s, @cos(v)); } },  // FSINCOS
                4 => fpuSet(s, 0, roundToIntegral(fpuGet(s, 0), roundModeFromCw(s.fpu_control_word))),  // FRNDINT (honors RC, default nearest-even)
                5 => { // FSCALE: ST0 *= 2^trunc(ST1). Stays in f80 (no @intFromFloat, which panics the host on NaN/Inf/huge ST1); the clamp only bounds exp2 -- 2^+-20000 already over/underflows f80.
                    const t = @trunc(fpuGet(s, 1));
                    const st0 = fpuGet(s, 0);
                    if (std.math.isNan(t)) fpuSet(s, 0, t)
                    else if (st0 != 0 and !std.math.isInf(st0)) fpuSet(s, 0, st0 * std.math.exp2(std.math.clamp(t, @as(f80, -20000), @as(f80, 20000))));
                },
                6 => { const v = fpuGet(s, 0); if (trigInRange(s, v)) fpuSet(s, 0, @sin(v)); },  // FSIN
                7 => { const v = fpuGet(s, 0); if (trigInRange(s, v)) fpuSet(s, 0, @cos(v)); },  // FCOS
                else => unreachable, // rm is 3 bits
            },
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, readFloat(s, addr)),  // FLD m32
            2 => writeFloat(s, addr, @floatCast(fpuGet(s, 0))),  // FST m32
            3 => { writeFloat(s, addr, @floatCast(fpuGet(s, 0))); fpuDrop(s); },  // FSTP m32
            4 => core.opFault(s),  // FLDENV: not implemented -- fail loudly, not a silent NOP
            5 => s.fpu_control_word = core.memRead16(s, addr),  // FLDCW
            6 => core.opFault(s),  // FNSTENV: not implemented -- fail loudly, not a silent NOP
            7 => core.memWrite16(s, addr, s.fpu_control_word),  // FNSTCW
            else => {},
        }
    }
}

pub fn opDA(s: *CpuState) void { // int32 ops / FCMOV
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => { if (core.getFlag(s, core.CF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVB
            1 => { if (core.getFlag(s, core.ZF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVE
            2 => { if (core.getFlag(s, core.CF_BIT) or core.getFlag(s, core.ZF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVBE
            3 => { if (core.getFlag(s, core.PF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVU
            5 => if (d.rm == 1) { fpuCompare(s, fpuGet(s, 0), fpuGet(s, 1)); fpuDrop(s); fpuDrop(s); },  // FUCOMPP
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const val: f80 = @floatFromInt(core.memReadS32(s, addr)); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); fpuDrop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opDB(s: *CpuState) void { // FILD/FISTP int32, FCLEX/FINIT, FUCOMI
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        if (d.reg < 4) { // FCMOVNB / FCMOVNE / FCMOVNBE / FCMOVNU
            const cf = core.getFlag(s, core.CF_BIT); const zf = core.getFlag(s, core.ZF_BIT); const pf = core.getFlag(s, core.PF_BIT);
            const take = switch (d.reg) { 0 => !cf, 1 => !zf, 2 => !cf and !zf, else => !pf };
            if (take) fpuSet(s, 0, fpuGet(s, d.rm));
        } else if (d.reg == 4) {
            if (d.rm == 2) { s.fpu_status_word &= 0x7F00; }  // FCLEX
            else if (d.rm == 3) { s.fpu_control_word = 0x037F; s.fpu_status_word = 0; s.fpu_tag_word = 0xFFFF; s.fpu_top = 0; }  // FINIT
        } else if (d.reg == 5) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), false)  // FUCOMI
        else if (d.reg == 6) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), false);  // FCOMI
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, @floatFromInt(core.memReadS32(s, addr))),  // FILD m32
            1 => { core.memWrite32(s, addr, @bitCast(fistConvert(i32, s, fpuGet(s, 0), .truncate))); fpuDrop(s); },  // FISTTP
            2 => { core.memWrite32(s, addr, @bitCast(fistConvert(i32, s, fpuGet(s, 0), roundModeFromCw(s.fpu_control_word)))); },  // FIST
            3 => { core.memWrite32(s, addr, @bitCast(fistConvert(i32, s, fpuGet(s, 0), roundModeFromCw(s.fpu_control_word)))); fpuDrop(s); },  // FISTP
            5 => { // FLD m80real: Zig's f80 has the x87 extended layout, so the 10 bytes are the value's bits
                const bits: u80 = (@as(u80, core.memRead16(s, addr + 8)) << 64) | (@as(u80, core.memRead32(s, addr + 4)) << 32) | @as(u80, core.memRead32(s, addr));
                fpuPush(s, @bitCast(bits));
            },
            7 => { // FSTP m80real
                const bits: u80 = @bitCast(fpuGet(s, 0));
                core.memWrite32(s, addr, @truncate(bits));
                core.memWrite32(s, addr + 4, @truncate(bits >> 32));
                core.memWrite16(s, addr + 8, @truncate(bits >> 64));
                fpuDrop(s);
            },
            else => {},
        }
    }
}

pub fn opDC(s: *CpuState) void { // float64 ops (reversed operands)
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        const st0 = fpuGet(s, 0); const sti = fpuGet(s, d.rm);
        switch (d.reg) {
            0 => fpuSet(s, d.rm, sti + st0), 1 => fpuSet(s, d.rm, sti * st0),
            2 => fpuCompare(s, st0, sti), 3 => { fpuCompare(s, st0, sti); fpuDrop(s); },
            // DC E0+i FSUBR / E8+i FSUB / F0+i FDIVR / F8+i FDIV (ST(i) is the destination)
            4 => fpuSet(s, d.rm, st0 - sti), 5 => fpuSet(s, d.rm, sti - st0),
            6 => fpuSet(s, d.rm, st0 / sti), 7 => fpuSet(s, d.rm, sti / st0),
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const val: f80 = @floatCast(readDouble(s, addr)); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); fpuDrop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opDD(s: *CpuState) void { // FLD/FST/FSTP float64, FUCOM
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => { const idx = ((@as(u8, @truncate(s.fpu_top)) +% d.rm) & 7); s.fpu_tag_word |= @as(u16, 3) << (@as(u4, @truncate(idx)) * 2); },  // FFREE
            2 => fpuSet(s, d.rm, fpuGet(s, 0)),  // FST
            3 => { fpuSet(s, d.rm, fpuGet(s, 0)); fpuDrop(s); },  // FSTP
            4 => fpuCompare(s, fpuGet(s, 0), fpuGet(s, d.rm)),  // FUCOM
            5 => { fpuCompare(s, fpuGet(s, 0), fpuGet(s, d.rm)); fpuDrop(s); },  // FUCOMP
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, @floatCast(readDouble(s, addr))),  // FLD m64
            1 => { writeDouble(s, addr, @floatCast(@trunc(fpuGet(s, 0)))); fpuDrop(s); },  // FISTTP m64
            2 => writeDouble(s, addr, @floatCast(fpuGet(s, 0))),  // FST m64
            3 => { writeDouble(s, addr, @floatCast(fpuGet(s, 0))); fpuDrop(s); },  // FSTP m64
            4, 6 => core.opFault(s),  // FRSTOR/FNSAVE: not implemented -- fail loudly, not a silent NOP
            7 => core.memWrite16(s, addr, s.fpu_status_word),  // FNSTSW m16
            else => {},
        }
    }
}

pub fn opDE(s: *CpuState) void { // FADDP/FMULP/etc / int16
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        const st0 = fpuGet(s, 0); const sti = fpuGet(s, d.rm);
        switch (d.reg) {
            0 => { fpuSet(s, d.rm, sti + st0); fpuDrop(s); },  // FADDP
            1 => { fpuSet(s, d.rm, sti * st0); fpuDrop(s); },  // FMULP
            2 => { fpuCompare(s, st0, sti); fpuDrop(s); },
            3 => if (d.rm == 1) { fpuCompare(s, st0, fpuGet(s, 1)); fpuDrop(s); fpuDrop(s); },  // FCOMPP
            4 => { fpuSet(s, d.rm, st0 - sti); fpuDrop(s); },  // FSUBRP
            5 => { fpuSet(s, d.rm, sti - st0); fpuDrop(s); },  // FSUBP
            6 => { fpuSet(s, d.rm, st0 / sti); fpuDrop(s); },  // FDIVRP
            7 => { fpuSet(s, d.rm, sti / st0); fpuDrop(s); },  // FDIVP
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const raw = core.memRead16(s, addr); const val: f80 = @floatFromInt(@as(i16, @bitCast(raw)));
        const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); fpuDrop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opDF(s: *CpuState) void { // FILD/FISTP int16/int64, FNSTSW AX, FUCOMIP
    core.hostFpuCheck(s);
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        if (d.reg == 4 and d.rm == 0) {  // FNSTSW AX
            s.regs[core.EAX] = (s.regs[core.EAX] & 0xFFFF0000) | @as(u32, s.fpu_status_word);
        } else if (d.reg == 5) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), true)   // FUCOMIP
        else if (d.reg == 6) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), true);   // FCOMIP
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => { const raw = core.memRead16(s, addr); fpuPush(s, @floatFromInt(@as(i16, @bitCast(raw)))); },  // FILD m16
            1 => { core.memWrite16(s, addr, @bitCast(fistConvert(i16, s, fpuGet(s, 0), .truncate))); fpuDrop(s); },  // FISTTP m16
            2 => { core.memWrite16(s, addr, @bitCast(fistConvert(i16, s, fpuGet(s, 0), roundModeFromCw(s.fpu_control_word)))); },  // FIST m16
            3 => { core.memWrite16(s, addr, @bitCast(fistConvert(i16, s, fpuGet(s, 0), roundModeFromCw(s.fpu_control_word)))); fpuDrop(s); },  // FISTP m16
            // FILD m64: f80 has 64-bit mantissa — all i64 values are exact, no precision loss.
            5 => { const lo = core.memRead32(s, addr); const hi = core.memReadS32(s, addr + 4); fpuPush(s, @as(f80, @floatFromInt(@as(i64, hi) * @as(i64, 0x100000000) + @as(i64, lo)))); },  // FILD m64
            // FISTP m64: f80→i64 is exact for all representable integers.
            4, 6 => core.opFault(s),  // FBLD / FBSTP (BCD): not implemented -- fail loudly
            7 => { const val = fpuGet(s, 0); const bits: u64 = @bitCast(fistConvert(i64, s, val, roundModeFromCw(s.fpu_control_word))); core.memWrite32(s, addr, @truncate(bits)); core.memWrite32(s, addr + 4, @truncate(bits >> 32)); fpuDrop(s); },  // FISTP m64
            else => {},
        }
    }
}

// ─── Exported accessors for C API (f64 interface for Python compatibility) ───
pub fn fpuGetF64(s: *CpuState, i: u32) f64 {
    return if (i < 8) @floatCast(s.fpu_stack[i]) else 0.0;
}
pub fn fpuSetF64(s: *CpuState, i: u32, val: f64) void {
    if (i < 8) s.fpu_stack[i] = @floatCast(val);
}
