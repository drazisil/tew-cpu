// fpu.zig — x87 FPU helpers and opcode handlers (D8–DF).
//
// The FPU stack is [8]f80 (80-bit extended precision).  This is intentional:
// x87 hardware uses 64-bit mantissa internally, which means all 64-bit integers
// are representable exactly.  f64 only has 53 bits of mantissa, which caused
// FILD m64 / FISTP m64 to silently corrupt data for values > 2^53.
const std = @import("std");
const core = @import("core.zig");
const CpuState = core.CpuState;

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
fn fpuPop(s: *CpuState) f80 {
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
    if (std.math.isNan(a) or std.math.isNan(b)) {
        core.setFlag(s, core.ZF_BIT, true); core.setFlag(s, core.CF_BIT, true);
    } else if (a > b) {
        core.setFlag(s, core.ZF_BIT, false); core.setFlag(s, core.CF_BIT, false);
    } else if (a < b) {
        core.setFlag(s, core.ZF_BIT, false); core.setFlag(s, core.CF_BIT, true);
    } else {
        core.setFlag(s, core.ZF_BIT, true); core.setFlag(s, core.CF_BIT, false);
    }
    core.setFlag(s, core.OF_BIT, false);
    if (do_pop) _ = fpuPop(s);
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
    const d = core.decodeModRM(s);
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
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const val: f80 = readFloat(s, addr); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opD9(s: *CpuState) void { // FLD/FST/FSTP/constants/misc
    const d = core.decodeModRM(s);
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
                5 => { const sc: i64 = @intFromFloat(@trunc(fpuGet(s, 1))); fpuSet(s, 0, fpuGet(s, 0) * std.math.exp2(@as(f80, @floatFromInt(sc)))); },  // FSCALE
                6 => fpuSet(s, 0, @sin(fpuGet(s, 0))),  // FSIN
                7 => fpuSet(s, 0, @cos(fpuGet(s, 0))),  // FCOS
                else => {},
            },
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, readFloat(s, addr)),  // FLD m32
            2 => writeFloat(s, addr, @floatCast(fpuGet(s, 0))),  // FST m32
            3 => { writeFloat(s, addr, @floatCast(fpuGet(s, 0))); _ = fpuPop(s); },  // FSTP m32
            4 => {},  // FLDENV NOP
            5 => s.fpu_control_word = core.memRead16(s, addr),  // FLDCW
            6 => {},  // FNSTENV NOP
            7 => core.memWrite16(s, addr, s.fpu_control_word),  // FNSTCW
            else => {},
        }
    }
}

pub fn opDA(s: *CpuState) void { // int32 ops / FCMOV
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        switch (d.reg) {
            0 => { if (core.getFlag(s, core.CF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVB
            1 => { if (core.getFlag(s, core.ZF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVE
            2 => { if (core.getFlag(s, core.CF_BIT) or core.getFlag(s, core.ZF_BIT)) fpuSet(s, 0, fpuGet(s, d.rm)); },  // FCMOVBE
            3 => fpuSet(s, 0, fpuGet(s, d.rm)),  // FCMOVU
            5 => if (d.rm == 1) { fpuCompare(s, fpuGet(s, 0), fpuGet(s, 1)); _ = fpuPop(s); _ = fpuPop(s); },  // FUCOMPP
            else => {},
        }
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const val: f80 = @floatFromInt(core.memReadS32(s, addr)); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opDB(s: *CpuState) void { // FILD/FISTP int32, FCLEX/FINIT, FUCOMI
    const d = core.decodeModRM(s);
    if (d.mod == 3) {
        if (d.reg == 4) {
            if (d.rm == 2) { s.fpu_status_word &= 0x7F00; }  // FCLEX
            else if (d.rm == 3) { s.fpu_control_word = 0x037F; s.fpu_status_word = 0; s.fpu_tag_word = 0xFFFF; s.fpu_top = 0; }  // FINIT
        } else if (d.reg == 5) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), false)  // FUCOMI
        else if (d.reg == 6) fpuComi(s, fpuGet(s, 0), fpuGet(s, d.rm), false);  // FCOMI
    } else {
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, @floatFromInt(core.memReadS32(s, addr))),  // FILD m32
            1 => { const i: i32 = @intFromFloat(@trunc(fpuGet(s, 0))); core.memWrite32(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTTP
            2 => { const i: i32 = @intFromFloat(@round(fpuGet(s, 0))); core.memWrite32(s, addr, @bitCast(i)); },  // FIST
            3 => { const i: i32 = @intFromFloat(@round(fpuGet(s, 0))); core.memWrite32(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTP
            5 => { // FLD m80real
                const lo = core.memRead32(s, addr); const hi = core.memRead32(s, addr + 4); const exp = core.memRead16(s, addr + 8);
                const sign: f80 = if ((exp & 0x8000) != 0) -1.0 else 1.0;
                const e: i32 = @as(i32, exp & 0x7FFF) - 16383;
                const mant: f80 = (@as(f80, @floatFromInt(@as(u64, hi))) * 4294967296.0 + @as(f80, @floatFromInt(lo))) / 9223372036854775808.0;
                if (e == -16383 and lo == 0 and hi == 0) fpuPush(s, sign * 0.0)
                else fpuPush(s, sign * @as(f80, @floatCast(std.math.pow(f64, 2.0, @as(f64, @floatFromInt(e))))) * mant);
            },
            7 => { writeDouble(s, addr, @floatCast(fpuGet(s, 0))); core.memWrite16(s, addr + 8, 0); _ = fpuPop(s); },  // FSTP m80
            else => {},
        }
    }
}

pub fn opDC(s: *CpuState) void { // float64 ops (reversed operands)
    const d = core.decodeModRM(s);
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
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const val: f80 = @floatCast(readDouble(s, addr)); const st0 = fpuGet(s, 0);
        switch (d.reg) {
            0 => fpuSet(s, 0, st0 + val), 1 => fpuSet(s, 0, st0 * val),
            2 => fpuCompare(s, st0, val), 3 => { fpuCompare(s, st0, val); _ = fpuPop(s); },
            4 => fpuSet(s, 0, st0 - val), 5 => fpuSet(s, 0, val - st0),
            6 => fpuSet(s, 0, st0 / val), 7 => fpuSet(s, 0, val / st0),
            else => {},
        }
    }
}

pub fn opDD(s: *CpuState) void { // FLD/FST/FSTP float64, FUCOM
    const d = core.decodeModRM(s);
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
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        switch (d.reg) {
            0 => fpuPush(s, @floatCast(readDouble(s, addr))),  // FLD m64
            1 => { writeDouble(s, addr, @floatCast(@trunc(fpuGet(s, 0)))); _ = fpuPop(s); },  // FISTTP m64
            2 => writeDouble(s, addr, @floatCast(fpuGet(s, 0))),  // FST m64
            3 => { writeDouble(s, addr, @floatCast(fpuGet(s, 0))); _ = fpuPop(s); },  // FSTP m64
            4, 6 => {},  // FRSTOR/FNSAVE NOP
            7 => core.memWrite16(s, addr, s.fpu_status_word),  // FNSTSW m16
            else => {},
        }
    }
}

pub fn opDE(s: *CpuState) void { // FADDP/FMULP/etc / int16
    const d = core.decodeModRM(s);
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
        const r = core.resolveRm(s, d.mod, d.rm); const addr = core.applySegOvr(s, r.addr);
        const raw = core.memRead16(s, addr); const val: f80 = @floatFromInt(@as(i16, @bitCast(raw)));
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

pub fn opDF(s: *CpuState) void { // FILD/FISTP int16/int64, FNSTSW AX, FUCOMIP
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
            1 => { const i: i16 = @intFromFloat(@trunc(fpuGet(s, 0))); core.memWrite16(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTTP m16
            2 => { const i: i16 = @intFromFloat(@round(fpuGet(s, 0))); core.memWrite16(s, addr, @bitCast(i)); },  // FIST m16
            3 => { const i: i16 = @intFromFloat(@round(fpuGet(s, 0))); core.memWrite16(s, addr, @bitCast(i)); _ = fpuPop(s); },  // FISTP m16
            // FILD m64: f80 has 64-bit mantissa — all i64 values are exact, no precision loss.
            5 => { const lo = core.memRead32(s, addr); const hi = core.memReadS32(s, addr + 4); fpuPush(s, @as(f80, @floatFromInt(@as(i64, hi) * @as(i64, 0x100000000) + @as(i64, lo)))); },  // FILD m64
            // FISTP m64: f80→i64 is exact for all representable integers.
            7 => { const val = fpuGet(s, 0); const iv: i64 = @intFromFloat(@trunc(val)); const bits: u64 = @bitCast(iv); core.memWrite32(s, addr, @truncate(bits)); core.memWrite32(s, addr + 4, @truncate(bits >> 32)); _ = fpuPop(s); },  // FISTP m64
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
