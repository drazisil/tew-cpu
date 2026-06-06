// two_byte.zig — 0x0F-prefixed opcode handler (op0F).
// Imported by cpu.zig; imports mmx.zig for MMX instructions.
const std = @import("std");
const core = @import("core.zig");
const mmx = @import("mmx.zig");

const CpuState = core.CpuState;

const EAX = core.EAX; const ECX = core.ECX; const EDX = core.EDX; const EBX = core.EBX;
const ESP = core.ESP;
const CF_BIT = core.CF_BIT; const ZF_BIT = core.ZF_BIT; const OF_BIT = core.OF_BIT;

// ─── op0F: dispatcher for 0x0F-prefixed opcodes ──────────────────────────────
pub fn op0F(s: *CpuState) void {
    const op2 = core.fetch8(s);
    switch (op2) {
        0xB6 => { // MOVZX r32, rm8
            const d = core.decodeModRM(s); const r = core.resolveRm(s, d.mod, d.rm);
            s.regs[d.reg] = if (r.is_reg) s.regs[r.addr] & 0xFF else core.memRead8(s, core.applySegOvr(s, r.addr));
        },
        0xB7 => { // MOVZX r32, rm16
            const d = core.decodeModRM(s); const r = core.resolveRm(s, d.mod, d.rm);
            s.regs[d.reg] = if (r.is_reg) s.regs[r.addr] & 0xFFFF else @as(u32, core.memRead16(s, core.applySegOvr(s, r.addr)));
        },
        0xBE => { // MOVSX r32, rm8
            const d = core.decodeModRM(s); const r = core.resolveRm(s, d.mod, d.rm);
            const v: u8 = if (r.is_reg) @truncate(s.regs[r.addr]) else core.memRead8(s, core.applySegOvr(s, r.addr));
            s.regs[d.reg] = @bitCast(@as(i32, @as(i8, @bitCast(v))));
        },
        0xBF => { // MOVSX r32, rm16
            const d = core.decodeModRM(s); const r = core.resolveRm(s, d.mod, d.rm);
            const v: u16 = if (r.is_reg) @truncate(s.regs[r.addr]) else core.memRead16(s, core.applySegOvr(s, r.addr));
            s.regs[d.reg] = @bitCast(@as(i32, @as(i16, @bitCast(v))));
        },
        0xAF => { // IMUL r32, rm32
            const d = core.decodeModRM(s);
            const imul_op1: i64 = @as(i32, @bitCast(s.regs[d.reg]));
            const imul_op2: i64 = @as(i32, @bitCast(core.readRm32(s, d.mod, d.rm)));
            const r32: u32 = @truncate(@as(u64, @bitCast(imul_op1 * imul_op2)));
            s.regs[d.reg] = r32;
            const ov = (imul_op1 * imul_op2) != @as(i64, @as(i32, @bitCast(r32)));
            core.setFlag(s, CF_BIT, ov); core.setFlag(s, OF_BIT, ov);
        },
        0x90...0x9F => { // SETcc rm8
            const d = core.decodeModRM(s); const r = core.resolveRm(s, d.mod, d.rm);
            const v: u8 = if (core.evalCond(s, op2 & 0xF)) 1 else 0;
            if (r.is_reg) s.regs[r.addr] = (s.regs[r.addr] & 0xFFFFFF00) | v
            else core.memWrite8(s, core.applySegOvr(s, r.addr), v);
        },
        0x80...0x8F => { // Jcc rel32 (near)
            const rel = core.fetchS32(s);
            if (core.evalCond(s, op2 & 0xF)) s.eip = s.eip +% @as(u32, @bitCast(rel));
        },
        0xC1 => { // XADD rm32, r32
            const d = core.decodeModRM(s); const dst = core.readRm32(s, d.mod, d.rm); const src = s.regs[d.reg];
            s.regs[d.reg] = dst; core.writeRm32(s, d.mod, d.rm, dst +% src);
            core.updateFlagsArith(s, @as(i64, dst) + @as(i64, src), dst, src, false);
        },
        0xBD => { // BSR r32, rm32
            const d = core.decodeModRM(s); const v = core.readRm32(s, d.mod, d.rm);
            if (v == 0) core.setFlag(s, ZF_BIT, true)
            else { core.setFlag(s, ZF_BIT, false); s.regs[d.reg] = 31 - @clz(v); }
        },
        0xBC => { // BSF r32, rm32
            const d = core.decodeModRM(s); const v = core.readRm32(s, d.mod, d.rm);
            if (v == 0) core.setFlag(s, ZF_BIT, true)
            else { core.setFlag(s, ZF_BIT, false); s.regs[d.reg] = @ctz(v); }
        },
        0x40...0x4F => { // CMOVcc r32, rm32
            const d = core.decodeModRM(s); const v = core.readRm32(s, d.mod, d.rm);
            if (core.evalCond(s, op2 & 0xF)) s.regs[d.reg] = v;
        },
        0xC8...0xCF => { // BSWAP r32
            const r: u8 = op2 & 7; const v = s.regs[r];
            s.regs[r] = ((v & 0xFF) << 24) | (((v >> 8) & 0xFF) << 16) | (((v >> 16) & 0xFF) << 8) | (v >> 24);
        },
        0xA3 => { // BT rm32, r32
            const d = core.decodeModRM(s); const bit: u5 = @truncate(s.regs[d.reg] & 0x1F);
            core.setFlag(s, CF_BIT, ((core.readRm32(s, d.mod, d.rm) >> bit) & 1) != 0);
        },
        0xBA => { // Group 8: BT/BTS/BTR/BTC rm32, imm8
            const d = core.decodeModRM(s); const bit: u5 = @truncate(core.fetch8(s) & 0x1F);
            const v = core.readRm32(s, d.mod, d.rm); core.setFlag(s, CF_BIT, ((v >> bit) & 1) != 0);
            switch (d.reg) {
                5 => core.writeRm32(s, d.mod, d.rm, v | (@as(u32, 1) << bit)),
                6 => core.writeRm32(s, d.mod, d.rm, v & ~(@as(u32, 1) << bit)),
                7 => core.writeRm32(s, d.mod, d.rm, (v ^ (@as(u32, 1) << bit))),
                else => {},
            }
        },
        0x34 => { // SYSENTER — fast NT syscall gate
            if (s.int_handler) |h| h(s, 0x2E);
        },
        0x35 => { // SYSEXIT — fast return from kernel
            s.eip = s.regs[ECX];
            s.regs[ESP] = s.regs[EDX];
        },
        0xA2 => { // CPUID
            const leaf = s.regs[EAX];
            switch (leaf) {
                0 => { s.regs[EAX] = 1; s.regs[EBX] = 0x756E6547; s.regs[EDX] = 0x49656E69; s.regs[ECX] = 0x6C65746E; },
                // EDX bit 0=FPU, bit 15=CMOV, bit 23=MMX.
                1 => { s.regs[EAX] = 0x00000600; s.regs[EBX] = 0; s.regs[ECX] = 0; s.regs[EDX] = 0x00808001; },
                else => { s.regs[EAX] = 0; s.regs[EBX] = 0; s.regs[ECX] = 0; s.regs[EDX] = 0; },
            }
        },
        // ── MMX ──────────────────────────────────────────────────────────────
        0x6E => mmx.opMovdLoad(s),    // MOVD mm, r/m32
        0x6F => mmx.opMovqLoad(s),    // MOVQ mm, m64/mm
        0x7E => mmx.opMovdStore(s),   // MOVD r/m32, mm
        0x7F => mmx.opMovqStore(s),   // MOVQ m64/mm, mm
        0x62 => mmx.opPunpckldq(s),   // PUNPCKLDQ mm, mm/m32
        0x77 => mmx.opEmms(s),        // EMMS
        else => { s.faulted = true; s.halted = true; },
    }
}
