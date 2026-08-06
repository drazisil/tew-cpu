// SPDX-License-Identifier: LGPL-3.0-or-later
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
            const imul_op2: i64 = @as(i32, @bitCast(core.readRmFixed32(s, d.mod, d.rm)));
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
        0xB1 => { // CMPXCHG rm32, r32 -- ported from pe-walker (confirmed live
            // against real Windows XP kernel32.dll: `lock cmpxchg dword ptr
            // [edx], ecx`, a classic InterlockedCompareExchange shape; the
            // `lock` prefix needs no special handling, already tolerated as
            // an ordinary skippable prefix byte for this single-threaded
            // interpreter). Adapted to tew's current width-aware flags API
            // (updateFlagsArithW + explicit .w32, not pe-walker's
            // pre-width-split updateFlagsArith).
            const d = core.decodeModRM(s);
            const res = core.readRmFixed32Resolved(s, d.mod, d.rm);
            const acc = s.regs[EAX];
            core.updateFlagsArithW(s, @as(i64, acc) - @as(i64, res.value), acc, res.value, true, .w32);
            if (acc == res.value) {
                core.writeRmFixed32Resolved(s, res.is_reg, res.addr, s.regs[d.reg]);
            } else {
                s.regs[EAX] = res.value;
            }
        },
        0xAC => { // SHRD rm32, r32, imm8 -- ported from pe-walker, see doShrd
            const d = core.decodeModRM(s); const res = core.readRmFixed32Resolved(s, d.mod, d.rm);
            const count = core.fetch8(s) & 0x1F;
            doShrd(s, res.is_reg, res.addr, res.value, s.regs[d.reg], count);
        },
        0xAD => { // SHRD rm32, r32, CL -- ported from pe-walker, see doShrd
            const d = core.decodeModRM(s); const res = core.readRmFixed32Resolved(s, d.mod, d.rm);
            const count: u8 = @truncate(s.regs[ECX] & 0x1F);
            doShrd(s, res.is_reg, res.addr, res.value, s.regs[d.reg], count);
        },
        0xA4 => { // SHLD rm32, r32, imm8 -- ported from pe-walker, see doShld
            const d = core.decodeModRM(s); const res = core.readRmFixed32Resolved(s, d.mod, d.rm);
            const count = core.fetch8(s) & 0x1F;
            doShld(s, res.is_reg, res.addr, res.value, s.regs[d.reg], count);
        },
        0xA5 => { // SHLD rm32, r32, CL -- ported from pe-walker, see doShld
            const d = core.decodeModRM(s); const res = core.readRmFixed32Resolved(s, d.mod, d.rm);
            const count: u8 = @truncate(s.regs[ECX] & 0x1F);
            doShld(s, res.is_reg, res.addr, res.value, s.regs[d.reg], count);
        },
        0xC1 => { // XADD rmv, rv -- was hardcoded 32-bit throughout, ignoring
            // 0x66 (op_size_ovr); now width-aware on both the rm operand and
            // the register operand, matching op8B's MOV rv,rmv pattern.
            const d = core.decodeModRM(s);
            const width: core.Width = if (s.op_size_ovr) .w16 else .w32;
            const dst = core.readRmv(s, d.mod, d.rm);
            const src = if (s.op_size_ovr) s.regs[d.reg] & 0xFFFF else s.regs[d.reg];
            if (s.op_size_ovr) s.regs[d.reg] = (s.regs[d.reg] & 0xFFFF0000) | (dst & 0xFFFF)
            else s.regs[d.reg] = dst;
            core.writeRmv(s, d.mod, d.rm, dst +% src);
            core.updateFlagsArithW(s, @as(i64, dst) + @as(i64, src), dst, src, false, width);
        },
        0xBD => { // BSR r32, rm32
            const d = core.decodeModRM(s); const v = core.readRmFixed32(s, d.mod, d.rm);
            if (v == 0) core.setFlag(s, ZF_BIT, true)
            else { core.setFlag(s, ZF_BIT, false); s.regs[d.reg] = 31 - @clz(v); }
        },
        0xBC => { // BSF r32, rm32
            const d = core.decodeModRM(s); const v = core.readRmFixed32(s, d.mod, d.rm);
            if (v == 0) core.setFlag(s, ZF_BIT, true)
            else { core.setFlag(s, ZF_BIT, false); s.regs[d.reg] = @ctz(v); }
        },
        0x40...0x4F => { // CMOVcc r32, rm32
            const d = core.decodeModRM(s); const v = core.readRmFixed32(s, d.mod, d.rm);
            if (core.evalCond(s, op2 & 0xF)) s.regs[d.reg] = v;
        },
        0xC8...0xCF => { // BSWAP r32
            const r: u8 = op2 & 7; const v = s.regs[r];
            s.regs[r] = ((v & 0xFF) << 24) | (((v >> 8) & 0xFF) << 16) | (((v >> 16) & 0xFF) << 8) | (v >> 24);
        },
        0xA3 => { // BT rm32, r32
            const d = core.decodeModRM(s); const bit: u5 = @truncate(s.regs[d.reg] & 0x1F);
            core.setFlag(s, CF_BIT, ((core.readRmFixed32(s, d.mod, d.rm) >> bit) & 1) != 0);
        },
        0xBA => { // Group 8: BT/BTS/BTR/BTC rm32, imm8
            const d = core.decodeModRM(s); const bit: u5 = @truncate(core.fetch8(s) & 0x1F);
            const v = core.readRmFixed32(s, d.mod, d.rm); core.setFlag(s, CF_BIT, ((v >> bit) & 1) != 0);
            switch (d.reg) {
                5 => core.writeRmFixed32(s, d.mod, d.rm, v | (@as(u32, 1) << bit)),
                6 => core.writeRmFixed32(s, d.mod, d.rm, v & ~(@as(u32, 1) << bit)),
                7 => core.writeRmFixed32(s, d.mod, d.rm, (v ^ (@as(u32, 1) << bit))),
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

// ─── SHRD/SHLD (double-precision shift) ───────────────────────────────────────
// Ported from pe-walker (see PROVENANCE.md there): real x86 semantics, dest
// is shifted by count with bits shifted in from src on the far side rather
// than zero-filled -- the classic pattern for extracting a shifted field out
// of a 64-bit value held across two 32-bit registers. Confirmed live there
// against real ntdll.dll heap-manager code (`shrd eax, edx, 0x18`). count is
// already masked to 5 bits by both callers (0xAC/0xAD and 0xA4/0xA5 above),
// matching real hardware's behavior for a 32-bit destination.
//
// count == 0 is a real, legal no-op (write the unchanged value back, touch
// no flags) -- required before computing `32 - count` below, which would
// otherwise be an invalid (>31) shift amount for a u5.
//
// Flags: CF is the last bit shifted out (same formula SHR/SHL use in
// doGroup2). OF is only defined for a 1-bit shift (a real sign change),
// matching ROL/ROR's existing convention.
fn doShrd(s: *CpuState, is_reg: bool, addr: u32, dest: u32, src: u32, count: u8) void {
    if (count == 0) { core.writeRmFixed32Resolved(s, is_reg, addr, dest); return; }
    const c: u5 = @truncate(count);
    const result = (dest >> c) | (src << @as(u5, @truncate(32 - @as(u8, c))));
    core.writeRmFixed32Resolved(s, is_reg, addr, result);
    core.updateFlagsLogicW(s, result, .w32);
    core.setFlag(s, CF_BIT, ((dest >> @as(u5, @truncate(count - 1))) & 1) != 0);
    if (count == 1) core.setFlag(s, OF_BIT, ((result & 0x80000000) != 0) != ((dest & 0x80000000) != 0));
}

fn doShld(s: *CpuState, is_reg: bool, addr: u32, dest: u32, src: u32, count: u8) void {
    if (count == 0) { core.writeRmFixed32Resolved(s, is_reg, addr, dest); return; }
    const c: u5 = @truncate(count);
    const result = (dest << c) | (src >> @as(u5, @truncate(32 - @as(u8, c))));
    core.writeRmFixed32Resolved(s, is_reg, addr, result);
    core.updateFlagsLogicW(s, result, .w32);
    core.setFlag(s, CF_BIT, ((dest >> @as(u5, @truncate(32 - @as(u8, count)))) & 1) != 0);
    if (count == 1) core.setFlag(s, OF_BIT, ((result & 0x80000000) != 0) != ((dest & 0x80000000) != 0));
}
