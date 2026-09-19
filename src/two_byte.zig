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

// ─── BT/BTS/BTR/BTC rm32, r32 (0F A3/AB/B3/BB) ───────────────────────────────
// Two things the previous per-opcode copies got wrong (found live 2026-09-18
// via a `0F AB 04 24` in the game's static CRT, a strspn-style 256-bit
// char-set bitmap built on the stack with `bts [esp], eax`):
//  1. The operand must be resolved exactly ONCE. readRmFixed32 + writeRmFixed32
//     each call resolveRm, so a memory operand's SIB/displacement bytes were
//     fetched twice -- consuming the next instruction's opcode byte as a
//     phantom SIB and faulting one byte into the following instruction.
//  2. With a MEMORY operand the register bit offset is NOT taken mod 32: it
//     addresses the dword at base + 4*(offset >> 5) (arithmetic shift, so a
//     negative offset reaches below base) and the bit is offset & 31. Only the
//     register-destination form (and the imm8 Group 8 form) wraps mod 32.
const BitOp = enum { bt, bts, btr, btc };

fn bitTestReg(s: *CpuState, comptime op: BitOp) void {
    const d = core.decodeModRM(s);
    const offset: u32 = s.regs[d.reg];
    const bit: u5 = @truncate(offset & 0x1F);
    const mask: u32 = @as(u32, 1) << bit;
    const r = core.resolveRm(s, d.mod, d.rm);
    if (r.is_reg) {
        const v = s.regs[r.addr];
        core.setFlag(s, CF_BIT, (v & mask) != 0);
        switch (op) {
            .bt => {},
            .bts => s.regs[r.addr] = v | mask,
            .btr => s.regs[r.addr] = v & ~mask,
            .btc => s.regs[r.addr] = v ^ mask,
        }
        return;
    }
    const signed_offset: i32 = @bitCast(offset);
    const dword_delta: u32 = @bitCast((signed_offset >> 5) *% 4);
    const addr = core.applySegOvr(s, r.addr) +% dword_delta;
    const v = core.memRead32(s, addr);
    core.setFlag(s, CF_BIT, (v & mask) != 0);
    switch (op) {
        .bt => {},
        .bts => core.memWrite32(s, addr, v | mask),
        .btr => core.memWrite32(s, addr, v & ~mask),
        .btc => core.memWrite32(s, addr, v ^ mask),
    }
}

// Group 8: 0F BA /4../7 ib. Same single-resolve rule as bitTestReg, plus the
// encoding order matters: for a memory operand the SIB/displacement bytes
// come BEFORE the imm8, so the operand must be resolved first and the imm8
// fetched last (the previous code fetched the imm8 first and would have read
// the SIB byte as the bit offset). The imm8 form always wraps mod 32 within
// the addressed dword -- unlike the register-offset form there is no
// dword displacement. /0../3 are undefined encodings: fault loudly rather
// than silently acting like BT.
fn bitTestImm(s: *CpuState, op2: u8) void {
    const d = core.decodeModRM(s);
    const kind: BitOp = switch (d.reg) {
        4 => .bt,
        5 => .bts,
        6 => .btr,
        7 => .btc,
        else => {
            s.last_opcode = op2;
            core.opFault(s);
            return;
        },
    };
    const r = core.resolveRm(s, d.mod, d.rm);
    const bit: u5 = @truncate(core.fetch8(s) & 0x1F);
    const mask: u32 = @as(u32, 1) << bit;
    if (r.is_reg) {
        const v = s.regs[r.addr];
        core.setFlag(s, CF_BIT, (v & mask) != 0);
        switch (kind) {
            .bt => {},
            .bts => s.regs[r.addr] = v | mask,
            .btr => s.regs[r.addr] = v & ~mask,
            .btc => s.regs[r.addr] = v ^ mask,
        }
        return;
    }
    const addr = core.applySegOvr(s, r.addr);
    const v = core.memRead32(s, addr);
    core.setFlag(s, CF_BIT, (v & mask) != 0);
    switch (kind) {
        .bt => {},
        .bts => core.memWrite32(s, addr, v | mask),
        .btr => core.memWrite32(s, addr, v & ~mask),
        .btc => core.memWrite32(s, addr, v ^ mask),
    }
}

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
        0xA3 => bitTestReg(s, .bt),  // BT  rm32, r32
        0xAB => bitTestReg(s, .bts), // BTS rm32, r32
        0xB3 => bitTestReg(s, .btr), // BTR rm32, r32
        0xBB => bitTestReg(s, .btc), // BTC rm32, r32
        0xBA => bitTestImm(s, op2), // Group 8: BT/BTS/BTR/BTC rm32, imm8
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
        // TEMPORARY debug counters (2026-09-03) -- see core.zig's
        // mmx_call_count/mmx_last_eip comment. Remove once resolved.
        0x6E => { s.mmx_call_count += 1; s.mmx_last_eip = s.eip; mmx.opMovdLoad(s); },    // MOVD mm, r/m32
        0x6F => { s.mmx_call_count += 1; s.mmx_last_eip = s.eip; mmx.opMovqLoad(s); },    // MOVQ mm, m64/mm
        0x7E => { s.mmx_call_count += 1; s.mmx_last_eip = s.eip; mmx.opMovdStore(s); },   // MOVD r/m32, mm
        0x7F => { s.mmx_call_count += 1; s.mmx_last_eip = s.eip; mmx.opMovqStore(s); },   // MOVQ m64/mm, mm
        0x62 => { s.mmx_call_count += 1; s.mmx_last_eip = s.eip; mmx.opPunpckldq(s); },   // PUNPCKLDQ mm, mm/m32
        0x77 => { s.mmx_call_count += 1; s.mmx_last_eip = s.eip; mmx.opEmms(s); },        // EMMS
        else => {
            // Reuse the shared opFault path (unknown_opcode diagnostic,
            // 2026-09-18) so a missing two-byte (0x0F xx) opcode reports
            // cleanly too, not just a missing single-byte one. last_opcode
            // is overwritten with the real second byte here -- cpuStep
            // already set it to 0x0F, which alone wouldn't say which
            // 0x0F-prefixed instruction was actually missing (found live:
            // BTS/BTR/BTC rm32,r32 -- 0x0F 0xAB/0xB3/0xBB -- were all
            // missing here despite BT and the Group 8 imm8 form both
            // being wired).
            s.last_opcode = op2;
            core.opFault(s);
        },
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
