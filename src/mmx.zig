// mmx.zig — MMX instruction handlers (MOVQ, EMMS).
// Called from two_byte.zig for 0x0F-prefixed opcodes 6F, 7F, 77.
const core = @import("core.zig");

const CpuState = core.CpuState;

// MOVQ mm, m64/mm  (0x0F 6F /r)
pub fn opMovqLoad(s: *CpuState) void {
    const d = core.decodeModRM(s);
    const dst = d.reg & 7;
    if (d.mod == 3) {
        s.mmx_regs[dst] = s.mmx_regs[d.rm & 7];
    } else {
        const r = core.resolveRm(s, d.mod, d.rm);
        const addr = core.applySegOvr(s, r.addr);
        const lo: u64 = core.memRead32(s, addr);
        const hi: u64 = core.memRead32(s, addr + 4);
        s.mmx_regs[dst] = lo | (hi << 32);
    }
}

// MOVQ m64/mm, mm  (0x0F 7F /r)
pub fn opMovqStore(s: *CpuState) void {
    const d = core.decodeModRM(s);
    const src = d.reg & 7;
    if (d.mod == 3) {
        s.mmx_regs[d.rm & 7] = s.mmx_regs[src];
    } else {
        const r = core.resolveRm(s, d.mod, d.rm);
        const addr = core.applySegOvr(s, r.addr);
        const val = s.mmx_regs[src];
        core.memWrite32(s, addr, @truncate(val));
        core.memWrite32(s, addr + 4, @truncate(val >> 32));
    }
}

// MOVD mm, r/m32  (0x0F 6E /r)
// Loads 32 bits into low dword of MMX register, zero-extends to 64 bits.
pub fn opMovdLoad(s: *CpuState) void {
    const d = core.decodeModRM(s);
    const dst = d.reg & 7;
    const val: u32 = if (d.mod == 3)
        s.regs[d.rm & 7]
    else
        core.memRead32(s, core.applySegOvr(s, core.resolveRm(s, d.mod, d.rm).addr));
    s.mmx_regs[dst] = @as(u64, val);
}

// MOVD r/m32, mm  (0x0F 7E /r)
// Stores low 32 bits of MMX register to register or memory.
pub fn opMovdStore(s: *CpuState) void {
    const d = core.decodeModRM(s);
    const src = d.reg & 7;
    const val: u32 = @truncate(s.mmx_regs[src]);
    if (d.mod == 3) {
        s.regs[d.rm & 7] = val;
    } else {
        const addr = core.applySegOvr(s, core.resolveRm(s, d.mod, d.rm).addr);
        core.memWrite32(s, addr, val);
    }
}

// PUNPCKLDQ mm, mm/m64  (0x0F 62 /r)
// Unpacks low doublewords: dst[0..31] stays, dst[32..63] = src[0..31].
pub fn opPunpckldq(s: *CpuState) void {
    const d = core.decodeModRM(s);
    const dst = d.reg & 7;
    const src: u32 = if (d.mod == 3)
        @truncate(s.mmx_regs[d.rm & 7])
    else
        core.memRead32(s, core.applySegOvr(s, core.resolveRm(s, d.mod, d.rm).addr));
    const lo: u32 = @truncate(s.mmx_regs[dst]);
    s.mmx_regs[dst] = @as(u64, lo) | (@as(u64, src) << 32);
}

// EMMS  (0x0F 77) — marks all FPU tag word entries as empty (11b each).
pub fn opEmms(s: *CpuState) void { s.fpu_tag_word = 0xFFFF; }
