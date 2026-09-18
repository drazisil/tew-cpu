// SPDX-License-Identifier: LGPL-3.0-or-later
// kernel.zig — the entire Python-facing C ABI for libcpu.so, and nothing
// else. This is the build root (see build.zig): every export fn in the
// project is a top-level declaration directly in this file, on purpose --
// Zig's lazy per-declaration analysis only guarantees a root file's own
// top-level exports are emitted into the compiled .so regardless of whether
// anything calls them; a foreign file's export fn that's merely @import-ed
// but never called gets silently dropped (bit us once already with
// memory.zig/alloc.zig, worked around at the time with a comptime
// force-reference block -- now unnecessary and removed, since there's no
// other file left with an export fn nothing calls). Do not add a new file
// with its own export fn and just @import it here without calling
// something in it -- move the export itself into this file instead.
//
// The internal execution engine (dispatch table, cpuStep, opcode handlers)
// lives in engine.zig and is never exported; CpuState/register/flag
// definitions live in core.zig; the shared bounds-check/byte-access
// arithmetic that memory access is built on lives in primitives.zig -- see
// its own header comment for why the single-byte and combined-width bounds
// checks are kept as two distinct formulas rather than unified into one.
const std = @import("std");
const core = @import("core.zig");
const engine = @import("engine.zig");
const primitives = @import("primitives.zig");
const scheduler = @import("scheduler.zig");

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
// IntHandlerFn: ported from pe-walker (PROVENANCE.md's "IntHandlerFn alias
// visibility fix") -- this alias already existed and worked internally, but
// wasn't `pub`, so an external Zig consumer importing this file as a module
// couldn't reference the type to install its own interrupt handler (every
// software interrupt unconditionally faulted there as a result). Same
// reasoning as CpuState/RunResult/WriteHookFn/StepHookFn above.
pub const IntHandlerFn = core.IntHandlerFn;
const EAX = core.EAX; const ECX = core.ECX; const EDX = core.EDX; const EBX = core.EBX;
const ESP = core.ESP; const EBP = core.EBP; const ESI = core.ESI; const EDI = core.EDI;
const CF_BIT = core.CF_BIT; const PF_BIT = core.PF_BIT; const ZF_BIT = core.ZF_BIT;
const SF_BIT = core.SF_BIT; const DF_BIT = core.DF_BIT; const OF_BIT = core.OF_BIT;
const SEG_NONE = core.SEG_NONE; const SEG_FS = core.SEG_FS; const SEG_GS = core.SEG_GS;
const REP_NONE = core.REP_NONE; const REP_REP = core.REP_REP; const REP_REPNE = core.REP_REPNE;
const OpFn = core.OpFn;
const RmInfo = core.RmInfo;
const Rm8Result = core.Rm8Result;
const Rm32Result = core.Rm32Result;
const ModRm = core.ModRm;


const testing = std.testing;

// ─── C API ────────────────────────────────────────────────────────────────────
export fn cpu_create(memory: [*]u8, memory_size: usize) ?*CpuState {
    const s = std.heap.c_allocator.create(CpuState) catch return null;
    s.* = CpuState{ .memory = memory, .memory_size = memory_size };
    // run_id: the core's own native notion of "a new session" is the birth
    // of a fresh CpuState -- see history/capture.zig. Deliberately NOT
    // std.crypto.random here: confirmed via Valgrind that its thread-local
    // CSPRNG state (crypto.tlcsprng) segfaults (NULL-pointer write inside
    // fillWithCsprng) on the second call when this library is loaded into
    // a foreign host process and invoked repeatedly via ctypes/libffi
    // (Python, in this codebase's case) -- it doesn't reliably survive
    // being called from a thread Zig's own runtime didn't start. run_id
    // only needs to be unique per session, not unpredictable, so mix a
    // timestamp with this CpuState's own fresh heap address instead.
    s.run_id = @as(u64, @bitCast(std.time.milliTimestamp())) ^ @intFromPtr(s);
    return s;
}
export fn cpu_destroy(s: *CpuState) void { std.heap.c_allocator.destroy(s); }
export fn cpu_set_int_handler(s: *CpuState, handler: IntHandlerFn) void { s.int_handler = handler; }
pub export fn cpu_run(s: *CpuState, max_steps: u64) RunResult {
    var i: u64 = 0;
    while (!s.halted and !s.fatal_halted and i < max_steps) : (i += 1) {
        const eip = s.eip;
        // Logpoints: fire inline C callback, no halt.
        for (0..8) |j| {
            if (s.lp_eip[j] != 0 and eip == s.lp_eip[j]) {
                if (s.lp_cb[j]) |cb| cb(eip, &s.regs, s.memory, s.memory_size);
            }
        }
        // Breakpoints: halt before executing the instruction.
        for (s.bp_table) |bp| {
            if (bp != 0 and eip == bp) {
                s.bp_hit = true;
                s.bp_hit_eip = eip;
                s.halted = true;
                break;
            }
        }
        if (s.halted) break;
        // step_no captured pre-increment, before cpuStep runs -- matches
        // what memWrite8 sees via s.step_count during this same
        // instruction's dispatch (cpuStep only increments step_count
        // after dispatch completes). Fires once per successfully-
        // dispatched instruction using post-execution state, regardless
        // of max_steps -- correct even if a caller runs cpu_run with a
        // count greater than 1.
        const step_no = s.step_count;
        engine.cpuStep(s);
        if (!s.faulted) {
            if (s.step_hook) |hook| hook(s.history_ctx, s.run_id, step_no, s.eip, &s.regs, s.eflags);
        }
    }
    if (s.faulted) return .faulted;
    if (s.halted) return .halted;
    return .step_limit;
}
export fn cpu_get_reg(s: *CpuState, idx: u32) u32 { return if (idx < 8) s.regs[idx] else 0; }
export fn cpu_set_reg(s: *CpuState, idx: u32, val: u32) void {
    if (s.fatal_halted) return;
    if (idx < 8) s.regs[idx] = val;
}
export fn cpu_get_eip(s: *CpuState) u32 { return s.eip; }
export fn cpu_set_eip(s: *CpuState, val: u32) void {
    if (s.fatal_halted) return;
    s.eip = val;
}
export fn cpu_get_eflags(s: *CpuState) u32 { return s.eflags; }
export fn cpu_set_eflags(s: *CpuState, val: u32) void {
    if (s.fatal_halted) return;
    s.eflags = val;
}
export fn cpu_is_halted(s: *CpuState) bool { return s.halted; }
export fn cpu_is_faulted(s: *CpuState) bool { return s.faulted; }
export fn cpu_set_halted(s: *CpuState) void { s.halted = true; }
export fn cpu_clear_halted(s: *CpuState) void {
    if (s.fatal_halted) return;
    s.halted = false;
    s.faulted = false;
    s.unknown_opcode = false;
}
// Fatal halt: the emulator (not real x86) hit something it cannot simulate.
// Permanent -- no clear function exists on purpose. Reuses the real halt
// mechanism as its terminal action (cpu_run's loop already stops on
// s.halted), and additionally sets s.fatal_halted so cpu_clear_halted and
// every register/eflags/FPU setter above refuse to touch state afterward.
export fn cpu_set_fatal_halt(s: *CpuState) void {
    s.fatal_halted = true;
    s.halted = true;
}
export fn cpu_is_fatal_halted(s: *CpuState) bool { return s.fatal_halted; }

// Opt-in null-page guard (see CpuState.guard_null_page's comment in
// core.zig): real emulator startup calls this once; unit tests constructing
// a bare CpuState never do, so their tiny address-0-based buffers are
// unaffected by default.
export fn cpu_enable_null_page_guard(s: *CpuState) void {
    s.guard_null_page = true;
}
export fn cpu_get_step_count(s: *CpuState) u64 { return s.step_count; }
export fn cpu_get_run_id(s: *CpuState) u64 { return s.run_id; }
export fn cpu_get_last_opcode(s: *CpuState) u8 { return s.last_opcode; }
export fn cpu_is_unknown_opcode(s: *CpuState) bool { return s.unknown_opcode; }
export fn cpu_get_last_instr_eip(s: *CpuState) u32 { return s.last_instr_eip; }
export fn cpu_set_fs_base(s: *CpuState, val: u32) void { s.fs_base = val; }
export fn cpu_set_gs_base(s: *CpuState, val: u32) void { s.gs_base = val; }
export fn cpu_get_fs_base(s: *CpuState) u32 { return s.fs_base; }
export fn cpu_get_gs_base(s: *CpuState) u32 { return s.gs_base; }
// fpu_stack is f80 — narrow to f64 for the C API.
export fn cpu_fpu_get(s: *CpuState, i: u32) f64 { return if (i < 8) @floatCast(s.fpu_stack[i]) else 0.0; }
export fn cpu_fpu_set(s: *CpuState, i: u32, val: f64) void {
    if (s.fatal_halted) return;
    if (i < 8) s.fpu_stack[i] = @floatCast(val);
}
export fn cpu_fpu_get_top(s: *CpuState) u32 { return s.fpu_top; }
// TEMPORARY debug getters (2026-09-03) -- see core.zig's mmx_call_count/
// mmx_last_eip comment. Remove once resolved.
export fn cpu_get_mmx_call_count(s: *CpuState) u32 { return s.mmx_call_count; }
export fn cpu_get_mmx_last_eip(s: *CpuState) u32 { return s.mmx_last_eip; }
// TEMPORARY debug getters (2026-09-03) -- see core.zig's host_fpu_cw/
// host_fpu_sw/host_fpu_st0 comment. Remove once resolved.
export fn cpu_get_host_fpu_cw(s: *CpuState) u16 { return s.host_fpu_cw; }
export fn cpu_get_host_fpu_sw(s: *CpuState) u16 { return s.host_fpu_sw; }
export fn cpu_get_host_fpu_st0(s: *CpuState) f64 { return @floatCast(s.host_fpu_st0); }
export fn cpu_fpu_set_top(s: *CpuState, val: u32) void {
    if (s.fatal_halted) return;
    s.fpu_top = val & 7;
}
export fn cpu_fpu_get_status(s: *CpuState) u16 { return s.fpu_status_word; }
export fn cpu_fpu_set_status(s: *CpuState, val: u16) void {
    if (s.fatal_halted) return;
    s.fpu_status_word = val;
}
export fn cpu_fpu_get_control(s: *CpuState) u16 { return s.fpu_control_word; }
export fn cpu_fpu_set_control(s: *CpuState, val: u16) void {
    if (s.fatal_halted) return;
    s.fpu_control_word = val;
}
export fn cpu_fpu_get_tag(s: *CpuState) u16 { return s.fpu_tag_word; }
export fn cpu_fpu_set_tag(s: *CpuState, val: u16) void {
    if (s.fatal_halted) return;
    s.fpu_tag_word = val;
}
export fn cpu_set_watchpoint(s: *CpuState, addr: u32) void { s.watchpoint = addr; s.watchpoint_hit = false; }
export fn cpu_clear_watchpoint(s: *CpuState) void { s.watchpoint = 0; s.watchpoint_hit = false; }
export fn cpu_watchpoint_hit(s: *CpuState) bool { return s.watchpoint_hit; }
export fn cpu_watchpoint_eip(s: *CpuState) u32 { return s.watchpoint_eip; }
export fn cpu_watchpoint_val(s: *CpuState) u32 { return s.watchpoint_val; }

// ─── Breakpoints (halt-type) ─────────────────────────────────────────────────
export fn cpu_add_breakpoint(s: *CpuState, eip: u32) void {
    for (&s.bp_table) |*slot| { if (slot.* == 0) { slot.* = eip; return; } }
}
export fn cpu_remove_breakpoint(s: *CpuState, eip: u32) void {
    for (&s.bp_table) |*slot| { if (slot.* == eip) slot.* = 0; }
}
export fn cpu_clear_breakpoints(s: *CpuState) void {
    s.bp_table = .{0} ** 8; s.bp_hit = false; s.bp_hit_eip = 0;
}
export fn cpu_breakpoint_hit(s: *CpuState) bool { return s.bp_hit; }
export fn cpu_breakpoint_hit_eip(s: *CpuState) u32 { return s.bp_hit_eip; }
export fn cpu_clear_breakpoint_hit(s: *CpuState) void {
    s.bp_hit = false; s.halted = false; s.faulted = false;
}

// ─── Logpoints (inline callback, no halt) ───────────────────────────────────
export fn cpu_add_logpoint(s: *CpuState, eip: u32, cb: core.LogpointFn) void {
    for (0..8) |j| {
        if (s.lp_eip[j] == 0) { s.lp_eip[j] = eip; s.lp_cb[j] = cb; return; }
    }
}
export fn cpu_remove_logpoint(s: *CpuState, eip: u32) void {
    for (0..8) |j| { if (s.lp_eip[j] == eip) { s.lp_eip[j] = 0; s.lp_cb[j] = null; } }
}
export fn cpu_clear_logpoints(s: *CpuState) void {
    s.lp_eip = .{0} ** 8; s.lp_cb = .{null} ** 8;
}

// ─── Scheduler C ABI (see scheduler.zig) ────────────────────────────────────
// Stage 1 of the scheduler-to-Zig port (~/.claude/plans/vast-drifting-pike.md):
// context switch + reentrancy guard only. Blocking/wake/tick (Stage 2) and
// the TLS bitset alloc/free/allocated exports + direct-field accessors
// (Stage 3) land in later stages.
pub const SchedulerState = scheduler.SchedulerState;

export fn scheduler_create() ?*SchedulerState {
    const s = std.heap.c_allocator.create(SchedulerState) catch return null;
    s.* = SchedulerState{};
    return s;
}
export fn scheduler_destroy(s: *SchedulerState) void {
    std.heap.c_allocator.destroy(s);
}
export fn scheduler_create_main_thread(s: *SchedulerState, thread_id: u32, handle: u32) void {
    scheduler.createMainThread(s, thread_id, handle);
}
export fn scheduler_create_thread(s: *SchedulerState, thread_id: u32, handle: u32, start_address: u32, parameter: u32, suspended: bool) u32 {
    return scheduler.createThread(s, thread_id, handle, start_address, parameter, suspended);
}
export fn scheduler_switch_to(s: *SchedulerState, cpu: *CpuState, idx: u32) bool {
    return scheduler.switchTo(s, cpu, idx);
}
export fn scheduler_preempt_slice(s: *SchedulerState, cpu: *CpuState) bool {
    return scheduler.preemptSlice(s, cpu);
}
export fn scheduler_pick_next_ready(s: *SchedulerState, cpu: *CpuState) i32 {
    if (scheduler.pickNextReady(s, cpu)) |idx| return @intCast(idx);
    return -1;
}
export fn scheduler_enter_reentrant_call(s: *SchedulerState) void {
    scheduler.enterReentrantCall(s);
}
export fn scheduler_exit_reentrant_call(s: *SchedulerState) void {
    scheduler.exitReentrantCall(s);
}
export fn scheduler_reentrant_depth(s: *SchedulerState) u32 {
    return s.reentrant_depth;
}
export fn scheduler_current_idx(s: *SchedulerState) i32 {
    return s.current_idx;
}
// thread_count / virtual_ticks_ms get+set: gap-filled during Stage 4 Python
// wiring -- CRTState.virtual_ticks_ms (_state.py) is a read/write property
// forwarding straight to the scheduler, and kernel32_io.py's CreateThread
// handler logs `len(state.scheduler.threads)` for its assigned-idx message.
// Trivial field access, same inline-in-kernel.zig style as
// scheduler_current_idx/scheduler_reentrant_depth above -- no scheduler.zig
// indirection needed for a plain field read/write.
export fn scheduler_thread_count(s: *SchedulerState) u32 {
    return s.thread_count;
}
export fn scheduler_get_virtual_ticks_ms(s: *SchedulerState) u32 {
    return s.virtual_ticks_ms;
}
export fn scheduler_set_virtual_ticks_ms(s: *SchedulerState, val: u32) void {
    s.virtual_ticks_ms = val;
}

// ─── Scheduler C ABI, Stage 2: blocking/wake/tick ───────────────────────────
// See scheduler.zig's "Public: blocking operations" section header for the
// two-call kernel-tick protocol these `complete_*` wrappers are one half of
// -- callers MUST resolve `next_idx` via scheduler_pick_next_ready (with a
// Kernel.tick() retry on -1) themselves; these do not scan.
export fn scheduler_complete_block_on_cs(s: *SchedulerState, cpu: *CpuState, cs_ptr: u32, retry_eip: u32, next_idx: i32) bool {
    return scheduler.completeBlockOnCs(s, cpu, cs_ptr, retry_eip, next_idx);
}
export fn scheduler_complete_block_on_handles(s: *SchedulerState, cpu: *CpuState, handles: [*]const u32, handles_count: u32, retry_eip: u32, has_deadline: bool, deadline_ms: u32, next_idx: i32) bool {
    return scheduler.completeBlockOnHandles(s, cpu, handles[0..handles_count], retry_eip, has_deadline, deadline_ms, next_idx);
}
export fn scheduler_complete_sleep_current(s: *SchedulerState, cpu: *CpuState, return_eip: u32, eax_val: u32, sleep_ms: u32, next_idx: i32) bool {
    return scheduler.completeSleepCurrent(s, cpu, return_eip, eax_val, sleep_ms, next_idx);
}
export fn scheduler_complete_mark_current_dead(s: *SchedulerState, cpu: *CpuState, next_idx: i32) bool {
    return scheduler.completeMarkCurrentDead(s, cpu, next_idx);
}
export fn scheduler_terminate_thread(s: *SchedulerState, cpu: *CpuState, handle: u32, next_idx: i32) i8 {
    return scheduler.terminateThread(s, cpu, handle, next_idx);
}
export fn scheduler_unblock_cs(s: *SchedulerState, cs_ptr: u32) void {
    scheduler.unblockCs(s, cs_ptr);
}
export fn scheduler_unblock_handle(s: *SchedulerState, handle: u32) u32 {
    return scheduler.unblockHandle(s, handle);
}
export fn scheduler_tick(s: *SchedulerState, ms: u32) void {
    scheduler.tick(s, ms);
}

// ─── Scheduler C ABI, Stage 3: TLS bitset + handle-keyed accessors ─────────
export fn scheduler_tls_alloc_slot(s: *SchedulerState, slot: u8) void {
    scheduler.tlsAllocSlot(s, @truncate(slot));
}
export fn scheduler_tls_free_slot(s: *SchedulerState, slot: u8) void {
    scheduler.tlsFreeSlot(s, @truncate(slot));
}
export fn scheduler_tls_slot_allocated(s: *SchedulerState, slot: u8) bool {
    return scheduler.tlsSlotAllocated(s, @truncate(slot));
}
export fn scheduler_get_suspended(s: *SchedulerState, handle: u32) bool {
    return scheduler.getSuspended(s, handle);
}
export fn scheduler_set_suspended(s: *SchedulerState, handle: u32, val: bool) void {
    scheduler.setSuspended(s, handle, val);
}
export fn scheduler_get_completed(s: *SchedulerState, handle: u32) bool {
    return scheduler.getCompleted(s, handle);
}
export fn scheduler_get_wait_timed_out(s: *SchedulerState, handle: u32) bool {
    return scheduler.getWaitTimedOut(s, handle);
}
export fn scheduler_set_wait_timed_out(s: *SchedulerState, handle: u32, val: bool) void {
    scheduler.setWaitTimedOut(s, handle, val);
}
export fn scheduler_get_status(s: *SchedulerState, handle: u32) u8 {
    return scheduler.getStatus(s, handle);
}
export fn scheduler_get_thread_id(s: *SchedulerState, handle: u32) i64 {
    return scheduler.getThreadId(s, handle);
}
export fn scheduler_handle_at_idx(s: *SchedulerState, idx: u32) i64 {
    return scheduler.handleAtIdx(s, idx);
}
export fn scheduler_current_handle(s: *SchedulerState) u32 {
    return scheduler.currentHandle(s);
}

// ─── Execution-history capture (see history/capture.zig) ───────────────────
const history_capture = @import("history/capture.zig");
pub const Capture = history_capture.Capture;

/// Wires a discard-sink Capture onto `s` -- exercises the full capture path
/// (buffering, batching, event shape) without sending anything anywhere.
/// Useful for wiring/smoke tests before a real ClickHouse endpoint exists.
/// Returns an opaque handle; caller must pass it to cpu_history_disable to
/// flush + free it.
export fn cpu_history_enable_discard(s: *CpuState) ?*Capture {
    const cap = std.heap.c_allocator.create(Capture) catch return null;
    cap.* = Capture.init(std.heap.c_allocator, .none);
    cap.armFor(s);
    s.history_ctx = cap;
    s.write_hook = history_capture.onCpuWrite;
    s.step_hook = history_capture.onCpuStep;
    return cap;
}

/// Wires a real ClickHouse-flushing Capture onto `s`. base_url/user/password
/// are copied (duped with the C allocator) since ctypes-supplied strings
/// aren't guaranteed to outlive this call.
export fn cpu_history_enable_clickhouse(
    s: *CpuState,
    base_url: [*:0]const u8,
    user: [*:0]const u8,
    password: [*:0]const u8,
) ?*Capture {
    const url_copy = std.heap.c_allocator.dupeZ(u8, std.mem.span(base_url)) catch return null;
    const user_copy = std.heap.c_allocator.dupeZ(u8, std.mem.span(user)) catch return null;
    const pass_copy = std.heap.c_allocator.dupeZ(u8, std.mem.span(password)) catch return null;
    const cap = std.heap.c_allocator.create(Capture) catch return null;
    cap.* = Capture.init(std.heap.c_allocator, .{ .clickhouse = .{
        .allocator = std.heap.c_allocator,
        .base_url = url_copy,
        .user = user_copy,
        .password = pass_copy,
    } });
    cap.armFor(s);
    s.history_ctx = cap;
    s.write_hook = history_capture.onCpuWrite;
    s.step_hook = history_capture.onCpuStep;
    return cap;
}

/// Forces a flush now, without waiting for flush_threshold or disable --
/// e.g. from Python at the end of a run/on halt, so the tail of a capture
/// isn't lost.
export fn cpu_history_flush(cap: *Capture) void {
    cap.flush();
}

/// Unwires the hooks from `s`, flushes any remaining buffered events, and
/// frees the Capture. `s` itself is untouched otherwise and remains usable.
export fn cpu_history_disable(s: *CpuState, cap: *Capture) void {
    s.history_ctx = null;
    s.write_hook = null;
    s.step_hook = null;
    cap.deinit();
    std.heap.c_allocator.destroy(cap);
}


// ─── Memory access C ABI (formerly memory.zig) ──────────────────────────────
pub export fn mem_read8(ptr: [*]const u8, size: usize, addr: u32, out: *u8) bool {
    if (!primitives.inBounds1(size, addr)) return false;
    out.* = primitives.readByte(ptr, addr);
    return true;
}
pub export fn mem_read_signed8(ptr: [*]const u8, size: usize, addr: u32, out: *i8) bool {
    var v: u8 = undefined;
    if (!mem_read8(ptr, size, addr, &v)) return false;
    out.* = @bitCast(v);
    return true;
}
pub export fn mem_write8(ptr: [*]u8, size: usize, addr: u32, val: u8) bool {
    if (!primitives.inBounds1(size, addr)) return false;
    primitives.writeByte(ptr, addr, val);
    return true;
}

pub export fn mem_read16(ptr: [*]const u8, size: usize, addr: u32, out: *u16) bool {
    if (!primitives.inBoundsWidth(size, addr, 2)) return false;
    out.* = @as(u16, ptr[addr]) | (@as(u16, ptr[addr + 1]) << 8);
    return true;
}
pub export fn mem_write16(ptr: [*]u8, size: usize, addr: u32, val: u16) bool {
    if (!primitives.inBoundsWidth(size, addr, 2)) return false;
    ptr[addr] = @truncate(val);
    ptr[addr + 1] = @truncate(val >> 8);
    return true;
}

pub export fn mem_read32(ptr: [*]const u8, size: usize, addr: u32, out: *u32) bool {
    if (!primitives.inBoundsWidth(size, addr, 4)) return false;
    out.* = @as(u32, ptr[addr]) | (@as(u32, ptr[addr + 1]) << 8) |
        (@as(u32, ptr[addr + 2]) << 16) | (@as(u32, ptr[addr + 3]) << 24);
    return true;
}
pub export fn mem_read_signed32(ptr: [*]const u8, size: usize, addr: u32, out: *i32) bool {
    var v: u32 = undefined;
    if (!mem_read32(ptr, size, addr, &v)) return false;
    out.* = @bitCast(v);
    return true;
}
pub export fn mem_write32(ptr: [*]u8, size: usize, addr: u32, val: u32) bool {
    if (!primitives.inBoundsWidth(size, addr, 4)) return false;
    ptr[addr] = @truncate(val);
    ptr[addr + 1] = @truncate(val >> 8);
    ptr[addr + 2] = @truncate(val >> 16);
    ptr[addr + 3] = @truncate(val >> 24);
    return true;
}

pub export fn mem_load(ptr: [*]u8, size: usize, addr: u32, data: [*]const u8, data_len: usize) bool {
    if (!primitives.inBoundsWidth(size, addr, @intCast(data_len))) return false;
    @memcpy(ptr[addr .. addr + data_len], data[0..data_len]);
    return true;
}

pub export fn mem_is_valid_address(size: usize, addr: u32) bool {
    return primitives.inBounds1(size, addr);
}
pub export fn mem_is_valid_range(size: usize, addr: u32, range_size: usize) bool {
    return primitives.inBoundsWidth(size, addr, @intCast(range_size));
}


test "mem_read8/mem_write8 round-trip" {
    var buf = [_]u8{0} ** 16;
    try testing.expect(mem_write8(&buf, buf.len, 0, 0xFF));
    var out: u8 = undefined;
    try testing.expect(mem_read8(&buf, buf.len, 0, &out));
    try testing.expectEqual(@as(u8, 0xFF), out);
}

test "mem_read_signed8 sign-extends" {
    var buf = [_]u8{0x80} ** 1;
    var out: i8 = undefined;
    try testing.expect(mem_read_signed8(&buf, buf.len, 0, &out));
    try testing.expectEqual(@as(i8, -128), out);
}

test "mem_read16/mem_write16 little-endian round-trip" {
    var buf = [_]u8{0} ** 4;
    try testing.expect(mem_write16(&buf, buf.len, 0, 0x1234));
    try testing.expectEqual(@as(u8, 0x34), buf[0]);
    try testing.expectEqual(@as(u8, 0x12), buf[1]);
    var out: u16 = undefined;
    try testing.expect(mem_read16(&buf, buf.len, 0, &out));
    try testing.expectEqual(@as(u16, 0x1234), out);
}

test "mem_read32/mem_write32 little-endian round-trip" {
    var buf = [_]u8{0} ** 8;
    try testing.expect(mem_write32(&buf, buf.len, 0, 0x12345678));
    try testing.expectEqual(@as(u8, 0x78), buf[0]);
    try testing.expectEqual(@as(u8, 0x56), buf[1]);
    try testing.expectEqual(@as(u8, 0x34), buf[2]);
    try testing.expectEqual(@as(u8, 0x12), buf[3]);
    var out: u32 = undefined;
    try testing.expect(mem_read32(&buf, buf.len, 0, &out));
    try testing.expectEqual(@as(u32, 0x12345678), out);
}

test "mem_read_signed32 sign-extends" {
    var buf = [_]u8{0} ** 4;
    try testing.expect(mem_write32(&buf, buf.len, 0, 0xFFFFFFFF));
    var out: i32 = undefined;
    try testing.expect(mem_read_signed32(&buf, buf.len, 0, &out));
    try testing.expectEqual(@as(i32, -1), out);
}

test "mem_load bulk copy" {
    var buf = [_]u8{0} ** 16;
    const data = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    try testing.expect(mem_load(&buf, buf.len, 4, &data, data.len));
    try testing.expectEqual(@as(u8, 0xDE), buf[4]);
    try testing.expectEqual(@as(u8, 0xAD), buf[5]);
    try testing.expectEqual(@as(u8, 0xBE), buf[6]);
    try testing.expectEqual(@as(u8, 0xEF), buf[7]);
}

test "out-of-bounds reads/writes are rejected, not undefined behavior" {
    var buf = [_]u8{0} ** 4;
    var out8: u8 = undefined;
    var out16: u16 = undefined;
    var out32: u32 = undefined;

    try testing.expect(!mem_read8(&buf, buf.len, 4, &out8));
    try testing.expect(!mem_write8(&buf, buf.len, 4, 0xFF));
    try testing.expect(!mem_read16(&buf, buf.len, 3, &out16));
    try testing.expect(!mem_write16(&buf, buf.len, 3, 0x1234));
    try testing.expect(!mem_read32(&buf, buf.len, 1, &out32));
    try testing.expect(!mem_write32(&buf, buf.len, 1, 0x12345678));

    const data = [_]u8{ 1, 2 };
    try testing.expect(!mem_load(&buf, buf.len, 3, &data, data.len));
}

test "mem_is_valid_address/mem_is_valid_range" {
    try testing.expect(mem_is_valid_address(4, 0));
    try testing.expect(mem_is_valid_address(4, 3));
    try testing.expect(!mem_is_valid_address(4, 4));

    try testing.expect(mem_is_valid_range(4, 0, 4));
    try testing.expect(!mem_is_valid_range(4, 1, 4));
    try testing.expect(!mem_is_valid_range(4, 0, 5));
}

// ─── Guest heap bump-allocator C ABI (formerly alloc.zig) ───────────────────

pub export fn bump_alloc_next(current: u32, size: u32) u32 {
    const next: u64 = @as(u64, current) + @as(u64, size) + 15;
    return @intCast(next & ~@as(u64, 15));
}


test "bump_alloc_next advances by size, 16-byte aligned" {
    try testing.expectEqual(@as(u32, 16), bump_alloc_next(0, 1));
    try testing.expectEqual(@as(u32, 16), bump_alloc_next(0, 16));
    try testing.expectEqual(@as(u32, 32), bump_alloc_next(0, 17));
}

test "bump_alloc_next matches Python (current + size + 15) & ~15" {
    try testing.expectEqual(@as(u32, 0x1000 + 16), bump_alloc_next(0x1000, 1));
    try testing.expectEqual(@as(u32, 0x1000 + 64), bump_alloc_next(0x1000, 50));
}

test "bump_alloc_next of zero size still 16-byte aligns forward from a non-aligned cursor" {
    try testing.expectEqual(@as(u32, 0x10), bump_alloc_next(0x5, 0));
}

// ─── Public C ABI tests ───────────────────────────────────────────────────────
// The tests above all call the internal cpuStep() directly -- real proof the
// core logic needs no Python/ctypes, but not proof the *exported* cpu_* C ABI
// itself works standalone. These three exercise only cpu_create/cpu_run/
// cpu_get_reg/cpu_destroy/cpu_add_breakpoint/cpu_breakpoint_hit/cpu_set_eip --
// the same surface any external consumer (e.g. pe-walker, over direct Zig
// @import rather than a C ABI at all, or the Python harness over ctypes)
// would use.

test "public C ABI: cpu_create/run/get_reg/destroy round-trip" {
    var mem = [_]u8{ 0xB8, 0x2A, 0x00, 0x00, 0x00, 0xF4 } ++ [_]u8{0} ** 58; // mov eax,0x2a; hlt
    const s = cpu_create(&mem, mem.len).?;
    defer cpu_destroy(s);
    const result = cpu_run(s, 100);
    try testing.expectEqual(RunResult.halted, result);
    try testing.expectEqual(@as(u32, 0x2a), cpu_get_reg(s, EAX));
    try testing.expect(!cpu_is_faulted(s));
}

test "public C ABI: breakpoint halts before executing, not after" {
    var mem = [_]u8{ 0x90, 0x90, 0xF4 } ++ [_]u8{0} ** 61; // nop; nop; hlt
    const s = cpu_create(&mem, mem.len).?;
    defer cpu_destroy(s);
    cpu_add_breakpoint(s, 2); // address of the hlt
    const result = cpu_run(s, 100);
    try testing.expectEqual(RunResult.halted, result);
    try testing.expect(cpu_breakpoint_hit(s));
    try testing.expectEqual(@as(u32, 2), cpu_breakpoint_hit_eip(s));
    try testing.expectEqual(@as(u32, 2), cpu_get_eip(s)); // hlt NOT yet executed
}

test "public C ABI: out-of-bounds fetch faults, not crashes" {
    var mem = [_]u8{ 0x90, 0x90 };
    const s = cpu_create(&mem, mem.len).?;
    defer cpu_destroy(s);
    cpu_set_eip(s, 10); // well past memory_size
    const result = cpu_run(s, 10);
    try testing.expectEqual(RunResult.faulted, result);
    try testing.expect(cpu_is_faulted(s));
}

test "public C ABI: null-page guard disabled by default -- low-address code/data still work" {
    // mov al, [0x190]; hlt -- a genuine data read from a low address, same
    // shape as MCity_d.exe's own anti-debug self-test (_CLayer_DetectDebugger
    // reading 0x00000190). Buffer is well over 0x190 bytes so both the
    // instruction fetch (near address 0) and the data read (at 0x190) are
    // in-bounds; with the guard off (the default every existing test in this
    // file already relies on), neither should fault.
    var mem = [_]u8{0} ** 0x400;
    mem[0] = 0xA0; // MOV AL, moffs8
    mem[1] = 0x90; mem[2] = 0x01; mem[3] = 0x00; mem[4] = 0x00; // addr = 0x00000190 (LE)
    mem[5] = 0xF4; // hlt
    const s = cpu_create(&mem, mem.len).?;
    defer cpu_destroy(s);
    const result = cpu_run(s, 10);
    try testing.expectEqual(RunResult.halted, result);
    try testing.expect(!cpu_is_faulted(s));
}

test "public C ABI: null-page guard, once enabled, faults a real data read at a low address" {
    var mem = [_]u8{0} ** 0x400;
    mem[0] = 0xA0; // MOV AL, moffs8
    mem[1] = 0x90; mem[2] = 0x01; mem[3] = 0x00; mem[4] = 0x00; // addr = 0x00000190 (LE)
    const s = cpu_create(&mem, mem.len).?;
    defer cpu_destroy(s);
    cpu_enable_null_page_guard(s);
    const result = cpu_run(s, 2);
    try testing.expectEqual(RunResult.faulted, result);
    try testing.expect(cpu_is_faulted(s));
}

test "public C ABI: null-page guard, once enabled, still allows addresses at and above 0x10000" {
    // Same mov al,[addr] shape, but this time addr is exactly NULL_PAGE_SIZE
    // -- the boundary itself must remain valid, only addresses strictly
    // below it are guarded. The code itself is placed well above 0x10000
    // too (via cpu_set_eip), not at address 0 like the other tests here --
    // with the guard enabled, a low-address code fetch would fault on its
    // own, which isn't what this test is isolating.
    const size: usize = 0x10100;
    const mem = try testing.allocator.alloc(u8, size);
    defer testing.allocator.free(mem);
    @memset(mem, 0);
    const code_addr: u32 = 0x10010;
    mem[code_addr] = 0xA0; // MOV AL, moffs8
    mem[code_addr + 1] = 0x00; mem[code_addr + 2] = 0x00; mem[code_addr + 3] = 0x01; mem[code_addr + 4] = 0x00; // addr = 0x00010000 (LE)
    mem[code_addr + 5] = 0xF4; // hlt
    const s = cpu_create(mem.ptr, mem.len).?;
    defer cpu_destroy(s);
    cpu_enable_null_page_guard(s);
    cpu_set_eip(s, code_addr);
    const result = cpu_run(s, 10);
    try testing.expectEqual(RunResult.halted, result); // in-bounds, guard doesn't apply here
    try testing.expect(!cpu_is_faulted(s));
}

test "public C ABI: scheduler create -> thread creation -> preempt_slice -> destroy" {
    // Heap-allocated (not a stack array like the other tests above): needs to
    // cover TEB (0x00320000+) plus a small test stack, too large to be safe
    // on the test runner's stack.
    const mem = try testing.allocator.alloc(u8, 0x00340000);
    defer testing.allocator.free(mem);
    @memset(mem, 0);
    const cpu = cpu_create(mem.ptr, mem.len).?;
    defer cpu_destroy(cpu);

    const sched = scheduler_create().?;
    defer scheduler_destroy(sched);
    sched.thread_stack_next = 0x1000;

    scheduler_create_main_thread(sched, 1000, 0xBEEF);
    _ = scheduler_create_thread(sched, 1001, 0xBEF0, 0x9F0000, 0x0, false);

    try testing.expectEqual(@as(i32, 0), scheduler_current_idx(sched));
    const switched = scheduler_preempt_slice(sched, cpu);
    try testing.expect(switched);
    try testing.expectEqual(@as(i32, 1), scheduler_current_idx(sched));
    try testing.expectEqual(@as(u32, 0x9F0000), cpu_get_eip(cpu));

    try testing.expectEqual(@as(u32, 0), scheduler_reentrant_depth(sched));
    scheduler_enter_reentrant_call(sched);
    try testing.expectEqual(@as(u32, 1), scheduler_reentrant_depth(sched));
    const refused = scheduler_switch_to(sched, cpu, 0);
    try testing.expect(!refused);
    scheduler_exit_reentrant_call(sched);
    try testing.expectEqual(@as(u32, 0), scheduler_reentrant_depth(sched));
}

test "public C ABI: scheduler block_on_handles -> pick_next_ready -> unblock -> tick" {
    const mem = try testing.allocator.alloc(u8, 0x00340000);
    defer testing.allocator.free(mem);
    @memset(mem, 0);
    const cpu = cpu_create(mem.ptr, mem.len).?;
    defer cpu_destroy(cpu);

    const sched = scheduler_create().?;
    defer scheduler_destroy(sched);
    sched.thread_stack_next = 0x1000;
    scheduler_create_main_thread(sched, 1000, 0xBEEF);
    _ = scheduler_create_thread(sched, 1001, 0xBEF0, 0x9F0000, 0x0, false);

    // Block the main thread on a handle; the two-call protocol: resolve
    // next_idx via pick_next_ready first, then complete the block.
    const next_idx = scheduler_pick_next_ready(sched, cpu);
    try testing.expectEqual(@as(i32, 1), next_idx);
    const handles = [_]u32{0x700B};
    const blocked = scheduler_complete_block_on_handles(sched, cpu, &handles, handles.len, 0x401000, true, 500, next_idx);
    try testing.expect(blocked);
    try testing.expectEqual(@as(i32, 1), scheduler_current_idx(sched));

    // Signal the handle: unblock_handle marks thread 0 READY again (does not
    // itself swap -- it only flips the flag, matching the real Win32
    // SetEvent-style handler that calls it).
    const n = scheduler_unblock_handle(sched, 0x700B);
    try testing.expectEqual(@as(u32, 1), n);
    try testing.expectEqual(scheduler.ThreadStatus.ready, sched.threads[0].status);

    // tick() advances the virtual clock -- smoke test for the C ABI wiring
    // itself; the actual wake/deadline-expiry math is covered by
    // scheduler.zig's own colocated tests.
    scheduler_tick(sched, 10);
    try testing.expectEqual(@as(u32, 10), sched.virtual_ticks_ms);
}

test "public C ABI: scheduler TLS bitset + handle-keyed accessors" {
    const sched = scheduler_create().?;
    defer scheduler_destroy(sched);
    scheduler_create_main_thread(sched, 1000, 0xBEEF);
    _ = scheduler_create_thread(sched, 1001, 0xBEF0, 0x9F0000, 0x0, false);

    try testing.expect(!scheduler_tls_slot_allocated(sched, 5));
    scheduler_tls_alloc_slot(sched, 5);
    try testing.expect(scheduler_tls_slot_allocated(sched, 5));
    scheduler_tls_free_slot(sched, 5);
    try testing.expect(!scheduler_tls_slot_allocated(sched, 5));

    try testing.expect(!scheduler_get_suspended(sched, 0xBEF0));
    scheduler_set_suspended(sched, 0xBEF0, true);
    try testing.expect(scheduler_get_suspended(sched, 0xBEF0));

    try testing.expectEqual(@as(u32, 0xBEEF), scheduler_current_handle(sched));
    try testing.expectEqual(@as(i64, 1001), scheduler_get_thread_id(sched, 0xBEF0));
    try testing.expectEqual(@as(i64, 0xBEF0), scheduler_handle_at_idx(sched, 1));
    try testing.expectEqual(@as(i64, -1), scheduler_get_thread_id(sched, 0xDEADBEEF));

    try testing.expectEqual(@as(u32, 2), scheduler_thread_count(sched));
    try testing.expectEqual(@as(u32, 0), scheduler_get_virtual_ticks_ms(sched));
    scheduler_set_virtual_ticks_ms(sched, 12345);
    try testing.expectEqual(@as(u32, 12345), scheduler_get_virtual_ticks_ms(sched));
}

test "public C ABI: scheduler create 2 threads, preempt, block on CS, unblock, verify swap" {
    const mem = try testing.allocator.alloc(u8, 0x00340000);
    defer testing.allocator.free(mem);
    @memset(mem, 0);
    const cpu = cpu_create(mem.ptr, mem.len).?;
    defer cpu_destroy(cpu);

    const sched = scheduler_create().?;
    defer scheduler_destroy(sched);
    sched.thread_stack_next = 0x1000;
    scheduler_create_main_thread(sched, 1000, 0xBEEF);
    _ = scheduler_create_thread(sched, 1001, 0xBEF0, 0x9F0000, 0x0, false);

    // preempt: main (idx 0) -> idx 1, fresh thread, real stack init.
    try testing.expect(scheduler_preempt_slice(sched, cpu));
    try testing.expectEqual(@as(i32, 1), scheduler_current_idx(sched));
    try testing.expectEqual(@as(u32, 0x9F0000), cpu_get_eip(cpu));

    // idx 1 blocks on a contested CS; two-call protocol: resolve next_idx
    // via pick_next_ready first (finds idx 0, READY), then complete.
    const next_idx = scheduler_pick_next_ready(sched, cpu);
    try testing.expectEqual(@as(i32, 0), next_idx);
    const blocked = scheduler_complete_block_on_cs(sched, cpu, 0xCAFE, 0x9F0010, next_idx);
    try testing.expect(blocked);
    try testing.expectEqual(@as(i32, 0), scheduler_current_idx(sched));
    // eip restored to idx 0's saved state (from the earlier preempt's save),
    // not 0x9F0010 -- that retry_eip belongs to idx 1's own saved state now.
    try testing.expectEqual(@as(u32, 0), cpu_get_eip(cpu));

    // unblock: idx 1 becomes READY again but current thread (idx 0) doesn't
    // move on its own -- unblock_cs only flips the flag, matching the real
    // LeaveCriticalSection handler that calls it.
    scheduler_unblock_cs(sched, 0xCAFE);
    const after_unblock = scheduler_pick_next_ready(sched, cpu);
    try testing.expectEqual(@as(i32, 1), after_unblock);

    const swapped = scheduler_switch_to(sched, cpu, 1);
    try testing.expect(swapped);
    try testing.expectEqual(@as(i32, 1), scheduler_current_idx(sched));
    try testing.expectEqual(@as(u32, 0x9F0010), cpu_get_eip(cpu)); // idx 1's own retry_eip, restored
}
