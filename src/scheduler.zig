// SPDX-License-Identifier: LGPL-3.0-or-later
// scheduler.zig — cooperative thread scheduler, ported from tew/kernel/scheduler.py.
//
// Stage 1 of the scheduler-to-Zig port (see
// ~/.claude/plans/vast-drifting-pike.md): SchedulerState, context switch
// (saveCurrent/loadThread/initThreadStack/loadNext/switchTo/preemptSlice),
// pickNextReady's scan A + scan B, and the reentrancy guard. Blocking/wake/
// tick operations (Stage 2), the TLS bitset alloc/free/allocated exports and
// direct-field accessors (Stage 3), and the Python wrapper (Stage 4) are not
// yet implemented here.
//
// Not ported here: `reentrancy_violations` (Python keeps a list of message
// strings for tests/diagnostics). Building/logging those strings stays a
// Python-side concern in the Stage 4 wrapper -- it constructs the message
// from the bool a native swap call returns, rather than allocating/crossing
// strings over the FFI boundary for every refusal.
const std = @import("std");
const core = @import("core.zig");
const CpuState = core.CpuState;
const EAX = core.EAX; const ECX = core.ECX; const EDX = core.EDX; const EBX = core.EBX;
const ESP = core.ESP; const EBP = core.EBP; const ESI = core.ESI; const EDI = core.EDI;

const testing = std.testing;

// ─── Constants (mirror tew/kernel/scheduler.py) ─────────────────────────────
pub const TEB_BASE: u32 = 0x00320000;
const TLS_TEB_OFFSET: u32 = 0xE0;
const LAST_ERROR_TEB_OFFSET: u32 = 0x34;
pub const THREAD_STACK_BASE: u32 = 0x08000000;
pub const THREAD_STACK_SIZE: u32 = 256 * 1024;
pub const THREAD_SENTINEL: u32 = 0x001FE000;

// No existing MAX_THREADS cap in the Python version (unbounded list); this
// is a generous fixed bound for a single-game-process emulator (grounding
// note in the plan). TLS_MAX_SLOTS mirrors the real, already-enforced
// TLS_MAX_SLOTS=64 Windows limit (tew/api/_state.py). MAX_WAIT_HANDLES
// mirrors real Windows' MAXIMUM_WAIT_OBJECTS=64 (WaitForMultipleObjects).
pub const MAX_THREADS: usize = 64;
pub const TLS_MAX_SLOTS: usize = 64;
pub const MAX_WAIT_HANDLES: usize = 64;

pub const ThreadStatus = enum(u8) { ready, blocked_cs, blocked_handles, sleeping, dead };

pub const SavedRegs = struct {
    regs: [8]u32 = .{0} ** 8,
    eip: u32 = 0,
    eflags: u32 = 0,
    fpu_stack: [8]f80 = .{0.0} ** 8,
    fpu_top: u32 = 0,
    fpu_status_word: u16 = 0,
    fpu_control_word: u16 = 0x037F,
    fpu_tag_word: u16 = 0xFFFF,
};

pub const ThreadEntry = struct {
    thread_id: u32 = 0,
    handle: u32 = 0,
    start_address: u32 = 0,
    parameter: u32 = 0,
    status: ThreadStatus = .ready,
    suspended: bool = false,
    // Mirrors Python's `saved_state is None` sentinel ("has this thread ever
    // been swapped away from, i.e. does `saved` hold real state yet?"). Set
    // only by saveCurrent -- NOT by loadNext's fresh-start branch, matching
    // Python exactly (see loadNext's comment).
    has_run: bool = false,
    saved: SavedRegs = .{},
    tls_values: [TLS_MAX_SLOTS]u32 = .{0} ** TLS_MAX_SLOTS,
    last_error: u32 = 0,
    waiting_on_cs: u32 = 0, // 0 = sentinel "not waiting" -- no legit guest CS pointer is ever 0
    wait_handles: [MAX_WAIT_HANDLES]u32 = .{0} ** MAX_WAIT_HANDLES,
    wait_handle_count: u8 = 0,
    wait_deadline_ms: ?u32 = null,
    wait_timed_out: bool = false,
    sleep_until_ms: u32 = 0,
};

pub const SchedulerState = struct {
    threads: [MAX_THREADS]ThreadEntry = .{ThreadEntry{}} ** MAX_THREADS,
    thread_count: u32 = 0,
    current_idx: i32 = -1,
    virtual_ticks_ms: u32 = 0,
    tls_allocated: u64 = 0,
    thread_stack_next: u32 = THREAD_STACK_BASE,
    last_scheduled_idx: u32 = 0,
    reentrant_depth: u32 = 0,
};

// ─── Thread registration ─────────────────────────────────────────────────────

pub fn createMainThread(sched: *SchedulerState, thread_id: u32, handle: u32) void {
    std.debug.assert(sched.thread_count == 0);
    sched.threads[0] = ThreadEntry{ .thread_id = thread_id, .handle = handle };
    sched.thread_count = 1;
    sched.current_idx = 0;
    sched.last_scheduled_idx = 0;
}

pub fn createThread(sched: *SchedulerState, thread_id: u32, handle: u32, start_address: u32, parameter: u32, suspended: bool) u32 {
    std.debug.assert(sched.thread_count < MAX_THREADS);
    const idx = sched.thread_count;
    sched.threads[idx] = ThreadEntry{
        .thread_id = thread_id,
        .handle = handle,
        .start_address = start_address,
        .parameter = parameter,
        .status = .ready,
        .suspended = suspended,
    };
    sched.thread_count += 1;
    return idx;
}

// ─── Internal: TLS / last-error (see saveCurrent/loadThread) ────────────────

fn tebTlsAddr(slot: u6) u32 {
    return TEB_BASE + TLS_TEB_OFFSET + @as(u32, slot) * 4;
}

fn saveTls(sched: *SchedulerState, cpu: *CpuState, t: *ThreadEntry) void {
    var bits = sched.tls_allocated;
    while (bits != 0) {
        const slot: u6 = @intCast(@ctz(bits));
        t.tls_values[slot] = core.memRead32(cpu, tebTlsAddr(slot));
        bits &= bits - 1;
    }
}

fn loadTls(sched: *SchedulerState, cpu: *CpuState, t: *ThreadEntry) void {
    var bits = sched.tls_allocated;
    while (bits != 0) {
        const slot: u6 = @intCast(@ctz(bits));
        core.memWrite32(cpu, tebTlsAddr(slot), t.tls_values[slot]);
        bits &= bits - 1;
    }
}

// ─── Internal: CPU state ─────────────────────────────────────────────────────

pub fn saveCurrent(sched: *SchedulerState, cpu: *CpuState) void {
    const idx: usize = @intCast(sched.current_idx);
    const t = &sched.threads[idx];
    t.saved = SavedRegs{
        .regs = cpu.regs,
        .eip = cpu.eip,
        .eflags = cpu.eflags,
        .fpu_stack = cpu.fpu_stack,
        .fpu_top = cpu.fpu_top,
        .fpu_status_word = cpu.fpu_status_word,
        .fpu_control_word = cpu.fpu_control_word,
        .fpu_tag_word = cpu.fpu_tag_word,
    };
    t.has_run = true;
    saveTls(sched, cpu, t);
    t.last_error = core.memRead32(cpu, TEB_BASE + LAST_ERROR_TEB_OFFSET);
}

pub fn loadThread(sched: *SchedulerState, cpu: *CpuState, idx: u32) void {
    const t = &sched.threads[idx];
    loadTls(sched, cpu, t);
    core.memWrite32(cpu, TEB_BASE + LAST_ERROR_TEB_OFFSET, t.last_error);
    cpu.regs = t.saved.regs;
    cpu.eip = t.saved.eip;
    cpu.eflags = t.saved.eflags;
    cpu.fpu_stack = t.saved.fpu_stack;
    cpu.fpu_top = t.saved.fpu_top;
    cpu.fpu_status_word = t.saved.fpu_status_word;
    cpu.fpu_control_word = t.saved.fpu_control_word;
    cpu.fpu_tag_word = t.saved.fpu_tag_word;
    // restore does not touch halted; clear explicitly -- but never override a
    // fatal halt (e.g. an unimplemented Win32 API): that must stop the whole
    // emulator, not just this one thread's slice. Permanent invariant, must
    // survive every swap/load/wake path ported into this file.
    if (!cpu.fatal_halted) cpu.halted = false;
    sched.current_idx = @intCast(idx);
}

pub fn initThreadStack(sched: *SchedulerState, cpu: *CpuState, idx: u32) void {
    const t = &sched.threads[idx];
    const stack_top = sched.thread_stack_next +% THREAD_STACK_SIZE -% 16;
    sched.thread_stack_next +%= THREAD_STACK_SIZE;
    var esp = stack_top -% 4;
    core.memWrite32(cpu, esp, t.parameter);
    esp -%= 4;
    core.memWrite32(cpu, esp, THREAD_SENTINEL);
    core.memWrite32(cpu, TEB_BASE + LAST_ERROR_TEB_OFFSET, t.last_error); // fresh thread: last_error=0
    cpu.regs[EAX] = 0;
    cpu.regs[ECX] = 0;
    cpu.regs[EDX] = 0;
    cpu.regs[EBX] = 0;
    cpu.regs[ESI] = 0;
    cpu.regs[EDI] = 0;
    cpu.regs[ESP] = esp;
    cpu.regs[EBP] = 0;
    cpu.eip = t.start_address;
    cpu.eflags = 0x202;
}

pub fn loadNext(sched: *SchedulerState, cpu: *CpuState, idx: u32) void {
    sched.last_scheduled_idx = idx;
    if (!sched.threads[idx].has_run) {
        initThreadStack(sched, cpu, idx);
        sched.current_idx = @intCast(idx);
        if (!cpu.fatal_halted) cpu.halted = false;
    } else {
        loadThread(sched, cpu, idx);
    }
}

// ─── Internal: scheduling ────────────────────────────────────────────────────
// Scan A + scan B only -- the coldest branch (kernel.tick() poll for pending
// I/O completions) has no Zig-side equivalent yet. Per the plan's Design
// Decision 2, that stays an explicit Python-driven two-call retry: callers
// that get null back call self._kernel.tick() then retry this once, rather
// than this function calling back into Python mid-scan.
pub fn pickNextReady(sched: *SchedulerState, cpu: *CpuState) ?u32 {
    const n = sched.thread_count;
    if (n == 0) return null;
    const start = (sched.last_scheduled_idx + 1) % n;

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const idx = (start + i) % n;
        if (@as(i32, @intCast(idx)) == sched.current_idx) continue;
        const t = &sched.threads[idx];
        if (t.suspended) continue;
        if (t.status == .dead) continue;
        if (t.status == .sleeping) {
            if (sched.virtual_ticks_ms < t.sleep_until_ms) continue;
            t.status = .ready;
        }
        if (t.status == .blocked_cs) {
            if (t.waiting_on_cs != 0) {
                const owner = core.memRead32(cpu, t.waiting_on_cs +% 0x0C);
                if (owner != 0) continue;
            }
            t.waiting_on_cs = 0;
            t.status = .ready;
        }
        if (t.status == .blocked_handles) {
            if (t.wait_deadline_ms) |dl| {
                if (sched.virtual_ticks_ms >= dl) {
                    t.wait_timed_out = true;
                    t.wait_deadline_ms = null;
                    t.wait_handle_count = 0;
                    t.status = .ready;
                } else continue;
            } else continue;
        }
        if (t.status == .ready) return idx;
    }

    // Fallback pass 1: wake the SLEEPING thread with the earliest deadline so
    // a blocking background thread doesn't starve a sleeping main thread.
    // Pass 2: if no sleeping thread, wake a BLOCKED_HANDLES thread so it can
    // retry its wait -- handles may be signaled by the heartbeat between
    // batches; cpu.halted is never used for this case.
    var earliest_sleep_idx: ?u32 = null;
    var earliest_sleep_ms: ?u32 = null;
    var blocked_fallback_idx: ?u32 = null;
    i = 0;
    while (i < n) : (i += 1) {
        const idx = (start + i) % n;
        if (@as(i32, @intCast(idx)) == sched.current_idx) continue;
        const t = &sched.threads[idx];
        if (t.suspended or t.status == .dead) continue;
        if (t.status == .sleeping) {
            if (earliest_sleep_ms == null or t.sleep_until_ms < earliest_sleep_ms.?) {
                earliest_sleep_ms = t.sleep_until_ms;
                earliest_sleep_idx = idx;
            }
        } else if (t.status == .blocked_handles and blocked_fallback_idx == null) {
            blocked_fallback_idx = idx;
        }
    }
    if (earliest_sleep_idx) |idx| {
        sched.threads[idx].status = .ready;
        return idx;
    }
    if (blocked_fallback_idx) |idx| {
        sched.threads[idx].status = .ready;
        return idx;
    }
    return null;
}

// ─── Reentrancy guard ─────────────────────────────────────────────────────────
// See tew/kernel/scheduler.py's matching section for the full rationale:
// tew's CPU has exactly one register file, shared by every thread; a nested
// synchronous cpu_run (e.g. invoking a DllMain from inside a stub handler)
// must not have the shared registers handed to a different thread out from
// under it. depth-counted, not a flag, so a nested call made from inside
// another nested call stays safe.

pub fn enterReentrantCall(sched: *SchedulerState) void {
    sched.reentrant_depth += 1;
}

pub fn exitReentrantCall(sched: *SchedulerState) void {
    std.debug.assert(sched.reentrant_depth > 0);
    sched.reentrant_depth -= 1;
}

fn reentrancyOk(sched: *SchedulerState) bool {
    return sched.reentrant_depth == 0;
}

// Single chokepoint for handing the shared CPU registers to a different
// thread. mark_current_dead/terminate_thread (Stage 2) deliberately do NOT
// route through here -- a thread dying mid-nested-call must still be able to
// hand off the CPU (see scheduler.py's _swap_current docstring).
fn swapCurrent(sched: *SchedulerState, cpu: *CpuState, target_idx: u32) bool {
    if (!reentrancyOk(sched)) return false;
    if (sched.current_idx >= 0) {
        const cur: usize = @intCast(sched.current_idx);
        if (sched.threads[cur].status != .dead) saveCurrent(sched, cpu);
    }
    loadNext(sched, cpu, target_idx);
    return true;
}

// ─── Public: context switch ──────────────────────────────────────────────────

pub fn switchTo(sched: *SchedulerState, cpu: *CpuState, idx: u32) bool {
    return swapCurrent(sched, cpu, idx);
}

pub fn preemptSlice(sched: *SchedulerState, cpu: *CpuState) bool {
    if (cpu.fatal_halted) return false; // single core, fatally locked up -- nothing to hand it to
    const cur: usize = @intCast(sched.current_idx);
    if (sched.threads[cur].status != .ready) return false; // already switched mid-batch
    const n = sched.thread_count;
    var i: u32 = 1;
    while (i < n) : (i += 1) {
        const idx = (@as(u32, @intCast(sched.current_idx)) + i) % n;
        const t = &sched.threads[idx];
        if (!t.suspended and t.status == .ready) return switchTo(sched, cpu, idx);
    }
    return false;
}

// ─── Public: blocking operations ─────────────────────────────────────────────
// Stage 2 of the port. Design Decision 2 (see the plan): pickNextReady above
// stops short of the kernel-tick fallback, so these `complete*` functions
// take an already-resolved `next_idx` rather than scanning themselves. The
// intended caller (Stage 4's Python wrapper) is:
//   1. check cpu.fatal_halt / scheduler_reentrant_depth==0 FIRST -- do not
//      call scheduler_pick_next_ready at all if either would refuse the
//      operation. This matters, not just style: pickNextReady has real side
//      effects (it flips a due sleeper or a now-free CS-owner's thread to
//      READY as it scans), and the original Python never reaches
//      _pick_next_ready at all on a fatal-halt/reentrancy short-circuit --
//      calling it unconditionally would leak those wake side effects into a
//      call that ends up refused, which is a real behavior change, not a
//      cosmetic one.
//   2. call scheduler_pick_next_ready(sched, cpu); if -1, call
//      self._kernel.tick() (unchanged plain Python) and retry once.
//   3. call the matching scheduler_complete_* with the resolved next_idx
//      (-1 meaning "reload self").
// Every complete_* function still re-checks fatal_halt/reentrancy itself
// (never trust the caller alone) -- each returns false, untouched, if either
// guard fires, exactly mirroring Python's own top-of-function short-circuit.

pub fn completeBlockOnCs(sched: *SchedulerState, cpu: *CpuState, cs_ptr: u32, retry_eip: u32, next_idx: i32) bool {
    if (cpu.fatal_halted) return false; // single core, fatally locked up -- nothing left to block/resume
    if (!reentrancyOk(sched)) {
        // Caller (e.g. _enter_cs) already skipped its own cleanup_stdcall on
        // this path, trusting the scheduler to redirect eip -- see the
        // matching Stage 1 comment on the same theme.
        cpu.eip = retry_eip;
        return false;
    }
    const cur: usize = @intCast(sched.current_idx);
    var t = &sched.threads[cur];
    t.waiting_on_cs = cs_ptr;
    t.status = .blocked_cs;
    cpu.eip = retry_eip;
    if (next_idx < 0) {
        t.status = .ready;
        _ = swapCurrent(sched, cpu, @intCast(cur));
        return true;
    }
    _ = swapCurrent(sched, cpu, @intCast(next_idx));
    return true;
}

pub fn completeBlockOnHandles(sched: *SchedulerState, cpu: *CpuState, handles: []const u32, retry_eip: u32, has_deadline: bool, deadline_ms: u32, next_idx: i32) bool {
    if (cpu.fatal_halted) return false;
    if (!reentrancyOk(sched)) {
        cpu.eip = retry_eip;
        return false;
    }
    const cur: usize = @intCast(sched.current_idx);
    var t = &sched.threads[cur];
    const n = @min(handles.len, MAX_WAIT_HANDLES);
    for (handles[0..n], 0..) |h, i| t.wait_handles[i] = h;
    t.wait_handle_count = @intCast(n);
    t.wait_deadline_ms = if (has_deadline) deadline_ms else null;
    t.wait_timed_out = false;
    t.status = .blocked_handles;
    cpu.eip = retry_eip;
    if (next_idx < 0) {
        t.status = .ready;
        t.wait_handle_count = 0;
        t.wait_deadline_ms = null;
        _ = swapCurrent(sched, cpu, @intCast(cur));
        return true;
    }
    _ = swapCurrent(sched, cpu, @intCast(next_idx));
    return true;
}

pub fn completeSleepCurrent(sched: *SchedulerState, cpu: *CpuState, return_eip: u32, eax_val: u32, sleep_ms: u32, next_idx: i32) bool {
    if (cpu.fatal_halted) return false;
    if (!reentrancyOk(sched)) {
        // Sleep/SleepEx callers already popped their own stack; GetMessageA's
        // poll-retry never pushed one -- either way cpu.eip (and EAX) must be
        // set unconditionally, see the matching Stage 1 comment.
        cpu.eip = return_eip;
        cpu.regs[EAX] = eax_val;
        return false;
    }
    const cur: usize = @intCast(sched.current_idx);
    cpu.eip = return_eip;
    cpu.regs[EAX] = eax_val;
    var t = &sched.threads[cur];
    t.sleep_until_ms = sched.virtual_ticks_ms +% sleep_ms;
    t.status = .sleeping;
    if (next_idx < 0) {
        t.status = .ready;
        _ = swapCurrent(sched, cpu, @intCast(cur));
        return true;
    }
    _ = swapCurrent(sched, cpu, @intCast(next_idx));
    return true;
}

// Deliberately bypasses the reentrancy guard entirely (no reentrancyOk check,
// no swapCurrent chokepoint) -- a thread dying mid-nested-call must still be
// able to hand off the CPU; that's the mechanism _invoke_emulated_proc's own
// thread-death detection depends on. Calls loadNext directly, same as
// Python's mark_current_dead calling _load_next directly. Returns true if the
// CPU halted (no runnable thread left), false if it switched to next_idx.
pub fn completeMarkCurrentDead(sched: *SchedulerState, cpu: *CpuState, next_idx: i32) bool {
    if (cpu.fatal_halted) return false; // nothing left to update
    const cur: usize = @intCast(sched.current_idx);
    var t = &sched.threads[cur];
    t.status = .dead;
    t.has_run = false; // mirrors Python's `saved_state = None`
    if (next_idx < 0) {
        cpu.halted = true;
        return true;
    }
    loadNext(sched, cpu, @intCast(next_idx));
    return false;
}

// Tri-state result matches Python's Optional[bool] exactly: -1 = handle not
// found, 1 = a *different* thread was terminated (caller does its own normal
// cleanup), 0 = the *current* thread terminated itself (already switched or
// halted -- caller must not touch cpu/EAX/ESP afterward). `next_idx` is used
// only for the self-terminate case (same two-call protocol as
// completeMarkCurrentDead); the different-thread case never touches the CPU
// at all, matching Python (`cpu.save_state.assert_not_called()`).
pub fn terminateThread(sched: *SchedulerState, cpu: *CpuState, handle: u32, next_idx: i32) i8 {
    var i: u32 = 0;
    while (i < sched.thread_count) : (i += 1) {
        if (sched.threads[i].handle != handle) continue;
        if (@as(i32, @intCast(i)) == sched.current_idx) {
            _ = completeMarkCurrentDead(sched, cpu, next_idx);
            return 0;
        }
        sched.threads[i].status = .dead;
        sched.threads[i].has_run = false;
        return 1;
    }
    return -1;
}

// ─── Public: unblocking ───────────────────────────────────────────────────────
// No reentrancy interaction at all -- these only flip *other* threads'
// status flags, never touch cpu or the current thread.

pub fn unblockCs(sched: *SchedulerState, cs_ptr: u32) void {
    var i: u32 = 0;
    while (i < sched.thread_count) : (i += 1) {
        var t = &sched.threads[i];
        if (t.status == .blocked_cs and t.waiting_on_cs == cs_ptr) {
            t.waiting_on_cs = 0;
            t.status = .ready;
        }
    }
}

pub fn unblockHandle(sched: *SchedulerState, handle: u32) u32 {
    var n: u32 = 0;
    var i: u32 = 0;
    while (i < sched.thread_count) : (i += 1) {
        var t = &sched.threads[i];
        if (t.status != .blocked_handles) continue;
        var found = false;
        var j: usize = 0;
        while (j < t.wait_handle_count) : (j += 1) {
            if (t.wait_handles[j] == handle) {
                found = true;
                break;
            }
        }
        if (found) {
            t.wait_handle_count = 0;
            t.status = .ready;
            n += 1;
        }
    }
    return n;
}

// ─── Public: clock ────────────────────────────────────────────────────────────

pub fn tick(sched: *SchedulerState, ms: u32) void {
    sched.virtual_ticks_ms +%= ms;
    var i: u32 = 0;
    while (i < sched.thread_count) : (i += 1) {
        var t = &sched.threads[i];
        if (t.status == .sleeping) {
            if (sched.virtual_ticks_ms >= t.sleep_until_ms) t.status = .ready;
        } else if (t.status == .blocked_handles) {
            if (t.wait_deadline_ms) |dl| {
                if (sched.virtual_ticks_ms >= dl) {
                    t.wait_timed_out = true;
                    t.wait_deadline_ms = null;
                    t.wait_handle_count = 0;
                    t.status = .ready;
                }
            }
        }
    }
}

// ─── Public: TLS bitset ───────────────────────────────────────────────────────
// TLS_MAX_SLOTS=64 mirrors the real, already-enforced Windows TLS_MAX_SLOTS
// limit (tew/api/_state.py's `_tls_alloc` already bounds-checks before ever
// calling this). saveTls/loadTls (Stage 1) already iterate this same bitset
// on every context switch -- these three exports are the alloc/free/query
// surface Stage 4's TlsAlloc/TlsFree/TlsSetValue/TlsGetValue handlers wire
// to. `slot` is caller-bounded (<64) by contract, same as core.zig's
// readReg8/fpu accessors accept an already-bounded index -- out-of-range
// values wrap via @truncate rather than fault, matching that precedent.

pub fn tlsAllocSlot(sched: *SchedulerState, slot: u6) void {
    sched.tls_allocated |= (@as(u64, 1) << slot);
}

pub fn tlsFreeSlot(sched: *SchedulerState, slot: u6) void {
    sched.tls_allocated &= ~(@as(u64, 1) << slot);
}

pub fn tlsSlotAllocated(sched: *SchedulerState, slot: u6) bool {
    return (sched.tls_allocated & (@as(u64, 1) << slot)) != 0;
}

// ─── Public: handle-keyed thread accessors ────────────────────────────────────
// One accessor per direct ThreadState field poke found in the plan's
// call-site inventory (kernel32_io.py's _resume_thread/_suspend_thread/
// _get_exit_code_thread/_wait_for_single/_wait_for_multiple_ex,
// user32_handlers.py's index-based DEAD check, crt_handlers.py's .thread_id
// read) -- keyed by handle, not index or thread_id, since that's what every
// external caller already has in hand. handleAtIdx translates the one
// index-based call site to a handle so it can reuse these instead of
// needing an `_by_idx` variant of each.
//
// Convention: an unresolvable handle/index gets a benign sentinel (false /
// -1 / 0xFF), never a crash -- matching cpu_get_reg/cpu_fpu_get's existing
// out-of-range convention elsewhere in this file. The invariant that a
// handle a caller holds actually exists is enforced where the handle is
// handed out (Python), not re-litigated on every read here.

fn findByHandle(sched: *SchedulerState, handle: u32) ?u32 {
    var i: u32 = 0;
    while (i < sched.thread_count) : (i += 1) {
        if (sched.threads[i].handle == handle) return i;
    }
    return null;
}

pub fn getSuspended(sched: *SchedulerState, handle: u32) bool {
    const idx = findByHandle(sched, handle) orelse return false;
    return sched.threads[idx].suspended;
}

pub fn setSuspended(sched: *SchedulerState, handle: u32, val: bool) void {
    const idx = findByHandle(sched, handle) orelse return;
    sched.threads[idx].suspended = val;
}

pub fn getCompleted(sched: *SchedulerState, handle: u32) bool {
    const idx = findByHandle(sched, handle) orelse return false;
    return sched.threads[idx].status == .dead;
}

pub fn getWaitTimedOut(sched: *SchedulerState, handle: u32) bool {
    const idx = findByHandle(sched, handle) orelse return false;
    return sched.threads[idx].wait_timed_out;
}

pub fn setWaitTimedOut(sched: *SchedulerState, handle: u32, val: bool) void {
    const idx = findByHandle(sched, handle) orelse return;
    sched.threads[idx].wait_timed_out = val;
}

// 0xFF = handle not found -- ThreadStatus only ever occupies values 0-4.
pub fn getStatus(sched: *SchedulerState, handle: u32) u8 {
    const idx = findByHandle(sched, handle) orelse return 0xFF;
    return @intFromEnum(sched.threads[idx].status);
}

pub fn getThreadId(sched: *SchedulerState, handle: u32) i64 {
    const idx = findByHandle(sched, handle) orelse return -1;
    return sched.threads[idx].thread_id;
}

pub fn handleAtIdx(sched: *SchedulerState, idx: u32) i64 {
    if (idx >= sched.thread_count) return -1;
    return sched.threads[idx].handle;
}

pub fn currentHandle(sched: *SchedulerState) u32 {
    if (sched.current_idx < 0) return 0; // no current thread; 0 is never a real handle
    const idx: usize = @intCast(sched.current_idx);
    return sched.threads[idx].handle;
}

// ─── Tests ────────────────────────────────────────────────────────────────────
// A real backing buffer is required (unlike the Python suite's MagicMock
// memory) since this code performs genuine memRead32/memWrite32 calls against
// `cpu.memory`. thread_stack_next is overridden to a small in-bounds value in
// every test that exercises initThreadStack, rather than sizing the buffer to
// cover the real THREAD_STACK_BASE (0x08000000) -- Python's Scheduler
// constructor accepts the same override (`thread_stack_next` param).
const TEST_MEM_SIZE: usize = 0x00340000;

fn allocTestMem() ![]u8 {
    const buf = try testing.allocator.alloc(u8, TEST_MEM_SIZE);
    @memset(buf, 0);
    return buf;
}

fn testCpu(mem: []u8) CpuState {
    return CpuState{ .memory = mem.ptr, .memory_size = mem.len };
}

fn twoThreadSched() SchedulerState {
    var sched = SchedulerState{};
    sched.thread_stack_next = 0x1000;
    createMainThread(&sched, 1000, 0xBEEF);
    _ = createThread(&sched, 1001, 0xBEF0, 0x9F0000, 0x0, false);
    return sched;
}

test "create_main_thread adds thread at index 0" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    try testing.expectEqual(@as(u32, 1), sched.thread_count);
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
    try testing.expectEqual(@as(u32, 1000), sched.threads[0].thread_id);
    try testing.expectEqual(@as(u32, 0xBEEF), sched.threads[0].handle);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status);
}

test "create_thread appends ready thread" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    const idx = createThread(&sched, 1001, 0xBEF0, 0x9F0000, 0x40001DC0, false);
    try testing.expectEqual(@as(u32, 1), idx);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[idx].status);
    try testing.expect(!sched.threads[idx].has_run);
}

test "create_thread suspended flag" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    const idx = createThread(&sched, 1001, 0xBEF0, 0x9F0000, 0x0, true);
    try testing.expect(sched.threads[idx].suspended);
}

// ── pickNextReady ────────────────────────────────────────────────────────────

test "pick_next_ready picks background thread, skips current" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    const idx = pickNextReady(&sched, &cpu);
    try testing.expectEqual(@as(?u32, 1), idx);
}

test "pick_next_ready skips dead" {
    var sched = twoThreadSched();
    sched.threads[1].status = .dead;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    try testing.expectEqual(@as(?u32, null), pickNextReady(&sched, &cpu));
}

test "pick_next_ready skips suspended" {
    var sched = twoThreadSched();
    sched.threads[1].suspended = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    try testing.expectEqual(@as(?u32, null), pickNextReady(&sched, &cpu));
}

test "pick_next_ready falls back to earliest sleeper when none due" {
    var sched = twoThreadSched();
    sched.threads[1].status = .sleeping;
    sched.threads[1].sleep_until_ms = 9999;
    sched.virtual_ticks_ms = 0;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    const idx = pickNextReady(&sched, &cpu);
    try testing.expectEqual(@as(?u32, 1), idx);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
}

test "pick_next_ready wakes sleeping thread when due" {
    var sched = twoThreadSched();
    sched.threads[1].status = .sleeping;
    sched.threads[1].sleep_until_ms = 100;
    sched.virtual_ticks_ms = 100;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    const idx = pickNextReady(&sched, &cpu);
    try testing.expectEqual(@as(?u32, 1), idx);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
}

test "pick_next_ready skips blocked_cs when owner nonzero" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_cs;
    sched.threads[1].waiting_on_cs = 0x1234;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    core.memWrite32(&cpu, 0x1234 + 0x0C, 0x3E9); // non-zero owner
    try testing.expectEqual(@as(?u32, null), pickNextReady(&sched, &cpu));
}

test "pick_next_ready unblocks blocked_cs when owner free" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_cs;
    sched.threads[1].waiting_on_cs = 0x1234;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    core.memWrite32(&cpu, 0x1234 + 0x0C, 0); // CS free
    const idx = pickNextReady(&sched, &cpu);
    try testing.expectEqual(@as(?u32, 1), idx);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
    try testing.expectEqual(@as(u32, 0), sched.threads[1].waiting_on_cs);
}

test "pick_next_ready falls back to blocked_handles" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_handles;
    sched.threads[1].wait_deadline_ms = null;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    const idx = pickNextReady(&sched, &cpu);
    try testing.expectEqual(@as(?u32, 1), idx);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
}

test "pick_next_ready unblocks handles on deadline" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_handles;
    sched.threads[1].wait_deadline_ms = 50;
    sched.virtual_ticks_ms = 50;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    const idx = pickNextReady(&sched, &cpu);
    try testing.expectEqual(@as(?u32, 1), idx);
    try testing.expect(sched.threads[1].wait_timed_out);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
}

test "pick_next_ready round robin" {
    var sched = SchedulerState{};
    sched.thread_stack_next = 0x1000;
    createMainThread(&sched, 1000, 0xBEEF);
    _ = createThread(&sched, 1001, 0xBEF0, 0x9F0000, 0x0, false);
    _ = createThread(&sched, 1002, 0xBEF1, 0x9F0000, 0x0, false);
    sched.last_scheduled_idx = 1;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    try testing.expectEqual(@as(?u32, 2), pickNextReady(&sched, &cpu));
}

// ── switch_to ────────────────────────────────────────────────────────────────

test "switch_to saves current and loads fresh next thread" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = switchTo(&sched, &cpu, 1);

    try testing.expect(result);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
    try testing.expectEqual(@as(u32, 0x9F0000), cpu.eip); // fresh thread -> initThreadStack ran
    try testing.expect(sched.threads[0].has_run); // outgoing thread was saved
}

test "switch_to loads saved state on resume" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    sched.threads[1].saved.eip = 0x9F1234;
    sched.threads[1].saved.regs[EAX] = 0x42;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    _ = switchTo(&sched, &cpu, 1);

    try testing.expectEqual(@as(i32, 1), sched.current_idx);
    try testing.expectEqual(@as(u32, 0x9F1234), cpu.eip);
    try testing.expectEqual(@as(u32, 0x42), cpu.regs[EAX]);
}

test "switch_to clears cpu.halted after load" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.halted = true;

    _ = switchTo(&sched, &cpu, 1);

    try testing.expect(!cpu.halted);
}

test "switch_to does not clear fatal_halt on saved thread load" {
    // Regression: an unimplemented-API handler sets both cpu.halted and
    // cpu.fatal_halted. A routine thread switch must not silently erase a
    // genuinely fatal halt and let the game keep running on garbage state.
    var sched = twoThreadSched();
    sched.threads[1].has_run = true; // resume branch, not fresh-start branch
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.halted = true;
    cpu.fatal_halted = true;

    _ = switchTo(&sched, &cpu, 1);

    try testing.expect(cpu.halted);
}

test "switch_to does not clear fatal_halt on fresh thread start" {
    // Same regression, for initThreadStack's own separate unconditional
    // cpu.halted = false.
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.halted = true;
    cpu.fatal_halted = true;

    _ = switchTo(&sched, &cpu, 1);

    try testing.expect(cpu.halted);
}

test "switch_to does not save a dead current thread" {
    var sched = twoThreadSched();
    sched.threads[0].status = .dead;
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.regs[EAX] = 0xDEAD; // would be captured by saveCurrent if wrongly called

    _ = switchTo(&sched, &cpu, 1);

    try testing.expectEqual(@as(u32, 0), sched.threads[0].saved.regs[EAX]);
}

// ── preempt_slice ────────────────────────────────────────────────────────────

test "preempt_slice switches to next ready thread" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = preemptSlice(&sched, &cpu);

    try testing.expect(result);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
}

test "preempt_slice does nothing when fatally halted" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.fatal_halted = true;

    const result = preemptSlice(&sched, &cpu);

    try testing.expect(!result);
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
}

// ── reentrancy guard ─────────────────────────────────────────────────────────

test "reentrant_depth starts at zero" {
    const sched = SchedulerState{};
    try testing.expectEqual(@as(u32, 0), sched.reentrant_depth);
}

test "reentrant_depth enter increments" {
    var sched = SchedulerState{};
    enterReentrantCall(&sched);
    try testing.expectEqual(@as(u32, 1), sched.reentrant_depth);
}

test "reentrant_depth exit decrements" {
    var sched = SchedulerState{};
    enterReentrantCall(&sched);
    exitReentrantCall(&sched);
    try testing.expectEqual(@as(u32, 0), sched.reentrant_depth);
}

test "reentrant_depth nested enter/exit tracks depth" {
    var sched = SchedulerState{};
    enterReentrantCall(&sched);
    enterReentrantCall(&sched);
    try testing.expectEqual(@as(u32, 2), sched.reentrant_depth);
    exitReentrantCall(&sched);
    try testing.expectEqual(@as(u32, 1), sched.reentrant_depth);
    exitReentrantCall(&sched);
    try testing.expectEqual(@as(u32, 0), sched.reentrant_depth);
}

test "switch_to refused while reentrant, current_idx unchanged" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    enterReentrantCall(&sched);

    const result = switchTo(&sched, &cpu, 1);

    try testing.expect(!result);
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
    try testing.expect(!sched.threads[0].has_run); // saveCurrent never ran
}

test "preempt_slice refused while reentrant" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    enterReentrantCall(&sched);

    const result = preemptSlice(&sched, &cpu);

    try testing.expect(!result);
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
}

test "switch_to still works at reentrant_depth zero" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = switchTo(&sched, &cpu, 1);

    try testing.expect(result);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
}

test "enter then exit restores normal swapping" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    enterReentrantCall(&sched);
    const refused = switchTo(&sched, &cpu, 1);
    exitReentrantCall(&sched);
    const allowed = switchTo(&sched, &cpu, 1);

    try testing.expect(!refused);
    try testing.expect(allowed);
}

// ── complete_block_on_cs ──────────────────────────────────────────────────────
// Two variants per operation, unlike the Python suite's single mock-backed
// test: with real (not MagicMock) state, restoring a genuinely different
// thread's saved registers really does overwrite cpu.eip, so "eip was
// redirected to retry_eip" and "we really swapped to the next thread" have to
// be checked in different scenarios (self-reload vs switch-away) rather than
// both asserted on the same call.

test "complete_block_on_cs self-reload sets status/waiting_on_cs and redirects eip" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF); // only thread -- next_idx will be -1
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.eip = 0x401002;

    const result = completeBlockOnCs(&sched, &cpu, 0x1234, 0x401000, -1);

    try testing.expect(result);
    try testing.expect(!cpu.halted);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status);
    try testing.expectEqual(@as(u32, 0x1234), sched.threads[0].waiting_on_cs); // NOT cleared (matches Python)
    try testing.expectEqual(@as(u32, 0x401000), cpu.eip);
}

test "complete_block_on_cs switches to resolved next thread" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    sched.threads[1].saved.eip = 0x9F1234;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.eip = 0x401002;

    const result = completeBlockOnCs(&sched, &cpu, 0x1234, 0x401000, 1);

    try testing.expect(result);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
    try testing.expectEqual(@as(u32, 0x9F1234), cpu.eip); // next thread's real state, not retry_eip
    try testing.expectEqual(ThreadStatus.blocked_cs, sched.threads[0].status);
    try testing.expectEqual(@as(u32, 0x1234), sched.threads[0].waiting_on_cs);
    try testing.expectEqual(@as(u32, 0x401000), sched.threads[0].saved.eip); // retry_eip captured by saveCurrent
}

test "complete_block_on_cs refused while reentrant still redirects eip" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.eip = 0x401002;
    enterReentrantCall(&sched);

    const result = completeBlockOnCs(&sched, &cpu, 0x1234, 0x401000, 1);

    try testing.expect(!result);
    try testing.expectEqual(@as(u32, 0x401000), cpu.eip); // redirected even on refusal
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status); // never marked blocked
}

test "complete_block_on_cs does not touch cpu.halted when fatal_halted" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.eip = 0x401002;
    cpu.halted = true;
    cpu.fatal_halted = true;

    const result = completeBlockOnCs(&sched, &cpu, 0x1234, 0x401000, 1);

    try testing.expect(!result);
    try testing.expect(cpu.halted);
    try testing.expectEqual(@as(u32, 0x401002), cpu.eip); // untouched, unlike the reentrancy-refusal case
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status);
}

// ── complete_block_on_handles ─────────────────────────────────────────────────

test "complete_block_on_handles self-reload clears wait state and redirects eip" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.eip = 0x401002;
    const handles = [_]u32{0x700B};

    const result = completeBlockOnHandles(&sched, &cpu, &handles, 0x401000, true, 500, -1);

    try testing.expect(result);
    try testing.expect(!cpu.halted);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status);
    try testing.expectEqual(@as(u8, 0), sched.threads[0].wait_handle_count); // cleared, unlike block_on_cs
    try testing.expectEqual(@as(?u32, null), sched.threads[0].wait_deadline_ms);
    try testing.expectEqual(@as(u32, 0x401000), cpu.eip);
}

test "complete_block_on_handles sets handles/deadline and switches to next" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    const handles = [_]u32{0x700B};

    const result = completeBlockOnHandles(&sched, &cpu, &handles, 0x401000, true, 500, 1);

    try testing.expect(result);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
    try testing.expectEqual(ThreadStatus.blocked_handles, sched.threads[0].status);
    try testing.expectEqual(@as(u8, 1), sched.threads[0].wait_handle_count);
    try testing.expectEqual(@as(u32, 0x700B), sched.threads[0].wait_handles[0]);
    try testing.expectEqual(@as(?u32, 500), sched.threads[0].wait_deadline_ms);
    try testing.expect(!sched.threads[0].wait_timed_out);
}

test "complete_block_on_handles does not touch cpu.halted when fatal_halted" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.halted = true;
    cpu.fatal_halted = true;
    const handles = [_]u32{0x700B};

    const result = completeBlockOnHandles(&sched, &cpu, &handles, 0x401000, false, 0, 1);

    try testing.expect(!result);
    try testing.expect(cpu.halted);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status);
}

// ── complete_sleep_current ────────────────────────────────────────────────────

test "complete_sleep_current self-reload restores eip/eax and clock math" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    sched.virtual_ticks_ms = 100;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = completeSleepCurrent(&sched, &cpu, 0x401010, 0, 50, -1);

    try testing.expect(result);
    try testing.expect(!cpu.halted);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status);
    try testing.expectEqual(@as(u32, 150), sched.threads[0].sleep_until_ms);
    try testing.expectEqual(@as(u32, 0x401010), cpu.eip);
    try testing.expectEqual(@as(u32, 0), cpu.regs[EAX]);
}

test "complete_sleep_current switches to next thread" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = completeSleepCurrent(&sched, &cpu, 0x401010, 0, 50, 1);

    try testing.expect(result);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
    try testing.expectEqual(ThreadStatus.sleeping, sched.threads[0].status);
}

test "complete_sleep_current refused while reentrant still redirects eip and eax" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.eip = 0x401002;
    enterReentrantCall(&sched);

    const result = completeSleepCurrent(&sched, &cpu, 0x401010, 0, 50, 1);

    try testing.expect(!result);
    try testing.expectEqual(@as(u32, 0x401010), cpu.eip);
    try testing.expectEqual(@as(u32, 0), cpu.regs[EAX]);
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
}

test "complete_sleep_current does not clear fatal_halt if no others" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF); // no other thread
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.halted = true;
    cpu.fatal_halted = true;

    const result = completeSleepCurrent(&sched, &cpu, 0x401010, 0, 50, -1);

    try testing.expect(!result);
    try testing.expect(cpu.halted);
}

// ── complete_mark_current_dead ────────────────────────────────────────────────

test "complete_mark_current_dead sets dead status and switches to next" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    sched.current_idx = 1;
    sched.last_scheduled_idx = 1;
    sched.threads[0].has_run = true; // so idx 0 can be resumed (loadThread, not init)
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const halted = completeMarkCurrentDead(&sched, &cpu, 0);

    try testing.expect(!halted);
    try testing.expectEqual(ThreadStatus.dead, sched.threads[1].status);
    try testing.expect(!sched.threads[1].has_run);
    try testing.expectEqual(@as(i32, 0), sched.current_idx);
}

test "complete_mark_current_dead halts when no threads left" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const halted = completeMarkCurrentDead(&sched, &cpu, -1);

    try testing.expect(halted);
    try testing.expect(cpu.halted);
}

test "complete_mark_current_dead still swaps while reentrant" {
    // Deliberately bypasses the reentrancy guard -- a thread dying
    // mid-nested-call must still hand off the CPU.
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    enterReentrantCall(&sched);

    const halted = completeMarkCurrentDead(&sched, &cpu, 1);

    try testing.expect(!halted);
    try testing.expectEqual(ThreadStatus.dead, sched.threads[0].status);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
}

test "complete_mark_current_dead does not touch cpu.halted when already fatal_halted" {
    var sched = twoThreadSched();
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    cpu.halted = true;
    cpu.fatal_halted = true;

    const result = completeMarkCurrentDead(&sched, &cpu, 1);

    try testing.expect(!result);
    try testing.expect(cpu.halted);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[0].status); // unchanged, never marked dead
}

// ── terminate_thread ──────────────────────────────────────────────────────────

test "terminate_thread returns -1 for unknown handle" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    try testing.expectEqual(@as(i8, -1), terminateThread(&sched, &cpu, 0xDEADBEEF, -1));
}

test "terminate_thread terminates a different thread without switching" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = terminateThread(&sched, &cpu, 0xBEF0, -1);

    try testing.expectEqual(@as(i8, 1), result);
    try testing.expectEqual(ThreadStatus.dead, sched.threads[1].status);
    try testing.expect(!sched.threads[1].has_run);
    try testing.expectEqual(@as(i32, 0), sched.current_idx); // never switched
}

test "terminate_thread terminating current thread behaves like mark_current_dead" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = terminateThread(&sched, &cpu, 0xBEEF, 1); // main thread's own handle

    try testing.expectEqual(@as(i8, 0), result);
    try testing.expectEqual(ThreadStatus.dead, sched.threads[0].status);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
}

test "terminate_thread halts when terminating current thread and none left" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);

    const result = terminateThread(&sched, &cpu, 0xBEEF, -1);

    try testing.expectEqual(@as(i8, 0), result);
    try testing.expect(cpu.halted);
}

test "terminate_thread still swaps while reentrant" {
    var sched = twoThreadSched();
    sched.threads[1].has_run = true;
    const mem = try allocTestMem();
    defer testing.allocator.free(mem);
    var cpu = testCpu(mem);
    enterReentrantCall(&sched);

    const result = terminateThread(&sched, &cpu, 0xBEEF, 1);

    try testing.expectEqual(@as(i8, 0), result);
    try testing.expectEqual(ThreadStatus.dead, sched.threads[0].status);
    try testing.expectEqual(@as(i32, 1), sched.current_idx);
}

// ── unblock_cs ─────────────────────────────────────────────────────────────────

test "unblock_cs marks waiting thread ready" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_cs;
    sched.threads[1].waiting_on_cs = 0x1234;

    unblockCs(&sched, 0x1234);

    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
    try testing.expectEqual(@as(u32, 0), sched.threads[1].waiting_on_cs);
}

test "unblock_cs does not affect other cs" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_cs;
    sched.threads[1].waiting_on_cs = 0x5678;

    unblockCs(&sched, 0x1234);

    try testing.expectEqual(ThreadStatus.blocked_cs, sched.threads[1].status);
}

test "unblock_cs unblocks multiple waiters" {
    var sched = SchedulerState{};
    sched.thread_stack_next = 0x1000;
    createMainThread(&sched, 1000, 0xBEEF);
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        const idx = createThread(&sched, 1001 + i, 0xBEF0 + i, 0x9F0000, 0x0, false);
        sched.threads[idx].status = .blocked_cs;
        sched.threads[idx].waiting_on_cs = 0x1234;
    }

    unblockCs(&sched, 0x1234);

    i = 1;
    while (i < sched.thread_count) : (i += 1) {
        try testing.expectEqual(ThreadStatus.ready, sched.threads[i].status);
    }
}

// ── unblock_handle ────────────────────────────────────────────────────────────

test "unblock_handle marks waiting thread ready" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_handles;
    sched.threads[1].wait_handles[0] = 0x700B;
    sched.threads[1].wait_handle_count = 1;

    const n = unblockHandle(&sched, 0x700B);

    try testing.expectEqual(@as(u32, 1), n);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
    try testing.expectEqual(@as(u8, 0), sched.threads[1].wait_handle_count);
}

test "unblock_handle does not affect different handle" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_handles;
    sched.threads[1].wait_handles[0] = 0x700B;
    sched.threads[1].wait_handle_count = 1;

    const n = unblockHandle(&sched, 0x700C);

    try testing.expectEqual(@as(u32, 0), n);
    try testing.expectEqual(ThreadStatus.blocked_handles, sched.threads[1].status);
}

test "unblock_handle does not affect non-blocked thread" {
    var sched = twoThreadSched();
    sched.threads[1].status = .ready;

    const n = unblockHandle(&sched, 0x700B);

    try testing.expectEqual(@as(u32, 0), n);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
}

// ── tick ───────────────────────────────────────────────────────────────────────

test "tick advances clock" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    sched.virtual_ticks_ms = 0;
    tick(&sched, 50);
    try testing.expectEqual(@as(u32, 50), sched.virtual_ticks_ms);
}

test "tick wakes sleeping thread when due" {
    var sched = twoThreadSched();
    sched.threads[1].status = .sleeping;
    sched.threads[1].sleep_until_ms = 100;
    sched.virtual_ticks_ms = 99;
    tick(&sched, 1);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
}

test "tick does not wake sleeping thread before due" {
    var sched = twoThreadSched();
    sched.threads[1].status = .sleeping;
    sched.threads[1].sleep_until_ms = 200;
    sched.virtual_ticks_ms = 0;
    tick(&sched, 100);
    try testing.expectEqual(ThreadStatus.sleeping, sched.threads[1].status);
}

test "tick expires wait deadline" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_handles;
    sched.threads[1].wait_handles[0] = 0x700B;
    sched.threads[1].wait_handle_count = 1;
    sched.threads[1].wait_deadline_ms = 100;
    sched.virtual_ticks_ms = 99;
    tick(&sched, 1);
    try testing.expectEqual(ThreadStatus.ready, sched.threads[1].status);
    try testing.expect(sched.threads[1].wait_timed_out);
    try testing.expectEqual(@as(?u32, null), sched.threads[1].wait_deadline_ms);
    try testing.expectEqual(@as(u8, 0), sched.threads[1].wait_handle_count);
}

test "tick does not expire before deadline" {
    var sched = twoThreadSched();
    sched.threads[1].status = .blocked_handles;
    sched.threads[1].wait_deadline_ms = 500;
    sched.virtual_ticks_ms = 0;
    tick(&sched, 100);
    try testing.expectEqual(ThreadStatus.blocked_handles, sched.threads[1].status);
    try testing.expect(!sched.threads[1].wait_timed_out);
}

test "tick clock wraps at 32-bit" {
    var sched = SchedulerState{};
    createMainThread(&sched, 1000, 0xBEEF);
    sched.virtual_ticks_ms = 0xFFFFFFFF;
    tick(&sched, 1);
    try testing.expectEqual(@as(u32, 0), sched.virtual_ticks_ms);
}

// ── TLS bitset ────────────────────────────────────────────────────────────────

test "tls slot alloc/free/allocated round-trip" {
    var sched = SchedulerState{};
    try testing.expect(!tlsSlotAllocated(&sched, 5));
    tlsAllocSlot(&sched, 5);
    try testing.expect(tlsSlotAllocated(&sched, 5));
    tlsFreeSlot(&sched, 5);
    try testing.expect(!tlsSlotAllocated(&sched, 5));
}

test "tls slot alloc does not affect other slots" {
    var sched = SchedulerState{};
    tlsAllocSlot(&sched, 0);
    tlsAllocSlot(&sched, 63);
    try testing.expect(tlsSlotAllocated(&sched, 0));
    try testing.expect(tlsSlotAllocated(&sched, 63));
    try testing.expect(!tlsSlotAllocated(&sched, 1));
    tlsFreeSlot(&sched, 0);
    try testing.expect(!tlsSlotAllocated(&sched, 0));
    try testing.expect(tlsSlotAllocated(&sched, 63)); // untouched
}

// ── handle-keyed accessors ────────────────────────────────────────────────────

test "get_suspended/set_suspended round-trip via handle" {
    var sched = twoThreadSched();
    try testing.expect(!getSuspended(&sched, 0xBEF0));
    setSuspended(&sched, 0xBEF0, true);
    try testing.expect(getSuspended(&sched, 0xBEF0));
}

test "get_suspended returns false for unknown handle" {
    var sched = twoThreadSched();
    try testing.expect(!getSuspended(&sched, 0xDEADBEEF));
}

test "set_suspended on unknown handle is a no-op, not a crash" {
    var sched = twoThreadSched();
    setSuspended(&sched, 0xDEADBEEF, true); // must not touch any real thread
    try testing.expect(!getSuspended(&sched, 0xBEF0));
}

test "get_completed reflects dead status" {
    var sched = twoThreadSched();
    try testing.expect(!getCompleted(&sched, 0xBEF0));
    sched.threads[1].status = .dead;
    try testing.expect(getCompleted(&sched, 0xBEF0));
}

test "get_wait_timed_out/set_wait_timed_out round-trip via handle" {
    var sched = twoThreadSched();
    try testing.expect(!getWaitTimedOut(&sched, 0xBEF0));
    setWaitTimedOut(&sched, 0xBEF0, true);
    try testing.expect(getWaitTimedOut(&sched, 0xBEF0));
}

test "get_status returns thread status, 0xFF for unknown handle" {
    var sched = twoThreadSched();
    try testing.expectEqual(@as(u8, @intFromEnum(ThreadStatus.ready)), getStatus(&sched, 0xBEF0));
    sched.threads[1].status = .blocked_cs;
    try testing.expectEqual(@as(u8, @intFromEnum(ThreadStatus.blocked_cs)), getStatus(&sched, 0xBEF0));
    try testing.expectEqual(@as(u8, 0xFF), getStatus(&sched, 0xDEADBEEF));
}

test "get_thread_id returns id, -1 for unknown handle" {
    var sched = twoThreadSched();
    try testing.expectEqual(@as(i64, 1001), getThreadId(&sched, 0xBEF0));
    try testing.expectEqual(@as(i64, -1), getThreadId(&sched, 0xDEADBEEF));
}

test "handle_at_idx translates index to handle, -1 out of range" {
    var sched = twoThreadSched();
    try testing.expectEqual(@as(i64, 0xBEEF), handleAtIdx(&sched, 0));
    try testing.expectEqual(@as(i64, 0xBEF0), handleAtIdx(&sched, 1));
    try testing.expectEqual(@as(i64, -1), handleAtIdx(&sched, 2));
}

test "current_handle returns current thread's handle" {
    var sched = twoThreadSched();
    try testing.expectEqual(@as(u32, 0xBEEF), currentHandle(&sched));
    sched.current_idx = 1;
    try testing.expectEqual(@as(u32, 0xBEF0), currentHandle(&sched));
}

test "current_handle returns 0 when no current thread" {
    var sched = SchedulerState{};
    try testing.expectEqual(@as(u32, 0), currentHandle(&sched));
}
