# cpu Changelog

Entries are newest-first.

---

## 2026-10-03 — 0.3.3: 16/32-bit memory accesses take one load/store

`memRead16/32` and `memWrite16/32` did a separate byte access per byte, each
repeating the fault, watchpoint and write-hook checks. When every byte is in
bounds (no u32 wrap, no null-page guard hit, and for writes no hook or
watchpoint inside the access) they now do a single unaligned load or store.
Every other case takes the original byte path, so faults, partial reads at the
end of memory, watchpoints and write hooks behave exactly as before (new tests
cover each). Measured on the MCity test-drive run: ~4% faster (first click
trigger 62.7s -> 60.4s; test-drive load 70s -> 67s; single runs, so treat as
approximate). The decode/dispatch loop is untouched.

## 2026-10-02 — 0.3.2: x87 operand-order and flag fixes; silent no-ops now fault

Found while chasing why MCity's RunEngSim writes no output (audit of the x87
handlers; this is not yet known to be the cause).

- FSTP m80 wrote a 64-bit double plus a zero exponent word; FLD m80 read the
  real extended layout, so any `fstp tbyte`/`fld tbyte` round trip (CRT
  temporaries) came back as ~0. Both now use the f80 bit layout.
- DC E0..FF register forms (FSUBR/FSUB/FDIVR/FDIV ST(i),ST) had their operands
  swapped relative to the SDM and to the DE popping forms.
- FCMOVNB/NE/NBE/NU (DB C0..DF) were not implemented (silent no-op); FCMOVU
  (DA D8..DF) copied unconditionally instead of testing PF.
- FCOMI/FUCOMI/FCOMIP/FUCOMIP never set PF; unordered now sets ZF=PF=CF=1.
- New opt-in `cpu_trace_start`/`cpu_trace_stop`: records each executed
  instruction's EIP into a host-owned ring buffer (off unless started).
- FLDENV, FNSTENV, FRSTOR, FNSAVE, FBLD and FBSTP were silent NOPs; they now
  fault as unknown opcodes (fail loudly) until implemented.

---

## 2026-09-30 — 0.3.1: x87 FPTAN/FPATAN/FXTRACT/FYL2XP1 implemented; FXAM, FPREM/FPREM1, FRNDINT, trig C2 fixed

Found through MCity's HOME avatar never drawing: its projection came out
collapsed and ~500px off-screen, and the CRT's math dispatcher (which
classifies arguments with FXAM) took its error path on every avatar draw.

- FPTAN, FPATAN, FXTRACT and FYL2XP1 were silent no-ops. FPTAN also never
  pushed its 1.0, leaving the FPU stack off by one for everything after it.
  FPATAN/FYL2XP1 use libc's `atan2l`/`log1pl` (Zig's std has no f80 versions).
- FXAM cleared C3/C2/C0 for every value, so everything classified as
  "unsupported format". It now reports empty/NaN/infinity/zero/denormal/
  normal, with C1 = sign.
- FPREM/FPREM1 never touched C2 or the quotient bits, and FPREM1 truncated
  like FPREM. Both now complete the reduction (C2 = 0), set C0/C3/C1 =
  Q2/Q1/Q0, FPREM1 rounds the quotient to nearest-even, and a zero divisor
  or infinite dividend gives IE + NaN.
- FSIN/FCOS/FSINCOS/FPTAN clear C2 in range and, for |x| >= 2^63, set C2
  and leave the operand alone (stale C2 used to leak through).
- FRNDINT honors the control word's rounding mode (was `@round`,
  half-away-from-zero).
- Reserved D9 register encodings (D9 D1-D7, E2, E3, E6, E7, EF) now fault as
  unknown opcodes instead of doing nothing.

## 2026-09-29 — 0.3.0: critical sections leave the scheduler (breaking ABI)

Version 0.2.0 -> 0.3.0, covering everything since 0.2.0 (this entry and the
two below). Breaking for hosts:

- Removed `scheduler_complete_block_on_cs` and `scheduler_unblock_cs`, and
  the scheduler's `blocked_cs` state with them. Hosts now implement critical
  sections the way XP does (the lock state lives in the guest's
  `RTL_CRITICAL_SECTION`, contention waits on its LockSemaphore event via
  `scheduler_complete_block_on_handles`/`scheduler_unblock_handle`), so the
  scheduler no longer reads the guest struct's OwningThread to decide wakes.
- `ThreadStatus` loses `blocked_cs`: `ready=0, blocked_handles=1,
  sleeping=2, dead=3` (`blocked_handles` and later shift down by one) --
  affects `scheduler_get_status`.
- `scheduler_pick_next_ready(s)` no longer takes the CPU; the owner read
  was its only use of it.

Also: the scheduler writes the incoming thread's id to TEB+0x24
(`ClientId.UniqueThread`, i.e. fs:[0x24]) on every switch, alongside the
existing TLS/LastError/ExceptionList swap. All threads share one TEB, and
it used to keep whatever the host wrote once (tew wrote 1), so guest code
reading fs:[0x24] disagreed with GetCurrentThreadId.

---

## 2026-09-27 — Add `scheduler_current_thread_id`

Returns the current thread's id in one call (-1 when no thread is current).
Hosts needed three calls for it before (`scheduler_current_idx`,
`scheduler_current_handle`, then `scheduler_get_thread_id`, which searches
threads by handle), on paths as hot as every EnterCriticalSection.

---

## 2026-09-26 — Add `cpu_stdcall_cleanup`

New export for hosts that implement API calls as `INT n; RET` trampolines:
moves the return address over a stdcall callee's args (`ret=[esp];
esp+=n; [esp]=ret`) in one call, so the host no longer needs six separate
register/memory crossings per API call. Uses raw bounds-checked buffer
access (like `mem_read32`/`mem_write32`), so it does not trip watchpoints,
the write-history hook, or the null-page guard; returns false without
touching anything if either stack slot is out of bounds. ESP is left
unchanged while fatal-halted, matching `cpu_set_reg`.

---

## 2026-08-06 — Extracted to its own repo (drazisil/tew-cpu); ported pe-walker's
5 unported instruction-coverage fixes

This history was extracted from `tew`'s `cpu/` subdirectory (`git filter-repo
--path cpu/ --path-rename cpu/:`) into its own standalone repository, so both
`tew` and `pe-walker` can consume a single real dependency instead of one
maintaining it and the other hand-copying a snapshot that immediately starts
drifting (pe-walker's own `vendor/tew-cpu/PROVENANCE.md` documented exactly
this: "there is no automatic sync", copied 2026-07-04, already missing over a
month of fixes by this date).

Ported pe-walker's own 5 unported fixes back the other direction, so this
starting point is a true merge of both sides rather than just "tew's state,
plus a promise to backport later":

- **CMPXCHG rm32,r32** (`0x0F 0xB1`) — was entirely unimplemented (fell to
  the generic fault). Confirmed live there against real Windows XP
  `kernel32.dll`: `lock cmpxchg dword ptr [edx], ecx`, a classic
  `InterlockedCompareExchange` shape.
- **SHRD/SHLD** (`0x0F 0xAC`/`0xAD`/`0xA4`/`0xA5`) — same, unimplemented.
  Confirmed live there against real `ntdll.dll` heap-manager code
  (`shrd eax, edx, 0x18`, extracting a shifted field from a 64-bit value
  held across two 32-bit registers).
- **MOV rm16,Sreg** (`0x8C`) — same, unimplemented; needed new `seg_cs`/
  `seg_ds`/`seg_es`/`seg_fs`/`seg_gs`/`seg_ss` fields on `CpuState` (distinct
  from the pre-existing `fs_base`/`gs_base`, which are only the linear base
  address segment overrides resolve against, not the selector itself).
  Confirmed live there against real `ntdll!RtlCaptureContext` populating a
  `CONTEXT` structure.
- **POP rm32, memory-operand form** (`0x8F`) — only the register form
  (`0x58`-`0x5F`) previously existed. Confirmed live there against the
  standard SEH epilogue, `pop dword ptr fs:[0]`.
- **`IntHandlerFn` visibility** — was a non-`pub` alias in the package root
  (`kernel.zig`, this repo's `root_source_file`), so an external Zig
  consumer `@import`-ing this as a module (rather than linking the compiled
  `libcpu.so`, i.e. exactly how pe-walker consumes it) couldn't reference
  the type to install its own interrupt handler.

All 5 adapted to this repo's current API, which has diverged from pe-walker's
2026-07-04 snapshot in the other direction since then: width-aware flags
computation (`updateFlagsArithW`/`updateFlagsLogicW` taking an explicit
`Width`, not pe-walker's older fixed-32-bit-only `updateFlagsArith`/
`updateFlagsLogic`) and the `kernel.zig`/`engine.zig`/`core.zig`/
`primitives.zig` module split (pe-walker's snapshot predates it, still a
single `cpu.zig` monolith). Confirmed the core memory bounds-check/read/
write logic itself has *not* drifted between the two — same formulas, same
watchpoint handling — while checking; the `write_hook`/`step_hook`
execution-history-capture pair was separately already backported into `tew`
before this extraction. pe-walker's own `read_hook`/`write_miss_hook`
(backing its synthetic KUSER_SHARED_DATA/TEB/PEB memory pages) intentionally
stay pe-walker-specific per its own `PROVENANCE.md` -- not general fixes,
not ported here.

8 new Zig-level unit tests (this repo's first-ever tests for any of these
five). 69/69 tests pass.

---

## 2026-07-04 — Relicense as LGPL-3.0, modernize build for Zig 0.15.2, add public-ABI tests

- Added `LICENSE` (LGPL-3.0-or-later, this subdirectory only -- the rest of
  the `tew` repo remains GPL-3.0). SPDX headers added to all `src/*.zig`
  files.
- `build.zig` rewritten for Zig 0.15.2's Module-based API
  (`b.createModule()` + `b.addLibrary(.{.linkage = .dynamic, ...})` instead
  of the old 0.14 `b.addSharedLibrary(.{.root_source_file = ...})` form).
  Also forces the build target to `x86_64-linux-gnu` explicitly, since the
  installed Zig toolchain is itself a 32-bit build whose "native" default
  otherwise resolves to i386 on this x86_64 host.
  `minimum_zig_version` bumped 0.14.0 -> 0.15.2 to match.
- Removed dead `zig init` boilerplate (`src/main.zig`, `src/root.zig`),
  unused since the actual library root has always been `src/cpu.zig`.
- Version reconciled: `build.zig`'s library `.version` and
  `build.zig.zon`'s `.version` were out of sync (0.1.0 vs a stale 0.0.0
  placeholder); both now read 0.2.0.
- `callconv(.C)` -> `callconv(.c)` in `core.zig`'s `IntHandlerFn`/
  `LogpointFn` typedefs, matching modern Zig style (both compiled fine
  either way -- cosmetic only).
- Added 3 new tests exercising the public `cpu_*` C ABI directly
  (`cpu_create`/`cpu_run`/`cpu_get_reg`/`cpu_destroy`, breakpoint-hit
  semantics, out-of-bounds fault handling) -- the existing 5 tests only
  ever called the internal `cpuStep` function, so nothing previously
  proved the exported C boundary itself works standalone (i.e. usable by
  a consumer other than this project's own Python harness). This was
  motivated by pe-walker (a separate Zig/GTK4 project) vendoring `cpu`'s
  source directly as a real execution engine, with no Python involved at
  all.
