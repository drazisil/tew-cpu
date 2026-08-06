# cpu Changelog

Entries are newest-first.

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
