# cpu Changelog

Entries are newest-first.

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
