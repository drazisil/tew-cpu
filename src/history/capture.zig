// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bitmap-indexed execution history capture layer. Buffers step-stamped
//! memory/register change events emitted by the write_hook/step_hook on
//! CpuState (see ../core.zig, ../cpu.zig) and flushes them to ClickHouse as
//! ndjson over HTTP.
//!
//! Ported from pe-walker's vendor/tew-cpu fork of this same core -- see
//! that repo's src/history/capture.zig and PROVENANCE.md. Unlike the
//! vendored copy (a separate package importing this core as `tew_cpu`),
//! this lives directly inside the canonical core, so it imports core.zig
//! by relative path instead of through a named module.
//!
//! Deliberately Option B (defer real flushing to deinit/explicit flush)
//! for this port, not a background-thread design -- see the original
//! plan's Step 4. The maybeFlush/flush call structure is intentionally the
//! same shape a future upgrade would reuse: only flush_threshold's value
//! and flush()'s body would need to change.

const std = @import("std");
const cpu_core = @import("../core.zig");
const clickhouse_client = @import("clickhouse_client.zig");

pub const ClickHouseClient = clickhouse_client.ClickHouseClient;

/// Matches the EAX=0,ECX=1,EDX=2,EBX=3,ESP=4,EBP=5,ESI=6,EDI=7 register-index
/// constants in ../core.zig exactly -- see the "RegKey alignment test" below,
/// which guards this against silent drift.
pub const RegKey = enum(u32) {
    eax = 0,
    ecx = 1,
    edx = 2,
    ebx = 3,
    esp = 4,
    ebp = 5,
    esi = 6,
    edi = 7,
    eip = 8,
    eflags = 9,
};

pub const MemEvent = struct { step: u64, addr: u32, old: u8, new: u8 };
pub const RegEvent = struct { step: u64, reg: RegKey, old: u32, new: u32 };

pub const Sink = union(enum) {
    /// Discard -- still exercises buffering/threshold logic. Default.
    none,
    /// Real flush over HTTP.
    clickhouse: ClickHouseClient,
    /// Test-only: capture the ndjson body that would have been sent,
    /// instead of sending it.
    collect: *std.ArrayList(u8),
};

/// Rows per outbound POST when actually flushing to ClickHouse -- keeps a
/// large deinit-time flush from becoming one giant request. Independent of
/// Capture.flush_threshold, which governs *when* a flush happens at all.
const post_chunk_rows: usize = 50_000;

pub const Capture = struct {
    allocator: std.mem.Allocator,
    /// Adopted from CpuState.run_id in armFor, never self-generated --
    /// run_id is minted once, by cpu_create, at the moment the CpuState it
    /// belongs to is born.
    run_id: u64 = 0,
    mem_events: std.ArrayList(MemEvent) = .empty,
    reg_events: std.ArrayList(RegEvent) = .empty,
    /// Set high enough that an ordinary POC/validation run never crosses it
    /// mid-run -- the real flush happens once, from deinit, internally
    /// chunked (post_chunk_rows).
    flush_threshold: usize = 1_000_000,
    sink: Sink = .none,
    // Diff baseline for recordStep. Lives here, not on CpuState -- the
    // step_hook reports raw post-instruction state each call; this is what
    // turns consecutive reports into "what changed since last time."
    last_regs: [8]u32 = .{0} ** 8,
    last_eip: u32 = 0,
    last_eflags: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, sink: Sink) Capture {
        return .{ .allocator = allocator, .sink = sink };
    }

    /// Adopts this CpuState's own run_id and seeds the register-diff
    /// baseline from its current values. Must be called BEFORE wiring
    /// hooks, so the first instruction doesn't spuriously diff against
    /// zero. Also the re-sync point after any out-of-band cpu_set_eip/
    /// cpu_set_reg call that bypasses both hooks -- without re-arming
    /// there, the next real instruction would spuriously diff against a
    /// stale baseline and misattribute the out-of-band patch to itself.
    pub fn armFor(self: *Capture, state: *const cpu_core.CpuState) void {
        self.run_id = state.run_id;
        self.last_regs = state.regs;
        self.last_eip = state.eip;
        self.last_eflags = state.eflags;
    }

    pub fn recordMemWrite(self: *Capture, step: u64, addr: u32, old: u8, new: u8) void {
        if (old == new) return; // defense in depth -- the write_hook already filters this
        self.mem_events.append(self.allocator, .{ .step = step, .addr = addr, .old = old, .new = new }) catch return;
        self.maybeFlush();
    }

    pub fn recordStep(self: *Capture, step: u64, eip: u32, regs: [8]u32, eflags: u32) void {
        inline for (0..8) |i| {
            if (regs[i] != self.last_regs[i]) {
                self.reg_events.append(self.allocator, .{
                    .step = step,
                    .reg = @enumFromInt(i),
                    .old = self.last_regs[i],
                    .new = regs[i],
                }) catch {};
            }
        }
        if (eip != self.last_eip) {
            self.reg_events.append(self.allocator, .{ .step = step, .reg = .eip, .old = self.last_eip, .new = eip }) catch {};
        }
        if (eflags != self.last_eflags) {
            self.reg_events.append(self.allocator, .{ .step = step, .reg = .eflags, .old = self.last_eflags, .new = eflags }) catch {};
        }
        self.last_regs = regs;
        self.last_eip = eip;
        self.last_eflags = eflags;
        self.maybeFlush();
    }

    fn maybeFlush(self: *Capture) void {
        if (self.mem_events.items.len + self.reg_events.items.len >= self.flush_threshold) self.flush();
    }

    /// Builds ndjson and dispatches per sink, chunked at post_chunk_rows
    /// for .clickhouse. MUST swallow all errors -- this is reachable
    /// synchronously from onCpuWrite/onCpuStep below, which are
    /// callconv(.c) void-returning hooks invoked from inside core.zig's
    /// memWrite8/cpu.zig's cpu_run; there is no way to bubble a Zig error
    /// out through that boundary, and a ClickHouse hiccup must never crash
    /// or corrupt emulation -- it should just drop that batch.
    pub fn flush(self: *Capture) void {
        defer {
            self.mem_events.clearRetainingCapacity();
            self.reg_events.clearRetainingCapacity();
        }
        if (self.mem_events.items.len == 0 and self.reg_events.items.len == 0) return;

        switch (self.sink) {
            .none => {},
            .collect => |out| self.appendNdjson(out) catch {},
            .clickhouse => |*client| self.flushToClickHouse(client),
        }
    }

    fn appendNdjson(self: *Capture, out: *std.ArrayList(u8)) !void {
        for (self.mem_events.items) |e| {
            try out.print(
                self.allocator,
                "{{\"run_id\":{d},\"step\":{d},\"kind\":\"mem\",\"key\":{d},\"old_value\":{d},\"new_value\":{d}}}\n",
                .{ self.run_id, e.step, e.addr, e.old, e.new },
            );
        }
        for (self.reg_events.items) |e| {
            try out.print(
                self.allocator,
                "{{\"run_id\":{d},\"step\":{d},\"kind\":\"reg\",\"key\":{d},\"old_value\":{d},\"new_value\":{d}}}\n",
                .{ self.run_id, e.step, @intFromEnum(e.reg), e.old, e.new },
            );
        }
    }

    fn flushToClickHouse(self: *Capture, client: *ClickHouseClient) void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.allocator);
        var count: usize = 0;

        // Takes the row count explicitly rather than inferring it from
        // b.items.len (the ndjson *byte* length, not a row count) -- see
        // pe-walker's commit abdf0b5, which confirmed live that using
        // items.len produced a wildly misleading "dropping batch of 1003
        // rows" for what was actually ~10 real events (~1003 bytes of
        // ndjson text).
        const sendChunk = struct {
            fn call(c: *ClickHouseClient, b: *std.ArrayList(u8), rows: usize) void {
                c.insertNdjson("history_events", b.items) catch |err| {
                    std.debug.print("history capture: ClickHouse insert failed, dropping batch of {d} rows: {}\n", .{ rows, err });
                };
                b.clearRetainingCapacity();
            }
        }.call;

        for (self.mem_events.items) |e| {
            buf.print(
                self.allocator,
                "{{\"run_id\":{d},\"step\":{d},\"kind\":\"mem\",\"key\":{d},\"old_value\":{d},\"new_value\":{d}}}\n",
                .{ self.run_id, e.step, e.addr, e.old, e.new },
            ) catch return;
            count += 1;
            if (count >= post_chunk_rows) {
                sendChunk(client, &buf, count);
                count = 0;
            }
        }
        for (self.reg_events.items) |e| {
            buf.print(
                self.allocator,
                "{{\"run_id\":{d},\"step\":{d},\"kind\":\"reg\",\"key\":{d},\"old_value\":{d},\"new_value\":{d}}}\n",
                .{ self.run_id, e.step, @intFromEnum(e.reg), e.old, e.new },
            ) catch return;
            count += 1;
            if (count >= post_chunk_rows) {
                sendChunk(client, &buf, count);
                count = 0;
            }
        }
        if (buf.items.len > 0) sendChunk(client, &buf, count);
    }

    pub fn deinit(self: *Capture) void {
        self.flush();
        self.mem_events.deinit(self.allocator);
        self.reg_events.deinit(self.allocator);
    }
};

/// Trampoline registered as CpuState.write_hook; ctx is always a *Capture.
/// run_id is already on the Capture (adopted via armFor) so it's ignored
/// here rather than threaded through again -- kept as a parameter for
/// symmetry with the hook shape itself.
pub fn onCpuWrite(ctx: ?*anyopaque, run_id: u64, step: u64, addr: u32, old: u8, new: u8) callconv(.c) void {
    _ = run_id;
    const cap: *Capture = @ptrCast(@alignCast(ctx orelse return));
    cap.recordMemWrite(step, addr, old, new);
}

/// Trampoline registered as CpuState.step_hook; same ctx as onCpuWrite.
pub fn onCpuStep(ctx: ?*anyopaque, run_id: u64, step: u64, eip: u32, regs: *const [8]u32, eflags: u32) callconv(.c) void {
    _ = run_id;
    const cap: *Capture = @ptrCast(@alignCast(ctx orelse return));
    cap.recordStep(step, eip, regs.*, eflags);
}

// ─── Tests ──────────────────────────────────────────────────────────────────
const testing = std.testing;

test "recordMemWrite drops a no-op (old==new) write" {
    var cap = Capture.init(testing.allocator, .none);
    defer cap.deinit();

    cap.recordMemWrite(5, 0x1000, 0x42, 0x42);
    try testing.expectEqual(@as(usize, 0), cap.mem_events.items.len);

    cap.recordMemWrite(5, 0x1000, 0x42, 0x43);
    try testing.expectEqual(@as(usize, 1), cap.mem_events.items.len);
}

test "recordStep emits one RegEvent per genuinely-changed field only" {
    var cap = Capture.init(testing.allocator, .none);
    defer cap.deinit();

    // last_regs/last_eip/last_eflags default to zero (armFor not called);
    // only eax differs from that baseline.
    var regs: [8]u32 = .{0} ** 8;
    regs[0] = 42;
    cap.recordStep(0, 0, regs, 0);

    try testing.expectEqual(@as(usize, 1), cap.reg_events.items.len);
    try testing.expectEqual(RegKey.eax, cap.reg_events.items[0].reg);
    try testing.expectEqual(@as(u32, 0), cap.reg_events.items[0].old);
    try testing.expectEqual(@as(u32, 42), cap.reg_events.items[0].new);
}

test "maybeFlush triggers exactly at flush_threshold, not before" {
    var cap = Capture.init(testing.allocator, .none);
    cap.flush_threshold = 3;
    defer cap.deinit();

    cap.recordMemWrite(0, 1, 0, 1);
    cap.recordMemWrite(1, 1, 1, 2);
    try testing.expectEqual(@as(usize, 2), cap.mem_events.items.len); // not yet flushed

    cap.recordMemWrite(2, 1, 2, 3);
    try testing.expectEqual(@as(usize, 0), cap.mem_events.items.len); // flushed (cleared) at threshold
}

test "flush with .collect sink appends valid ndjson lines" {
    var collected: std.ArrayList(u8) = .empty;
    defer collected.deinit(testing.allocator);

    var cap = Capture.init(testing.allocator, .{ .collect = &collected });
    defer cap.deinit();

    cap.recordMemWrite(7, 0x2000, 0x10, 0x20);
    cap.flush();

    try testing.expect(std.mem.indexOf(u8, collected.items, "\"step\":7") != null);
    try testing.expect(std.mem.indexOf(u8, collected.items, "\"kind\":\"mem\"") != null);
    try testing.expect(std.mem.indexOf(u8, collected.items, "\"key\":8192") != null); // 0x2000
    try testing.expectEqual(@as(usize, 0), cap.mem_events.items.len); // cleared after flush
}

test "RegKey alignment: each regs[] index maps to the correct RegKey" {
    var cap = Capture.init(testing.allocator, .none);
    defer cap.deinit();

    // Distinct nonzero sentinel per index so a misaligned RegKey<->index
    // mapping would show up as a wrong `.reg`/`.new` pairing, not just a
    // missing event -- must keep matching EAX=0,ECX=1,EDX=2,EBX=3,ESP=4,
    // EBP=5,ESI=6,EDI=7 in ../core.zig exactly.
    var regs: [8]u32 = .{0} ** 8;
    inline for (0..8) |i| regs[i] = 100 + i;
    cap.recordStep(0, 0, regs, 0);

    try testing.expectEqual(@as(usize, 8), cap.reg_events.items.len);
    const expected = [_]RegKey{ .eax, .ecx, .edx, .ebx, .esp, .ebp, .esi, .edi };
    for (cap.reg_events.items, 0..) |e, i| {
        try testing.expectEqual(expected[i], e.reg);
        try testing.expectEqual(@as(u32, @intCast(100 + i)), e.new);
    }
}

// Ported from pe-walker's equivalent test, which drove this through its
// own `Emulator` wrapper (Emulator.init/.step/.setReg) -- that wrapper
// doesn't exist in this core, so this drives cpu_run/CpuState directly
// (both now `pub`, see cpu.zig).
test "step-number consistency: a single instruction's MemEvents and RegEvent share the same step" {
    const cpu_mod = @import("../cpu.zig");

    var mem = [_]u8{0x50} ++ [_]u8{0} ** 63; // push eax
    var s = cpu_core.CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[@intFromEnum(RegKey.esp)] = 32;
    s.regs[@intFromEnum(RegKey.eax)] = 0xAABBCCDD; // nonzero so the write is a genuine change

    var cap = Capture.init(testing.allocator, .none);
    defer cap.deinit();
    cap.armFor(&s);
    s.history_ctx = &cap;
    s.write_hook = onCpuWrite;
    s.step_hook = onCpuStep;

    _ = cpu_mod.cpu_run(&s, 1);

    try testing.expectEqual(@as(usize, 4), cap.mem_events.items.len); // 4 bytes of EAX pushed
    // esp changes (push) AND eip changes (every instruction advances eip
    // during fetch) -- both are genuine, expected changes for this step.
    try testing.expectEqual(@as(usize, 2), cap.reg_events.items.len);
    for (cap.mem_events.items) |e| try testing.expectEqual(@as(u64, 0), e.step);
    try testing.expectEqual(@as(u64, 0), cap.reg_events.items[0].step);
    try testing.expectEqual(@as(u64, 0), cap.reg_events.items[1].step);
}

// Ported from pe-walker's equivalent end-to-end test. That version used a
// shared `test_support.code` byte sequence; hand-encoded here instead
// since this core has no such helper: push ebp; mov ebp,esp; mov eax,0x2a;
// ret.
test "end-to-end: real hooks capture push/mov/mov/ret exactly" {
    const cpu_mod = @import("../cpu.zig");

    const code = [_]u8{ 0x55, 0x89, 0xE5, 0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3 };
    var mem = code ++ [_]u8{0} ** (512 - code.len);
    var s = cpu_core.CpuState{ .memory = &mem, .memory_size = mem.len };
    s.regs[@intFromEnum(RegKey.esp)] = 256;
    // Non-zero so `push ebp`'s memory write is a genuine value change --
    // the buffer is zero-initialized, so pushing 0 onto already-zero memory
    // would be a filtered no-op, not a real write to assert on.
    s.regs[@intFromEnum(RegKey.ebp)] = 0x11223344;

    var cap = Capture.init(testing.allocator, .none);
    defer cap.deinit();
    cap.armFor(&s);
    s.history_ctx = &cap;
    s.write_hook = onCpuWrite;
    s.step_hook = onCpuStep;

    _ = cpu_mod.cpu_run(&s, 4);

    // Step 0 (push ebp): ESP 256->252, 4 bytes of 0x11223344 written
    // little-endian at addresses 252..255 (push32 decrements ESP first,
    // then writes at the new ESP).
    try testing.expectEqual(@as(usize, 4), cap.mem_events.items.len);
    const expected_bytes = [_]u8{ 0x44, 0x33, 0x22, 0x11 };
    for (cap.mem_events.items, 0..) |e, i| {
        try testing.expectEqual(@as(u64, 0), e.step);
        try testing.expectEqual(@as(u32, 252 + @as(u32, @intCast(i))), e.addr);
        try testing.expectEqual(@as(u8, 0), e.old);
        try testing.expectEqual(expected_bytes[i], e.new);
    }

    // Every instruction advances eip during fetch, so each of the 4 steps
    // produces an eip RegEvent in addition to whatever GPR changed --
    // recordStep checks regs[0..8] (in RegKey order) before eip, so eip
    // always lands right after that step's GPR change, not before it.
    //   step 0 (push ebp):    esp 256->252, eip 0->1
    //   step 1 (mov ebp,esp): ebp ->252,    eip 1->3
    //   step 2 (mov eax,imm): eax ->0x2a,   eip 3->8
    //   step 3 (ret):         esp 252->256, eip 8->0x11223344 (popped)
    try testing.expectEqual(@as(usize, 8), cap.reg_events.items.len);

    const Expected = struct { reg: RegKey, step: u64, new: u32 };
    const expected = [_]Expected{
        .{ .reg = .esp, .step = 0, .new = 252 },
        .{ .reg = .eip, .step = 0, .new = 1 },
        .{ .reg = .ebp, .step = 1, .new = 252 },
        .{ .reg = .eip, .step = 1, .new = 3 },
        .{ .reg = .eax, .step = 2, .new = 0x2a },
        .{ .reg = .eip, .step = 2, .new = 8 },
        .{ .reg = .esp, .step = 3, .new = 256 },
        .{ .reg = .eip, .step = 3, .new = 0x11223344 }, // popped return addr
    };
    for (cap.reg_events.items, expected) |actual, exp| {
        try testing.expectEqual(exp.reg, actual.reg);
        try testing.expectEqual(exp.step, actual.step);
        try testing.expectEqual(exp.new, actual.new);
    }
}
