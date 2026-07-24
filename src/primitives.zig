// SPDX-License-Identifier: LGPL-3.0-or-later
// primitives.zig — single shared "is this address in bounds, read/write the
// byte(s)" arithmetic for a raw (ptr, size) buffer. No CpuState, no
// bool/fault signaling, no Python-ABI shape. Nothing here is `export`.
const std = @import("std");

// Mirrors core.zig's ORIGINAL single-byte check (truncates memory_size to u32).
pub inline fn inBounds1(size: usize, addr: u32) bool {
    return addr < @as(u32, @truncate(size));
}

// Mirrors memory.zig's ORIGINAL combined-width check (u64 math, no truncation).
// Only used by mem_read16/32/write16/32 -- core.zig's CpuState path never
// calls this; the two formulas are kept deliberately distinct (a 16-bit
// access straddling the end of memory behaves differently under each: the
// per-byte-composed CpuState path retains the first byte's real value
// before faulting on the second, while a combined-width pre-check would
// reject both bytes up front -- unifying them would be an observable
// behavior change, not a pure refactor).
pub inline fn inBoundsWidth(size: usize, addr: u32, width: u32) bool {
    const end = @as(u64, addr) + @as(u64, width);
    return end <= size;
}

pub inline fn readByte(ptr: [*]const u8, addr: u32) u8 {
    return ptr[addr];
}
pub inline fn writeByte(ptr: [*]u8, addr: u32, v: u8) void {
    ptr[addr] = v;
}

const testing = std.testing;

test "inBounds1 matches core.zig's original single-byte formula" {
    try testing.expect(inBounds1(4, 0));
    try testing.expect(inBounds1(4, 3));
    try testing.expect(!inBounds1(4, 4));
}

test "inBoundsWidth matches memory.zig's original combined-width formula" {
    try testing.expect(inBoundsWidth(4, 0, 4));
    try testing.expect(!inBoundsWidth(4, 1, 4));
    try testing.expect(!inBoundsWidth(4, 0, 5));
}

test "readByte/writeByte round-trip" {
    var buf = [_]u8{0} ** 4;
    writeByte(&buf, 1, 0xAB);
    try testing.expectEqual(@as(u8, 0xAB), readByte(&buf, 1));
}
