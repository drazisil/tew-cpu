// SPDX-License-Identifier: LGPL-3.0-or-later
// memory.zig — Bounds-checked flat memory access over a caller-owned buffer.
//
// This mirrors core.zig's CpuState.memory/memory_size access pattern (a
// borrowed [*]u8 pointer + size, owned by the Python-side bytearray) but is
// NOT CpuState-bound: it exists so Python code that isn't stepping a CPU
// (e.g. the loader poking imports into place before execution starts) can
// still get bounds-checked reads/writes against the same underlying buffer.
// Failure is reported via a `bool` return instead of CpuState.faulted/halted,
// since there is no CPU here to fault.
//
// The raw bounds-check/byte-access arithmetic itself lives in primitives.zig,
// shared with core.zig's CpuState-bound memRead8/memWrite8 -- this file only
// owns the bool-return C-ABI shape around those primitives.
const std = @import("std");
const primitives = @import("primitives.zig");

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

const testing = std.testing;

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
