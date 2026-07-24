// SPDX-License-Identifier: LGPL-3.0-or-later
// alloc.zig — Bump-pointer arithmetic for the guest heap allocator.
//
// This is deliberately just the pointer math, not a whole allocator: the
// Python side (CRTState in tew/api/_state.py) still owns the cursor
// (next_heap_alloc) and the size/owner bookkeeping dicts used by
// realloc/HeapSize/HeapFree, the same way ZigMemory left the buffer
// Python-owned in memory.zig. bump_alloc_next computes the new cursor for a
// given (current, size); the caller reads `current` as the allocated
// address before advancing it, exactly like the old simple_alloc() did.
const std = @import("std");

pub export fn bump_alloc_next(current: u32, size: u32) u32 {
    const next: u64 = @as(u64, current) + @as(u64, size) + 15;
    return @intCast(next & ~@as(u64, 15));
}

const testing = std.testing;

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
