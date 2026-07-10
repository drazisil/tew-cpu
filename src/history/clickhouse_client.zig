//! Minimal ClickHouse HTTP-interface client -- POC only, no retry/
//! connection pooling. Talks to ClickHouse's HTTP interface (default port
//! 8123): POST /?query=<INSERT ... FORMAT JSONEachRow>, body = ndjson
//! rows.

const std = @import("std");

pub const ClickHouseClient = struct {
    allocator: std.mem.Allocator,
    base_url: []const u8, // e.g. "http://localhost:8123"
    user: []const u8 = "default",
    password: []const u8 = "",

    pub const InsertError = error{ClickHouseInsertFailed} || std.http.Client.FetchError || std.mem.Allocator.Error;

    /// POSTs `body` (ndjson, one JSON object per line) as an INSERT into
    /// `table` FORMAT JSONEachRow. Auth via X-ClickHouse-User/Key headers
    /// (avoids needing to base64-encode Basic Auth credentials).
    pub fn insertNdjson(self: *ClickHouseClient, table: []const u8, body: []const u8) InsertError!void {
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        // Query text is fixed/controlled (never contains user data), so
        // hand-building the %20-encoded form directly is simpler and safer
        // than a general-purpose URL encoder for this POC.
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/?query=INSERT%20INTO%20{s}%20FORMAT%20JSONEachRow",
            .{ self.base_url, table },
        );
        defer self.allocator.free(url);

        const result = try client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .extra_headers = &.{
                .{ .name = "X-ClickHouse-User", .value = self.user },
                .{ .name = "X-ClickHouse-Key", .value = self.password },
            },
        });
        if (result.status != .ok) return InsertError.ClickHouseInsertFailed;
    }
};
