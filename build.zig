const std = @import("std");

pub fn build(b: *std.Build) void {
    // The installed Zig toolchain (zig-x86-linux-0.15.2) is itself a 32-bit
    // x86 build, so its "native" default resolves to i386 -- even though the
    // host kernel is x86_64. Default to x86_64-linux-gnu explicitly;
    // -Dtarget still overrides this if ever needed.
    const target = b.standardTargetOptions(.{
        .default_target = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu },
    });
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/cpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_mod.link_libc = true;

    const lib = b.addLibrary(.{
        .name = "cpu",
        .root_module = lib_mod,
        .linkage = .dynamic,
        .version = .{ .major = 0, .minor = 2, .patch = 0 },
    });
    b.installArtifact(lib);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/cpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.link_libc = true;

    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    // `zig build test` alone doesn't touch zig-out/lib/libcpu.so -- only the
    // default install step does. Bitten by this twice (a fix compiled and
    // its tests passed, but tew/hardware/cpu_zig.py's ctypes binding kept
    // loading the stale pre-fix .so, since nothing had rebuilt it): make
    // `test` also depend on install, so the real artifact is always current
    // whenever tests are run, not just on a separate plain `zig build`.
    test_step.dependOn(b.getInstallStep());
}
