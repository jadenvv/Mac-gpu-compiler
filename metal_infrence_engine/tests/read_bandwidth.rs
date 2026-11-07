// Measured GPU read-bandwidth ceiling: the best DRAM read speed a plain streaming
// kernel can reach on this machine. That number, not Apple's spec, is the realistic
// 100% for decode (see mbu.rs).
//
// A kernel streams through a buffer far larger than the system level cache (8 MB on
// M1, up to 96 MB on Ultra parts), so every pass has to come from DRAM.
//
// run with: cargo test --release --test read_bandwidth -- --nocapture

mod common;

use common::peak_bandwidth_gbs;
use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Each thread sums float4s with a grid-sized stride, so one simdgroup's loads are
// contiguous (coalesced). 4 independent accumulators keep several loads in flight.
// One partial sum per threadgroup is written so the reads cannot be optimized away.
kernel void read_sum(
    device const float4* src  [[buffer(0)]],
    device float*        out  [[buffer(1)]],
    constant uint&       n    [[buffer(2)]],   // number of float4s
    uint gid  [[thread_position_in_grid]],
    uint grid [[threads_per_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint i = gid;
    for (; i + 3 * grid < n; i += 4 * grid) {
        a0 += src[i];
        a1 += src[i + grid];
        a2 += src[i + 2 * grid];
        a3 += src[i + 3 * grid];
    }
    for (; i < n; i += grid) a0 += src[i];

    float4 a = (a0 + a1) + (a2 + a3);
    float s = simd_sum(a.x + a.y + a.z + a.w);

    threadgroup float partial[32];
    if (lane == 0) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = simd_sum(partial[lane]);   // TG_SIZE = 1024 -> exactly 32 simdgroups
        if (lane == 0) out[tgid] = t;
    }
}
"#;

const TG_SIZE: u64 = 1024;
const MAX_TGS: u64 = 1024; // up to 1M threads: enough to fill every GPU core many times over
const TRIALS: usize = 5;
const BYTES_PER_TRIAL: u64 = 8 << 30; // ~8 GiB per trial so commit/wait overhead is noise

// sizes swept. Each one is a whole re-dispatch per pass, so they start at 16 MiB:
// below that, per-dispatch launch cost (~10 us) is comparable to the read itself
// and the number measures launch overhead, not memory. A flat line across sizes
// confirms the larger ones are DRAM-bound; >= 256 MiB counts as DRAM.
// (Don't "fix" small sizes by looping passes inside the kernel: threadgroups run
// in waves, so each wave re-reads its own stripe from cache and the result is bogus.)
const SIZES_MIB: &[u64] = &[16, 64, 256, 1024];

#[test]
fn read_bandwidth()
{
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SHADER, &metal::CompileOptions::new())
        .unwrap_or_else(|e| panic!("shader compile failed:\n{e}"));
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("read_sum", None).unwrap())
        .unwrap();
    assert!(pipeline.max_total_threads_per_threadgroup() >= TG_SIZE, "kernel can't run 1024-thread groups");

    let out = device.new_buffer(MAX_TGS * 4, MTLResourceOptions::StorageModeShared);
    let peak = peak_bandwidth_gbs();

    println!();
    println!("chip spec peak : {peak:.2} GB/s");
    println!("{:>10}  {:>10}  {:>10}  {:>7}", "size", "best GB/s", "median", "% spec");

    let mut dram_best = 0.0f64;
    for &mib in SIZES_MIB
    {
        let bytes = mib << 20;
        let n_f4 = (bytes / 16) as u32;
        let src = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        // fill with 1.0 so the result is checkable (also faults every page in before timing)
        unsafe { std::slice::from_raw_parts_mut(src.contents() as *mut f32, (bytes / 4) as usize).fill(1.0) };

        // one thread per float4 for small buffers, so threads aren't mostly idle
        let tgs = (n_f4 as u64).div_ceil(TG_SIZE).min(MAX_TGS);
        let reps = (BYTES_PER_TRIAL / bytes).max(1);
        let run = |reps: u64| -> f64 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&src), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_bytes(2, 4, &n_f4 as *const u32 as *const _);
            for _ in 0..reps
            {
                enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(TG_SIZE, 1, 1));
            }
            enc.end_encoding();
            let start = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            start.elapsed().as_secs_f64()
        };

        run(1); // warm-up: pipeline + first touch on the GPU side

        // check the kernel really read every element
        let partials = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, tgs as usize) };
        let total: f64 = partials.iter().map(|&x| x as f64).sum();
        assert_eq!(total, (bytes / 4) as f64, "{mib} MiB: kernel did not read the whole buffer");

        let mut gbs: Vec<f64> = (0..TRIALS).map(|_| (bytes * reps) as f64 / run(reps) / 1e9).collect();
        gbs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best = gbs[TRIALS - 1];
        let median = gbs[TRIALS / 2];
        println!("{:>6} MiB  {:>10.2}  {:>10.2}  {:>6.1}%", mib, best, median, best / peak * 100.0);

        if mib >= 256 { dram_best = dram_best.max(best); }
    }

    println!();
    println!("measured DRAM read ceiling: {dram_best:.2} GB/s ({:.1}% of spec)", dram_best / peak * 100.0);
    println!("use it as the MBU denominator with PEAK_BW_GBS={dram_best:.1}");

    assert!(dram_best > 0.0 && dram_best <= peak * 1.05,
        "DRAM read {dram_best:.2} GB/s is above spec {peak:.2}: buffer is probably being served from cache");
}
