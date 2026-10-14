use anyhow::{Result, ensure};
use photonloom::{gpu::GpuStacker, integrate::integrate_tile_cpu};
use std::time::Instant;

fn main() -> Result<()> {
    let mut gpu = GpuStacker::try_new()?;
    println!("Hardware adapter: {}", gpu.adapter_name);
    for frames in [7usize, 65, 117, 175, 1024] {
        let pixels = 4096usize;
        let dim = gpu.clamped_tile_dim(frames);
        ensure!(
            dim > 0 && pixels <= dim * dim,
            "Probe does not fit device limits"
        );
        let mut values = vec![0.0f32; pixels * frames];
        let mut masks = vec![1u32; pixels * frames];
        let weights: Vec<_> = (0..frames).map(|i| (1 + i % 5) as f32).collect();
        for k in 0..frames {
            for p in 0..pixels {
                values[k * pixels + p] =
                    40.0 + (p % 31) as f32 + ((k * 17 + p * 11) % 13) as f32 * 0.03;
                if k % 37 == 0 {
                    values[k * pixels + p] += 4000.0;
                }
                if p % 17 == 0 && k % 3 == 0 {
                    masks[k * pixels + p] = 0;
                }
                if p == 0 {
                    masks[k * pixels + p] = 0;
                }
                if p == 1 && k > 0 {
                    masks[k * pixels + p] = 0;
                }
                if p == 2 && k > 1 {
                    masks[k * pixels + p] = 0;
                }
                if p == 3 {
                    values[k * pixels + p] = 42.0;
                }
            }
        }
        let started = Instant::now();
        let cpu = integrate_tile_cpu(pixels, frames, &values, &masks, &weights);
        let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let output = gpu.integrate_tile(pixels, frames, &values, &masks, &weights)?;
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            cpu.1 == output.1,
            "Validity masks differ for {frames} frames"
        );
        let error = cpu
            .0
            .iter()
            .zip(&output.0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(
            error < 0.005,
            "CPU/GPU discrepancy: {error} for {frames} frames"
        );
        ensure!(
            output.1[0] == 0 && (output.0[3] - 42.0).abs() < 0.001,
            "Mask/constant signal invariant failed"
        );
        println!(
            "{frames} frames: safe tile {dim}px, CPU {cpu_ms:.2}ms, GPU {gpu_ms:.2}ms, max difference {error:.6}"
        );
    }

    let (w, h, c) = (256usize, 256usize, 3usize);
    let np = w * h;
    let mut orig = vec![0.0f32; np * c];
    let mut seed = vec![0.0f32; np * c];
    let mut star_mask = vec![0.0f32; np];
    let valid_mask = vec![1u32; np];
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            let r2 = (x as f32 - 128.0).powi(2) + (y as f32 - 128.0).powi(2);
            let m = (-r2 / 36.0).exp();
            star_mask[p] = if m > 0.01 { m } else { 0.0 };
            for ch in 0..c {
                let bg = 0.12 + 0.03 * (ch as f32) + 0.0005 * (x as f32);
                let star = 0.80 * (-r2 / 10.0).exp();
                orig[ch * np + p] = bg + star;
                seed[ch * np + p] = (1.0 - star_mask[p]) * orig[ch * np + p] + star_mask[p] * bg;
            }
        }
    }
    let started = Instant::now();
    let cpu_starless = photonloom::stars::inpaint_starless_tile_cpu(
        w, h, c, &orig, &seed, &star_mask, &valid_mask,
    );
    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let gpu_starless = gpu.inpaint_starless_tile(
        w, h, c, &orig, &seed, &star_mask, &valid_mask,
    )?;
    let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    let starless_err = cpu_starless
        .iter()
        .zip(&gpu_starless)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    ensure!(
        starless_err < 1e-4,
        "Starless CPU/GPU discrepancy: {starless_err}"
    );
    println!(
        "starless 256x256x3 (24 passes): CPU {cpu_ms:.2}ms, GPU {gpu_ms:.2}ms, max difference {starless_err:.6}"
    );
    Ok(())
}
