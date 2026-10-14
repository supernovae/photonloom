use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};

// Three device frame stacks (values, masks, scratch), two reserved for GPU
// upload staging/allocator overhead, plus outputs/readbacks.
const TILE_WORKING_SET_BYTES: u64 = 128 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StackParams {
    num_pixels: u32,
    num_frames: u32,
    _pad0: u32,
    _pad1: u32,
}

const STACK_SHADER_WGSL: &str = r#"
struct StackParams {
    num_pixels: u32,
    num_frames: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: StackParams;
@group(0) @binding(1) var<storage, read> tile_values: array<f32>;
@group(0) @binding(2) var<storage, read> tile_masks: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_values: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_masks: array<u32>;
// Frame-major scratch: consecutive invocations access consecutive pixels.
// Each invocation owns one pixel column; this is reused for median and MAD.
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;

fn median(p: u32, count: u32) -> f32 {
    let np = params.num_pixels;
    let half = count / 2u;
    var left = 0u;
    var right = count - 1u;
    // Three-way selection handles repeated/constant values in linear time.
    loop {
        if (left >= right) {
            break;
        }
        let pivot = scratch[((left + right) / 2u) * np + p];
        var lt = left;
        var i = left;
        var gt = right + 1u;
        while (i < gt) {
            let v = scratch[i * np + p];
            if (v < pivot) {
                let old = scratch[lt * np + p];
                scratch[lt * np + p] = v;
                scratch[i * np + p] = old;
                lt = lt + 1u;
                i = i + 1u;
            } else if (v > pivot) {
                gt = gt - 1u;
                let old = scratch[gt * np + p];
                scratch[gt * np + p] = v;
                scratch[i * np + p] = old;
            } else {
                i = i + 1u;
            }
        }
        if (half < lt) {
            right = lt - 1u;
        } else if (half >= gt) {
            left = gt;
        } else {
            break;
        }
    }
    let upper = scratch[half * np + p];
    if ((count & 1u) != 0u) {
        return upper;
    }
    // Selection partitions, but does not sort the lower half.
    var lower = scratch[p];
    for (var k = 1u; k < half; k = k + 1u) {
        lower = max(lower, scratch[k * np + p]);
    }
    return 0.5 * (lower + upper);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.num_pixels) {
        return;
    }
    let nf = params.num_frames;
    let np = params.num_pixels;
    var m = 0u;
    var first = 0.0;
    var second = 0.0;
    var first_w = 0.0;
    var second_w = 0.0;
    for (var k = 0u; k < nf; k = k + 1u) {
        let idx = k * np + p;
        if (tile_masks[idx] != 0u) {
            let v = tile_values[idx];
            scratch[m * np + p] = v;
            if (m == 0u) {
                first = v;
                first_w = weights[k];
            } else if (m == 1u) {
                second = v;
                second_w = weights[k];
            }
            m = m + 1u;
        }
    }
    if (m == 0u) {
        out_values[p] = 0.0;
        out_masks[p] = 0u;
        return;
    }
    if (m == 1u) {
        out_values[p] = first;
        out_masks[p] = 1u;
        return;
    }
    if (m == 2u) {
        let wsum = first_w + second_w;
        if (wsum > 0.0) {
            out_values[p] = (first_w * first + second_w * second) / wsum;
        } else {
            out_values[p] = 0.5 * (first + second);
        }
        out_masks[p] = 1u;
        return;
    }

    var mu = median(p, m);
    for (var i = 0u; i < m; i = i + 1u) {
        let idx = i * np + p;
        scratch[idx] = abs(scratch[idx] - mu);
    }
    let mad = median(p, m);
    var sigma = max(1.4826 * mad, 1e-4 * max(abs(mu), 1.0));

    // Reload originals in frame order for the same weighted sums as the CPU.
    for (var iter = 0u; iter < 3u; iter = iter + 1u) {
        let low = mu - 3.0 * sigma;
        let high = mu + 3.0 * sigma;
        var w_sum = 0.0;
        var wy_sum = 0.0;
        for (var k = 0u; k < nf; k = k + 1u) {
            let idx = k * np + p;
            if (tile_masks[idx] != 0u) {
                let w = weights[k];
                w_sum = w_sum + w;
                wy_sum = wy_sum + w * clamp(tile_values[idx], low, high);
            }
        }
        if (w_sum <= 0.0) {
            break;
        }
        let mu_new = wy_sum / w_sum;
        var var_sum = 0.0;
        for (var k = 0u; k < nf; k = k + 1u) {
            let idx = k * np + p;
            if (tile_masks[idx] != 0u) {
                let u = clamp(tile_values[idx], low, high);
                let diff = u - mu_new;
                var_sum = var_sum + weights[k] * diff * diff;
            }
        }
        mu = mu_new;
        sigma = max(1.134 * sqrt(max(var_sum / w_sum, 0.0)), 1e-4 * max(abs(mu), 1.0));
    }
    let final_low = mu - 3.0 * sigma;
    let final_high = mu + 3.0 * sigma;
    var inlier_w_sum = 0.0;
    var inlier_val_sum = 0.0;
    for (var k = 0u; k < nf; k = k + 1u) {
        let idx = k * np + p;
        if (tile_masks[idx] != 0u) {
            let v = tile_values[idx];
            if (v >= final_low && v <= final_high) {
                let w = weights[k];
                inlier_w_sum = inlier_w_sum + w;
                inlier_val_sum = inlier_val_sum + w * v;
            }
        }
    }
    if (inlier_w_sum > 0.0) {
        out_values[p] = inlier_val_sum / inlier_w_sum;
    } else {
        out_values[p] = mu;
    }
    out_masks[p] = 1u;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StarlessParams {
    width: u32,
    height: u32,
    channels: u32,
    step: u32,
}

const STARLESS_SHADER_WGSL: &str = r#"
struct StarlessParams {
    width: u32,
    height: u32,
    channels: u32,
    step: u32,
}

@group(0) @binding(0) var<uniform> params: StarlessParams;
@group(0) @binding(1) var<storage, read> orig_values: array<f32>;
@group(0) @binding(2) var<storage, read> src_values: array<f32>;
@group(0) @binding(3) var<storage, read> star_mask: array<f32>;
@group(0) @binding(4) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_values: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let np = params.width * params.height;
    let total = np * params.channels;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let c = idx / np;
    let p = idx % np;
    let orig = orig_values[idx];
    let alpha = clamp(star_mask[p], 0.0, 1.0);
    if (valid_mask[p] == 0u || alpha <= 1e-5) {
        dst_values[idx] = orig;
        return;
    }
    let w_i = i32(params.width);
    let h_i = i32(params.height);
    let x = i32(p % params.width);
    let y = i32(p / params.width);
    let s = i32(params.step);
    let plane_offset = c * np;

    var sum_v = 0.0;
    var sum_w = 0.0;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        let ny = clamp(y + dy * s, 0, h_i - 1);
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nx = clamp(x + dx * s, 0, w_i - 1);
            let np_idx = u32(ny) * params.width + u32(nx);
            if (valid_mask[np_idx] != 0u) {
                var w_dir = 0.10;
                if (dx == 0 || dy == 0) {
                    w_dir = 0.15;
                }
                let m_nbr = clamp(star_mask[np_idx], 0.0, 1.0);
                let w = w_dir * (1.0 - 0.5 * m_nbr);
                sum_v = sum_v + w * src_values[plane_offset + np_idx];
                sum_w = sum_w + w;
            }
        }
    }

    var avg = src_values[idx];
    if (sum_w > 0.0) {
        avg = sum_v / sum_w;
    }
    let cand = (1.0 - alpha) * orig + alpha * avg;
    dst_values[idx] = min(orig, cand);
}
"#;

/// Portable hardware compute, with optional native HIP on WSL/ROCm hosts.
/// ROCm is discovered at runtime and is never required to build Photonloom.
pub struct GpuStacker {
    pub adapter_name: String,
    backend: StackBackend,
}

enum StackBackend {
    Portable(PortableStacker),
    #[cfg(target_os = "linux")]
    Hip(crate::hip::HipStacker),
}

impl GpuStacker {
    pub fn try_new() -> Result<Self> {
        let mut errors = Vec::new();
        #[cfg(target_os = "linux")]
        let wsl = std::path::Path::new("/dev/dxg").exists();
        #[cfg(target_os = "linux")]
        if wsl {
            match crate::hip::HipStacker::try_new() {
                Ok(hip) => {
                    return Ok(Self {
                        adapter_name: hip.adapter_name.clone(),
                        backend: StackBackend::Hip(hip),
                    });
                }
                Err(error) => errors.push(format!("Native WSL HIP: {error:#}")),
            }
        }
        match PortableStacker::try_new() {
            Ok(portable) => {
                return Ok(Self {
                    adapter_name: portable.adapter_name.clone(),
                    backend: StackBackend::Portable(portable),
                });
            }
            Err(error) => errors.push(format!("Portable GPU: {error:#}")),
        }
        #[cfg(target_os = "linux")]
        if !wsl {
            match crate::hip::HipStacker::try_new() {
                Ok(hip) => {
                    return Ok(Self {
                        adapter_name: hip.adapter_name.clone(),
                        backend: StackBackend::Hip(hip),
                    });
                }
                Err(error) => errors.push(format!("Native HIP: {error:#}")),
            }
        }
        bail!("No usable hardware GPU backend: {}", errors.join("; "))
    }

    pub fn clamped_tile_dim(&self, frames: usize) -> usize {
        match &self.backend {
            StackBackend::Portable(gpu) => gpu.clamped_tile_dim(frames),
            #[cfg(target_os = "linux")]
            StackBackend::Hip(gpu) => gpu.clamped_tile_dim(frames),
        }
    }

    pub fn integrate_tile(
        &mut self,
        pixels: usize,
        frames: usize,
        values: &[f32],
        masks: &[u32],
        weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        match &mut self.backend {
            StackBackend::Portable(gpu) => {
                gpu.integrate_tile(pixels, frames, values, masks, weights)
            }
            #[cfg(target_os = "linux")]
            StackBackend::Hip(gpu) => gpu.integrate_tile(pixels, frames, values, masks, weights),
        }
    }

    pub fn inpaint_starless_tile(
        &mut self,
        width: usize,
        height: usize,
        channels: usize,
        orig_values: &[f32],
        seed_values: &[f32],
        star_mask: &[f32],
        valid_mask: &[u32],
    ) -> Result<Vec<f32>> {
        match &mut self.backend {
            StackBackend::Portable(gpu) => gpu.inpaint_starless_tile(
                width,
                height,
                channels,
                orig_values,
                seed_values,
                star_mask,
                valid_mask,
            ),
            #[cfg(target_os = "linux")]
            StackBackend::Hip(gpu) => gpu.inpaint_starless_tile(
                width,
                height,
                channels,
                orig_values,
                seed_values,
                star_mask,
                valid_mask,
            ),
        }
    }
}

struct PortableStacker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    starless_pipeline: wgpu::ComputePipeline,
    starless_bgl: wgpu::BindGroupLayout,
    max_binding_bytes: u64,
    max_workgroups: u32,
    pub adapter_name: String,
    // Reused GPU buffers
    capacity_pixels: usize,
    capacity_frames: usize,
    params_buf: Option<wgpu::Buffer>,
    values_buf: Option<wgpu::Buffer>,
    masks_buf: Option<wgpu::Buffer>,
    scratch_buf: Option<wgpu::Buffer>,
    weights_buf: Option<wgpu::Buffer>,
    out_values_buf: Option<wgpu::Buffer>,
    out_masks_buf: Option<wgpu::Buffer>,
    readback_values_buf: Option<wgpu::Buffer>,
    readback_masks_buf: Option<wgpu::Buffer>,
    // Reused starless buffers
    starless_cap_pixels: usize,
    starless_cap_channels: usize,
    starless_params_bufs: Option<[wgpu::Buffer; 3]>,
    starless_orig_buf: Option<wgpu::Buffer>,
    starless_ping_buf: Option<wgpu::Buffer>,
    starless_pong_buf: Option<wgpu::Buffer>,
    starless_mask_buf: Option<wgpu::Buffer>,
    starless_valid_buf: Option<wgpu::Buffer>,
    starless_readback_buf: Option<wgpu::Buffer>,
}

impl PortableStacker {
    /// Initializes a portable wgpu compute context (Vulkan on Linux/Windows, Metal on macOS, D3D12 on Windows).
    pub fn try_new() -> Result<Self> {
        pollster::block_on(Self::try_new_async())
    }

    async fn try_new_async() -> Result<Self> {
        let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let mut errors = Vec::new();
        for adapter in instance.enumerate_adapters(backends) {
            let info = adapter.get_info();
            if info.device_type == wgpu::DeviceType::Cpu {
                errors.push(format!(
                    "{} ({:?}, CPU software adapter)",
                    info.name, info.backend
                ));
                continue;
            }
            let adapter_name =
                format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
            let mut limits = wgpu::Limits::default();
            let supported = adapter.limits();
            limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size;
            limits.max_buffer_size = supported.max_buffer_size;
            let (device, queue) = match adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("photonloom-stack-device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: limits,
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
            {
                Ok(pair) => pair,
                Err(e) => {
                    errors.push(format!("{adapter_name}: {e}"));
                    continue;
                }
            };
            let actual = device.limits();
            let max_binding_bytes =
                u64::from(actual.max_storage_buffer_binding_size).min(actual.max_buffer_size);
            return Ok(Self::from_device(
                device,
                queue,
                adapter_name,
                max_binding_bytes,
            ));
        }
        bail!(
            "No usable hardware Vulkan, Metal, or D3D12 GPU adapter: {}",
            errors.join("; ")
        )
    }

    fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        max_binding_bytes: u64,
    ) -> Self {
        let max_workgroups = device.limits().max_compute_workgroups_per_dimension;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("photonloom-stack-wgsl"),
            source: wgpu::ShaderSource::Wgsl(STACK_SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("photonloom-stack-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("photonloom-stack-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("photonloom-stack-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let starless_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("photonloom-starless-wgsl"),
            source: wgpu::ShaderSource::Wgsl(STARLESS_SHADER_WGSL.into()),
        });
        let starless_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("photonloom-starless-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let starless_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("photonloom-starless-pl"),
            bind_group_layouts: &[&starless_bgl],
            push_constant_ranges: &[],
        });
        let starless_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("photonloom-starless-pipeline"),
            layout: Some(&starless_pl),
            module: &starless_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            starless_pipeline,
            starless_bgl,
            max_binding_bytes,
            max_workgroups,
            adapter_name,
            capacity_pixels: 0,
            capacity_frames: 0,
            params_buf: None,
            values_buf: None,
            masks_buf: None,
            weights_buf: None,
            scratch_buf: None,
            out_values_buf: None,
            out_masks_buf: None,
            readback_values_buf: None,
            readback_masks_buf: None,
            starless_cap_pixels: 0,
            starless_cap_channels: 0,
            starless_params_bufs: None,
            starless_orig_buf: None,
            starless_ping_buf: None,
            starless_pong_buf: None,
            starless_mask_buf: None,
            starless_valid_buf: None,
            starless_readback_buf: None,
        }
    }

    /// Largest safe square tile edge (at most 512). Zero means even one pixel
    /// cannot fit the device binding limits or conservative 128 MiB working set.
    pub fn clamped_tile_dim(&self, num_frames: usize) -> usize {
        let max_pixels = self.max_tile_pixels(num_frames);
        let mut side = (max_pixels as f64).sqrt() as usize;
        while (side as u64) * (side as u64) > max_pixels {
            side -= 1;
        }
        side.min(512)
    }

    fn max_tile_pixels(&self, num_frames: usize) -> u64 {
        let Some(nf) = u64::try_from(num_frames)
            .ok()
            .filter(|&n| n > 0 && n <= u32::MAX as u64)
        else {
            return 0;
        };
        let Some(frame_bytes) = nf.checked_mul(4) else {
            return 0;
        };
        if frame_bytes > self.max_binding_bytes || frame_bytes >= TILE_WORKING_SET_BYTES {
            return 0;
        }
        // Values, masks and scratch on device; reserve two additional stacks
        // for upload staging/allocator overhead and four output/readback buffers.
        let max_by_budget = (TILE_WORKING_SET_BYTES - frame_bytes) / (20 * nf + 16);
        (self.max_binding_bytes / frame_bytes)
            .min(self.max_binding_bytes / 4)
            .min(u64::from(self.max_workgroups) * 64)
            .min(u32::MAX as u64)
            .min(max_by_budget)
    }

    fn ensure_buffers(&mut self, num_pixels: usize, num_frames: usize) -> Result<()> {
        if self.params_buf.is_some()
            && num_pixels <= self.capacity_pixels
            && num_frames <= self.capacity_frames
        {
            return Ok(());
        }
        let cap_p = num_pixels;
        let cap_f = num_frames;
        let stack_bytes = (cap_p as u64) * (cap_f as u64) * 4;
        let out_bytes = (cap_p as u64) * 4;
        let weights_bytes = (cap_f as u64) * 4;
        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        self.params_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params_buf"),
            size: std::mem::size_of::<StackParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.values_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values_buf"),
            size: stack_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.masks_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("masks_buf"),
            size: stack_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.scratch_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_buf"),
            size: stack_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        self.weights_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights_buf"),
            size: weights_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.out_values_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_values_buf"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.out_masks_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_masks_buf"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.readback_values_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_values_buf"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.readback_masks_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_masks_buf"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            self.params_buf = None;
            self.capacity_pixels = 0;
            self.capacity_frames = 0;
            bail!("GPU buffer allocation failed within the tile budget: {error}");
        }

        self.capacity_pixels = cap_p;
        self.capacity_frames = cap_f;
        Ok(())
    }

    pub fn integrate_tile(
        &mut self,
        num_pixels: usize,
        num_frames: usize,
        tile_values: &[f32],
        tile_masks_u32: &[u32],
        weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        if num_frames == 0 {
            bail!("GPU integration requires at least one frame");
        }
        if tile_values.len()
            != num_pixels
                .checked_mul(num_frames)
                .context("GPU tile size overflows usize")?
            || tile_masks_u32.len() != tile_values.len()
            || weights.len() != num_frames
        {
            bail!("GPU tile values, masks, or weights have incorrect lengths");
        }
        if num_pixels == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if num_pixels > u32::MAX as usize
            || num_frames > u32::MAX as usize
            || (num_pixels as u64) > self.max_tile_pixels(num_frames)
        {
            bail!(
                "GPU tile {num_pixels} pixels x {num_frames} frames exceeds device binding, dispatch, or 128 MiB working-set limits"
            );
        }
        self.ensure_buffers(num_pixels, num_frames)?;

        let params = StackParams {
            num_pixels: num_pixels as u32,
            num_frames: num_frames as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = self.params_buf.as_ref().unwrap();
        let values_buf = self.values_buf.as_ref().unwrap();
        let masks_buf = self.masks_buf.as_ref().unwrap();
        let weights_buf = self.weights_buf.as_ref().unwrap();
        let scratch_buf = self.scratch_buf.as_ref().unwrap();
        let out_values_buf = self.out_values_buf.as_ref().unwrap();
        let out_masks_buf = self.out_masks_buf.as_ref().unwrap();
        let rb_values_buf = self.readback_values_buf.as_ref().unwrap();
        let rb_masks_buf = self.readback_masks_buf.as_ref().unwrap();

        self.queue
            .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(values_buf, 0, bytemuck::cast_slice(tile_values));
        self.queue
            .write_buffer(masks_buf, 0, bytemuck::cast_slice(tile_masks_u32));
        self.queue
            .write_buffer(weights_buf, 0, bytemuck::cast_slice(weights));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photonloom-stack-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: masks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_masks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        let out_byte_len = (num_pixels * 4) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("photonloom-stack-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("photonloom-stack-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_pixels as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(out_values_buf, 0, rb_values_buf, 0, out_byte_len);
        encoder.copy_buffer_to_buffer(out_masks_buf, 0, rb_masks_buf, 0, out_byte_len);
        self.queue.submit(Some(encoder.finish()));

        let val_slice = rb_values_buf.slice(0..out_byte_len);
        let mask_slice = rb_masks_buf.slice(0..out_byte_len);

        let (tx_v, rx_v) = std::sync::mpsc::channel();
        let (tx_m, rx_m) = std::sync::mpsc::channel();
        val_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_v.send(res);
        });
        mask_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_m.send(res);
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx_v.recv()
            .context("GPU value readback channel closed")?
            .context("Failed to map GPU output value buffer")?;
        rx_m.recv()
            .context("GPU mask readback channel closed")?
            .context("Failed to map GPU output mask buffer")?;

        let out_vals: Vec<f32> = {
            let view = val_slice.get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        let out_masks: Vec<u8> = {
            let view = mask_slice.get_mapped_range();
            let u32_slice: &[u32] = bytemuck::cast_slice(&view);
            u32_slice.iter().map(|&m| u8::from(m != 0)).collect()
        };

        rb_values_buf.unmap();
        rb_masks_buf.unmap();

        Ok((out_vals, out_masks))
    }

    fn ensure_starless_buffers(&mut self, num_pixels: usize, channels: usize) -> Result<()> {
        if self.starless_params_bufs.is_some()
            && num_pixels <= self.starless_cap_pixels
            && channels <= self.starless_cap_channels
        {
            return Ok(());
        }
        let total_bytes = (num_pixels as u64) * (channels as u64) * 4;
        let mask_bytes = (num_pixels as u64) * 4;
        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let make_param = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<StarlessParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        self.starless_params_bufs = Some([
            make_param("starless_params_s4"),
            make_param("starless_params_s2"),
            make_param("starless_params_s1"),
        ]);
        self.starless_orig_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starless_orig_buf"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.starless_ping_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starless_ping_buf"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.starless_pong_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starless_pong_buf"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        self.starless_mask_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starless_mask_buf"),
            size: mask_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.starless_valid_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starless_valid_buf"),
            size: mask_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.starless_readback_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("starless_readback_buf"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            self.starless_params_bufs = None;
            self.starless_cap_pixels = 0;
            self.starless_cap_channels = 0;
            bail!("GPU starless buffer allocation failed: {error}");
        }
        self.starless_cap_pixels = num_pixels;
        self.starless_cap_channels = channels;
        Ok(())
    }

    pub fn inpaint_starless_tile(
        &mut self,
        width: usize,
        height: usize,
        channels: usize,
        orig_values: &[f32],
        seed_values: &[f32],
        star_mask: &[f32],
        valid_mask: &[u32],
    ) -> Result<Vec<f32>> {
        let np = width.checked_mul(height).context("GPU starless tile dimensions overflow")?;
        let total = np.checked_mul(channels).context("GPU starless channel size overflow")?;
        if channels == 0 || orig_values.len() != total || seed_values.len() != total
            || star_mask.len() != np || valid_mask.len() != np
        {
            bail!("GPU starless tile inputs have invalid dimensions or lengths");
        }
        if np == 0 {
            return Ok(Vec::new());
        }
        let total_bytes = (total as u64) * 4;
        if total > u32::MAX as usize
            || total_bytes > self.max_binding_bytes
            || total_bytes * 4 > TILE_WORKING_SET_BYTES
            || (total as u64).div_ceil(64) > u64::from(self.max_workgroups)
        {
            bail!("GPU starless tile exceeds device binding or dispatch limits");
        }
        self.ensure_starless_buffers(np, channels)?;

        let params_bufs = self.starless_params_bufs.as_ref().unwrap();
        let orig_buf = self.starless_orig_buf.as_ref().unwrap();
        let ping_buf = self.starless_ping_buf.as_ref().unwrap();
        let pong_buf = self.starless_pong_buf.as_ref().unwrap();
        let mask_buf = self.starless_mask_buf.as_ref().unwrap();
        let valid_buf = self.starless_valid_buf.as_ref().unwrap();
        let rb_buf = self.starless_readback_buf.as_ref().unwrap();

        for (idx, step) in [4u32, 2u32, 1u32].into_iter().enumerate() {
            let p = StarlessParams {
                width: width as u32,
                height: height as u32,
                channels: channels as u32,
                step,
            };
            self.queue
                .write_buffer(&params_bufs[idx], 0, bytemuck::bytes_of(&p));
        }
        self.queue
            .write_buffer(orig_buf, 0, bytemuck::cast_slice(orig_values));
        self.queue
            .write_buffer(ping_buf, 0, bytemuck::cast_slice(seed_values));
        self.queue
            .write_buffer(mask_buf, 0, bytemuck::cast_slice(star_mask));
        self.queue
            .write_buffer(valid_buf, 0, bytemuck::cast_slice(valid_mask));

        let make_bg = |param_buf: &wgpu::Buffer, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("photonloom-starless-bg"),
                layout: &self.starless_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: orig_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mask_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: valid_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: dst.as_entire_binding(),
                    },
                ],
            })
        };

        let bg_a_to_b = [
            make_bg(&params_bufs[0], ping_buf, pong_buf),
            make_bg(&params_bufs[1], ping_buf, pong_buf),
            make_bg(&params_bufs[2], ping_buf, pong_buf),
        ];
        let bg_b_to_a = [
            make_bg(&params_bufs[0], pong_buf, ping_buf),
            make_bg(&params_bufs[1], pong_buf, ping_buf),
            make_bg(&params_bufs[2], pong_buf, ping_buf),
        ];

        let workgroups = (total as u32).div_ceil(64);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("photonloom-starless-enc"),
            });
        for (iter, &step) in crate::stars::STARLESS_DIFFUSION_STEPS.iter().enumerate() {
            let step_idx = match step {
                4 => 0,
                2 => 1,
                _ => 2,
            };
            let bg = if iter % 2 == 0 {
                &bg_a_to_b[step_idx]
            } else {
                &bg_b_to_a[step_idx]
            };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("photonloom-starless-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.starless_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(ping_buf, 0, rb_buf, 0, total_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = rb_buf.slice(0..total_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .context("GPU starless readback channel closed")?
            .context("Failed to map GPU starless output buffer")?;
        let out: Vec<f32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        rb_buf.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::PortableStacker;
    use crate::integrate::integrate_tile_cpu;

    // Run the production WGSL on a real Vulkan software device, even though
    // production intentionally rejects CPU adapters as GPU accelerators.
    #[test]
    fn vulkan_software_shader_matches_cpu_above_64_frames() {
        let backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(backends);
        let software = adapters.into_iter().filter(|adapter| {
            let info = adapter.get_info();
            info.backend == wgpu::Backend::Vulkan && info.device_type == wgpu::DeviceType::Cpu
        });
        let mut unavailable = Vec::new();
        let mut stacker = None;
        for adapter in software {
            let info = adapter.get_info();
            let mut limits = wgpu::Limits::default();
            let supported = adapter.limits();
            limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size;
            limits.max_buffer_size = supported.max_buffer_size;
            match pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("photonloom-test-software-vulkan"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )) {
                Ok((device, queue)) => {
                    let actual = device.limits();
                    let max_binding_bytes = u64::from(actual.max_storage_buffer_binding_size)
                        .min(actual.max_buffer_size);
                    stacker = Some(PortableStacker::from_device(
                        device,
                        queue,
                        info.name,
                        max_binding_bytes,
                    ));
                    break;
                }
                Err(error) => unavailable.push(format!("{}: {error}", info.name)),
            }
        }
        let Some(mut stacker) = stacker else {
            eprintln!(
                "Skipping actual WGSL regression: no usable Vulkan CPU software adapter ({})",
                if unavailable.is_empty() {
                    "none enumerated".to_string()
                } else {
                    unavailable.join("; ")
                }
            );
            return;
        };
        eprintln!("WGSL regression adapter: {}", stacker.adapter_name);

        // Exercise reallocations as well as 65/175/1024-frame selection and
        // 3-sigma clipping. Frame-major indexing matches consumer tiles.
        const NP: usize = 256;
        for nf in [65, 175, 1024] {
            let weights: Vec<f32> = (0..nf)
                .map(|k| 0.5 + ((k * 13 % 31) as f32) * 0.125)
                .collect();
            let mut values = vec![0.0f32; NP * nf];
            let mut masks = vec![0u32; NP * nf];
            for k in 0..nf {
                for p in 0..NP {
                    let idx = k * NP + p;
                    // Deliberately nonzero masked samples catch mask leakage.
                    values[idx] = 90_000.0 + k as f32;
                    masks[idx] = match p {
                        0 => 0,                                // no valid samples
                        1 => u32::from(k == nf / 3),           // one
                        2 => u32::from(k == 3 || k == nf - 2), // two
                        _ => u32::from((k + p * 17) % 23 != 0),
                    };
                    if masks[idx] == 0 {
                        continue;
                    }
                    values[idx] = match p {
                        1 => -17.25,
                        2 if k == 3 => -7.0,
                        2 => 11.0,
                        3 => 42.0, // zero MAD even with unequal weights
                        4 => -42.0,
                        5 => {
                            if k == nf / 2 {
                                100_000.0
                            } else {
                                42.0
                            }
                        }
                        6 => {
                            if k == nf / 2 {
                                -100_000.0
                            } else {
                                -42.0
                            }
                        }
                        _ => {
                            let baseline = (p as f32 - 125.0) * 0.13;
                            let noise = ((k * 37 + p * 19) % 47) as f32 * 0.019 - 0.44;
                            if (k * 11 + p) % 97 == 0 {
                                baseline + if p % 2 == 0 { 170.0 } else { -170.0 }
                            } else {
                                baseline + noise
                            }
                        }
                    };
                }
            }
            let expected = integrate_tile_cpu(NP, nf, &values, &masks, &weights);
            let actual = stacker
                .integrate_tile(NP, nf, &values, &masks, &weights)
                .expect("actual Vulkan shader dispatch and readback");
            assert_eq!(actual.1, expected.1, "output masks for {nf} frames");
            for p in 0..NP {
                let tolerance = 0.002 + 0.0001 * expected.0[p].abs();
                assert!(
                    (actual.0[p] - expected.0[p]).abs() <= tolerance,
                    "{nf} frames pixel {p}: Vulkan {}, CPU {}, tolerance {tolerance}",
                    actual.0[p],
                    expected.0[p],
                );
            }
            assert_eq!(actual.1[0], 0, "fully masked pixel");
            assert_eq!(actual.0[0], 0.0, "fully masked value");
            assert_eq!(actual.0[1], -17.25, "one valid negative sample");
            assert!(
                (actual.0[3] - 42.0).abs() < 0.002,
                "constant positive samples"
            );
            assert!(
                (actual.0[4] + 42.0).abs() < 0.002,
                "constant negative samples"
            );
            assert!(
                (actual.0[5] - 42.0).abs() < 0.002,
                "positive satellite rejected"
            );
            assert!(
                (actual.0[6] + 42.0).abs() < 0.002,
                "negative outlier rejected"
            );
        }
    }

    #[test]
    fn vulkan_software_starless_shader_matches_cpu() {
        let backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(backends);
        let software = adapters.into_iter().filter(|adapter| {
            let info = adapter.get_info();
            info.backend == wgpu::Backend::Vulkan && info.device_type == wgpu::DeviceType::Cpu
        });
        let mut stacker = None;
        for adapter in software {
            let info = adapter.get_info();
            let mut limits = wgpu::Limits::default();
            let supported = adapter.limits();
            limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size;
            limits.max_buffer_size = supported.max_buffer_size;
            if let Ok((device, queue)) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("photonloom-test-software-vulkan-starless"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )) {
                let actual = device.limits();
                let max_binding_bytes = u64::from(actual.max_storage_buffer_binding_size)
                    .min(actual.max_buffer_size);
                stacker = Some(PortableStacker::from_device(
                    device,
                    queue,
                    info.name,
                    max_binding_bytes,
                ));
                break;
            }
        }
        let Some(mut stacker) = stacker else {
            return;
        };

        let (w, h, c) = (32usize, 32usize, 3usize);
        let np = w * h;
        let mut orig = vec![0.0f32; np * c];
        let mut seed = vec![0.0f32; np * c];
        let mut star_mask = vec![0.0f32; np];
        let valid_mask = vec![1u32; np];

        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let r2 = (x as f32 - 16.0).powi(2) + (y as f32 - 16.0).powi(2);
                let m = (-r2 / 18.0).exp();
                star_mask[p] = if m > 0.02 { m } else { 0.0 };
                for ch in 0..c {
                    let bg = 0.15 + 0.04 * (ch as f32) + 0.002 * (x as f32) + 0.001 * (y as f32);
                    let star = 0.75 * (-r2 / 6.0).exp();
                    orig[ch * np + p] = bg + star;
                    seed[ch * np + p] = (1.0 - star_mask[p]) * orig[ch * np + p] + star_mask[p] * bg;
                }
            }
        }

        let cpu = crate::stars::inpaint_starless_tile_cpu(
            w, h, c, &orig, &seed, &star_mask, &valid_mask,
        );
        let gpu = stacker
            .inpaint_starless_tile(w, h, c, &orig, &seed, &star_mask, &valid_mask)
            .expect("Vulkan starless dispatch");
        assert_eq!(cpu.len(), gpu.len());
        for (idx, (&a, &b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4,
                "Starless mismatch at {idx}: cpu={a}, gpu={b}"
            );
        }
    }
}
