//! Optional ROCm compute backend. No HIP library or headers are needed to build photonloom.
//! The kernel is compiled for the active device by HIPRTC when this backend is selected.
use anyhow::{Context, Result, bail};
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;

const WORKING_SET: usize = 128 * 1024 * 1024;
const THREADS: usize = 64;
type HipResult = c_int;
type RtcResult = c_int;
type Handle = *mut c_void;
type DevicePtr = *mut c_void;

// Signatures are from hip_runtime_api.h and hiprtc.h. Opaque handles and typed
// function pointers avoid depending on the layout of versioned hipDeviceProp_t.
type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipResult;
type SetDevice = unsafe extern "C" fn(c_int) -> HipResult;
type GetDeviceName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipResult;
type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> HipResult;
type Malloc = unsafe extern "C" fn(*mut DevicePtr, usize) -> HipResult;
type Free = unsafe extern "C" fn(DevicePtr) -> HipResult;
type Memcpy = unsafe extern "C" fn(DevicePtr, *const c_void, usize, c_int) -> HipResult;
type ModuleLoadData = unsafe extern "C" fn(*mut Handle, *const c_void) -> HipResult;
type ModuleGetFunction = unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> HipResult;
type ModuleUnload = unsafe extern "C" fn(Handle) -> HipResult;
type ModuleLaunchKernel = unsafe extern "C" fn(
    Handle,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipResult;
type DeviceSynchronize = unsafe extern "C" fn() -> HipResult;
type GetErrorString = unsafe extern "C" fn(HipResult) -> *const c_char;
type RtcCreate = unsafe extern "C" fn(
    *mut Handle,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> RtcResult;
type RtcCompile = unsafe extern "C" fn(Handle, c_int, *const *const c_char) -> RtcResult;
type RtcDestroy = unsafe extern "C" fn(*mut Handle) -> RtcResult;
type RtcGetSize = unsafe extern "C" fn(Handle, *mut usize) -> RtcResult;
type RtcGetData = unsafe extern "C" fn(Handle, *mut c_char) -> RtcResult;
type RtcGetErrorString = unsafe extern "C" fn(RtcResult) -> *const c_char;

struct Api {
    // Libraries must outlive all function pointers and buffers using them.
    _runtime: Library,
    _rtc: Library,
    device_count: GetDeviceCount,
    set_device: SetDevice,
    device_name: GetDeviceName,
    mem_info: MemGetInfo,
    malloc: Malloc,
    free: Free,
    memcpy: Memcpy,
    module_load: ModuleLoadData,
    module_function: ModuleGetFunction,
    module_unload: ModuleUnload,
    launch: ModuleLaunchKernel,
    synchronize: DeviceSynchronize,
    error_string: GetErrorString,
    rtc_create: RtcCreate,
    rtc_compile: RtcCompile,
    rtc_destroy: RtcDestroy,
    rtc_code_size: RtcGetSize,
    rtc_code: RtcGetData,
    rtc_log_size: RtcGetSize,
    rtc_log: RtcGetData,
    rtc_error_string: RtcGetErrorString,
}

impl Api {
    fn load() -> Result<&'static Self> {
        // HIP/HIPRTC own background runtime threads. Keep their shared libraries
        // mapped for the process lifetime; dlclose after a job can invalidate
        // those threads even after all modules and buffers have been released.
        static API: std::sync::LazyLock<Result<Api>> = std::sync::LazyLock::new(Api::open);
        match &*API {
            Ok(api) => Ok(api),
            Err(error) => bail!("{error:#}"),
        }
    }

    fn open() -> Result<Self> {
        fn library(names: &[&str]) -> Result<Library> {
            let mut last = None;
            for name in names {
                // SAFETY: symbols are resolved by name to the signatures defined above;
                // the Library is retained until every module and device buffer is dropped.
                match unsafe { Library::new(name) } {
                    Ok(lib) => return Ok(lib),
                    Err(err) => last = Some(err),
                }
            }
            Err(last.context("No HIP shared library candidate")?.into())
        }
        let runtime = library(&[
            "libamdhip64.so.7",
            "/opt/rocm/lib/libamdhip64.so.7",
            "/opt/rocm/core-10.0/lib/libamdhip64.so.7",
            "libamdhip64.so",
        ])
        .context("Loading AMD HIP runtime")?;
        let rtc = library(&[
            "libhiprtc.so.7",
            "/opt/rocm/lib/libhiprtc.so.7",
            "/opt/rocm/core-10.0/lib/libhiprtc.so.7",
            "libhiprtc.so",
        ])
        .context("Loading HIP runtime compiler")?;
        unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
            // SAFETY: caller supplies the official C declaration for this exact symbol.
            Ok(unsafe { *library.get::<T>(name)? })
        }
        // SAFETY: each symbol's type matches its C ABI declaration in HIP's public API.
        unsafe {
            Ok(Self {
                device_count: symbol(&runtime, b"hipGetDeviceCount\0")?,
                set_device: symbol(&runtime, b"hipSetDevice\0")?,
                device_name: symbol(&runtime, b"hipDeviceGetName\0")?,
                mem_info: symbol(&runtime, b"hipMemGetInfo\0")?,
                malloc: symbol(&runtime, b"hipMalloc\0")?,
                free: symbol(&runtime, b"hipFree\0")?,
                memcpy: symbol(&runtime, b"hipMemcpy\0")?,
                module_load: symbol(&runtime, b"hipModuleLoadData\0")?,
                module_function: symbol(&runtime, b"hipModuleGetFunction\0")?,
                module_unload: symbol(&runtime, b"hipModuleUnload\0")?,
                launch: symbol(&runtime, b"hipModuleLaunchKernel\0")?,
                synchronize: symbol(&runtime, b"hipDeviceSynchronize\0")?,
                error_string: symbol(&runtime, b"hipGetErrorString\0")?,
                rtc_create: symbol(&rtc, b"hiprtcCreateProgram\0")?,
                rtc_compile: symbol(&rtc, b"hiprtcCompileProgram\0")?,
                rtc_destroy: symbol(&rtc, b"hiprtcDestroyProgram\0")?,
                rtc_code_size: symbol(&rtc, b"hiprtcGetCodeSize\0")?,
                rtc_code: symbol(&rtc, b"hiprtcGetCode\0")?,
                rtc_log_size: symbol(&rtc, b"hiprtcGetProgramLogSize\0")?,
                rtc_log: symbol(&rtc, b"hiprtcGetProgramLog\0")?,
                rtc_error_string: symbol(&rtc, b"hiprtcGetErrorString\0")?,
                _runtime: runtime,
                _rtc: rtc,
            })
        }
    }

    fn check(&self, code: HipResult, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: HIP returns a static NUL-terminated message or null.
        let message = unsafe { self.error_string_string(code) };
        bail!("{operation}: HIP error {code}: {message}")
    }
    unsafe fn error_string_string(&self, code: HipResult) -> String {
        let p = unsafe { (self.error_string)(code) };
        if p.is_null() {
            return "unknown error".into();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
    fn rtc_check(&self, code: RtcResult, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: HIPRTC returns a static NUL-terminated message or null.
        let p = unsafe { (self.rtc_error_string)(code) };
        let message = if p.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        bail!("{operation}: HIPRTC error {code}: {message}")
    }
}

struct Program<'a> {
    api: &'a Api,
    handle: Handle,
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: program belongs to the live HIPRTC library.
            unsafe {
                (self.api.rtc_destroy)(&mut self.handle);
            }
        }
    }
}

struct Buffer {
    ptr: DevicePtr,
    free: Free,
}
impl Buffer {
    fn new(api: &Api, bytes: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        // SAFETY: HIP owns the returned allocation; Buffer frees it on every exit path.
        let result = unsafe { (api.malloc)(&mut ptr, bytes) };
        if result != 0 && !ptr.is_null() {
            unsafe {
                (api.free)(ptr);
            }
        }
        api.check(result, "hipMalloc")?;
        if ptr.is_null() {
            bail!("hipMalloc returned a null allocation");
        }
        Ok(Self {
            ptr,
            free: api.free,
        })
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated by hipMalloc on the same device; no outstanding kernels
        // after a successful synchronous transfer or device synchronization.
        unsafe {
            (self.free)(self.ptr);
        }
    }
}

struct Buffers {
    values: Buffer,
    masks: Buffer,
    scratch: Buffer,
    weights: Buffer,
    output: Buffer,
    output_masks: Buffer,
    pixels: usize,
    frames: usize,
}
impl Buffers {
    fn new(api: &Api, pixels: usize, frames: usize) -> Result<Self> {
        let stack_bytes = pixels
            .checked_mul(frames)
            .and_then(|n| n.checked_mul(4))
            .context("HIP stack size overflow")?;
        let output_bytes = pixels.checked_mul(4).context("HIP output size overflow")?;
        let weight_bytes = frames.checked_mul(4).context("HIP weights size overflow")?;
        Ok(Self {
            values: Buffer::new(api, stack_bytes)?,
            masks: Buffer::new(api, stack_bytes)?,
            scratch: Buffer::new(api, stack_bytes)?,
            weights: Buffer::new(api, weight_bytes)?,
            output: Buffer::new(api, output_bytes)?,
            output_masks: Buffer::new(api, output_bytes)?,
            pixels,
            frames,
        })
    }
}

struct StarlessBuffers {
    orig: Buffer,
    ping: Buffer,
    pong: Buffer,
    star_mask: Buffer,
    valid_mask: Buffer,
    pixels: usize,
    channels: usize,
}
impl StarlessBuffers {
    fn new(api: &Api, pixels: usize, channels: usize) -> Result<Self> {
        let total_bytes = pixels
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(4))
            .context("HIP starless buffer size overflow")?;
        let mask_bytes = pixels
            .checked_mul(4)
            .context("HIP starless mask size overflow")?;
        Ok(Self {
            orig: Buffer::new(api, total_bytes)?,
            ping: Buffer::new(api, total_bytes)?,
            pong: Buffer::new(api, total_bytes)?,
            star_mask: Buffer::new(api, mask_bytes)?,
            valid_mask: Buffer::new(api, mask_bytes)?,
            pixels,
            channels,
        })
    }
}

pub(crate) struct HipStacker {
    // Drop buffers and module before the shared libraries (explicit Drop below).
    buffers: Option<Buffers>,
    starless_buffers: Option<StarlessBuffers>,
    module: Handle,
    kernel: Handle,
    starless_kernel: Handle,
    api: &'static Api,
    budget: usize,
    pub(crate) adapter_name: String,
}

// SAFETY: All HIP calls bind device 0 before touching device resources and synchronize before returning.
unsafe impl Send for HipStacker {}
impl HipStacker {
    pub(crate) fn try_new() -> Result<Self> {
        let api = Api::load()?;
        let mut devices = 0;
        api.check(
            unsafe { (api.device_count)(&mut devices) },
            "hipGetDeviceCount",
        )?;
        if devices < 1 {
            bail!("No AMD HIP device available");
        }
        api.check(unsafe { (api.set_device)(0) }, "hipSetDevice")?;
        let mut name = [0 as c_char; 256];
        api.check(
            unsafe { (api.device_name)(name.as_mut_ptr(), name.len() as c_int, 0) },
            "hipDeviceGetName",
        )?;
        name[name.len() - 1] = 0;
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut free = 0;
        let mut total = 0;
        api.check(
            unsafe { (api.mem_info)(&mut free, &mut total) },
            "hipMemGetInfo",
        )?;
        // Device allocations are three stacks, weights, and two outputs; retain
        // at least half the available device memory for the driver and other work.
        let budget = WORKING_SET.min(free / 2);
        if budget < 28 {
            bail!("HIP device has insufficient free memory for a tile");
        }

        let source = CString::new(include_str!("hip_stack.hip"))?;
        let filename = c"photonloom_stack.hip";
        let mut handle = ptr::null_mut();
        let create = unsafe {
            (api.rtc_create)(
                &mut handle,
                source.as_ptr(),
                filename.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };
        let program = Program { api: &api, handle };
        api.rtc_check(create, "hiprtcCreateProgram")?;
        // HIPRTC detects the selected GPU architecture through the HIP runtime
        // when no --gpu-architecture flag is supplied (HIPRTC documentation).
        // Disable floating-point contraction to preserve CPU f32 weighted sums.
        let option = c"-ffp-contract=off";
        let options = [option.as_ptr()];
        let compile = unsafe { (api.rtc_compile)(program.handle, 1, options.as_ptr()) };
        if compile != 0 {
            let mut size = 0;
            let log = if api
                .rtc_check(
                    unsafe { (api.rtc_log_size)(program.handle, &mut size) },
                    "hiprtcGetProgramLogSize",
                )
                .is_ok()
                && size > 0
            {
                let mut bytes = vec![0u8; size];
                if api
                    .rtc_check(
                        unsafe { (api.rtc_log)(program.handle, bytes.as_mut_ptr().cast()) },
                        "hiprtcGetProgramLog",
                    )
                    .is_ok()
                {
                    String::from_utf8_lossy(&bytes)
                        .trim_end_matches('\0')
                        .to_owned()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            api.rtc_check(
                compile,
                &format!("hiprtcCompileProgram ({device_name}): {log}"),
            )?;
        }
        let mut code_size = 0;
        api.rtc_check(
            unsafe { (api.rtc_code_size)(program.handle, &mut code_size) },
            "hiprtcGetCodeSize",
        )?;
        if code_size == 0 {
            bail!("HIPRTC returned an empty device code object");
        }
        let mut code = vec![0u8; code_size];
        api.rtc_check(
            unsafe { (api.rtc_code)(program.handle, code.as_mut_ptr().cast()) },
            "hiprtcGetCode",
        )?;
        drop(program);
        let mut module = ptr::null_mut();
        let loaded = unsafe { (api.module_load)(&mut module, code.as_ptr().cast()) };
        if loaded != 0 && !module.is_null() {
            unsafe {
                (api.module_unload)(module);
            }
        }
        api.check(loaded, "hipModuleLoadData")?;
        // Guard module immediately: if lookup fails, Drop must unload it.
        let mut stacker = Self {
            buffers: None,
            starless_buffers: None,
            module,
            kernel: ptr::null_mut(),
            starless_kernel: ptr::null_mut(),
            api,
            budget,
            adapter_name: format!("AMD HIP: {device_name}"),
        };
        stacker.api.check(
            unsafe {
                (stacker.api.module_function)(
                    &mut stacker.kernel,
                    stacker.module,
                    c"stack_tile".as_ptr(),
                )
            },
            "hipModuleGetFunction (stack_tile)",
        )?;
        stacker.api.check(
            unsafe {
                (stacker.api.module_function)(
                    &mut stacker.starless_kernel,
                    stacker.module,
                    c"starless_diffuse_step".as_ptr(),
                )
            },
            "hipModuleGetFunction (starless_diffuse_step)",
        )?;
        Ok(stacker)
    }

    fn max_pixels(&self, frames: usize) -> usize {
        if frames == 0 || frames > u32::MAX as usize {
            return 0;
        }
        let Some(frame_bytes) = frames.checked_mul(4) else {
            return 0;
        };
        let Some(per_pixel) = frames.checked_mul(12).and_then(|n| n.checked_add(8)) else {
            return 0;
        };
        self.budget
            .checked_sub(frame_bytes)
            .map_or(0, |room| room / per_pixel)
            .min(u32::MAX as usize)
            .min((u32::MAX as usize) / frames)
    }

    pub(crate) fn clamped_tile_dim(&self, frames: usize) -> usize {
        let max_pixels = self.max_pixels(frames);
        let mut side = (max_pixels as f64).sqrt() as usize;
        while side.saturating_mul(side) > max_pixels {
            side -= 1;
        }
        side.min(512)
    }

    pub(crate) fn integrate_tile(
        &mut self,
        np: usize,
        nf: usize,
        values: &[f32],
        masks: &[u32],
        weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        if nf == 0 {
            bail!("HIP integration requires at least one frame");
        }
        let items = np.checked_mul(nf).context("HIP tile length overflow")?;
        if values.len() != items || masks.len() != items || weights.len() != nf {
            bail!("HIP tile values, masks, or weights have incorrect lengths");
        }
        if np == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if np > self.max_pixels(nf) {
            bail!("HIP tile exceeds available device memory or 128 MiB working-set limit");
        }
        self.api
            .check(unsafe { (self.api.set_device)(0) }, "hipSetDevice")?;
        if self
            .buffers
            .as_ref()
            .is_none_or(|b| b.pixels < np || b.frames < nf)
        {
            // Reuse larger allocations; kernel parameters supply the current
            // frame-major stride. Allocate new exact capacities only on growth.
            self.buffers = None;
            self.buffers = Some(Buffers::new(&self.api, np, nf)?);
        }
        let b = self.buffers.as_ref().expect("HIP buffers allocated");
        let stack_bytes = items
            .checked_mul(4)
            .context("HIP stack byte count overflow")?;
        let weight_bytes = nf
            .checked_mul(4)
            .context("HIP weights byte count overflow")?;
        let output_bytes = np
            .checked_mul(4)
            .context("HIP output byte count overflow")?;
        // hipMemcpy with kind 1 (host-to-device) and 2 (device-to-host), as
        // declared by hipMemcpyKind in hip_runtime_api.h. Copies are synchronous.
        for (dst, src, len, label) in [
            (
                b.values.ptr,
                values.as_ptr().cast(),
                stack_bytes,
                "HIP values upload",
            ),
            (
                b.masks.ptr,
                masks.as_ptr().cast(),
                stack_bytes,
                "HIP masks upload",
            ),
            (
                b.weights.ptr,
                weights.as_ptr().cast(),
                weight_bytes,
                "HIP weights upload",
            ),
        ] {
            self.api
                .check(unsafe { (self.api.memcpy)(dst, src, len, 1) }, label)?;
        }
        let mut arg_values = b.values.ptr;
        let mut arg_masks = b.masks.ptr;
        let mut arg_weights = b.weights.ptr;
        let mut arg_scratch = b.scratch.ptr;
        let mut arg_output = b.output.ptr;
        let mut arg_output_masks = b.output_masks.ptr;
        let mut arg_np = np as u32;
        let mut arg_nf = nf as u32;
        let mut args: [*mut c_void; 8] = [
            (&mut arg_values as *mut DevicePtr).cast(),
            (&mut arg_masks as *mut DevicePtr).cast(),
            (&mut arg_weights as *mut DevicePtr).cast(),
            (&mut arg_scratch as *mut DevicePtr).cast(),
            (&mut arg_output as *mut DevicePtr).cast(),
            (&mut arg_output_masks as *mut DevicePtr).cast(),
            (&mut arg_np as *mut u32).cast(),
            (&mut arg_nf as *mut u32).cast(),
        ];
        let blocks = np.div_ceil(THREADS);
        let blocks = u32::try_from(blocks).context("HIP dispatch exceeds u32 workgroups")?;
        self.api.check(
            unsafe {
                (self.api.launch)(
                    self.kernel,
                    blocks,
                    1,
                    1,
                    THREADS as u32,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )?;
        self.api.check(
            unsafe { (self.api.synchronize)() },
            "hipDeviceSynchronize (stack_tile)",
        )?;
        let mut out = vec![0f32; np];
        let mut out_masks = vec![0u32; np];
        self.api.check(
            unsafe { (self.api.memcpy)(out.as_mut_ptr().cast(), b.output.ptr, output_bytes, 2) },
            "HIP values readback",
        )?;
        self.api.check(
            unsafe {
                (self.api.memcpy)(
                    out_masks.as_mut_ptr().cast(),
                    b.output_masks.ptr,
                    output_bytes,
                    2,
                )
            },
            "HIP masks readback",
        )?;
        Ok((
            out,
            out_masks.into_iter().map(|m| u8::from(m != 0)).collect(),
        ))
    }

    pub(crate) fn inpaint_starless_tile(
        &mut self,
        width: usize,
        height: usize,
        channels: usize,
        orig_values: &[f32],
        seed_values: &[f32],
        star_mask: &[f32],
        valid_mask: &[u32],
    ) -> Result<Vec<f32>> {
        let np = width.checked_mul(height).context("HIP starless tile dimensions overflow")?;
        let total = np.checked_mul(channels).context("HIP starless channel length overflow")?;
        if channels == 0 || orig_values.len() != total || seed_values.len() != total
            || star_mask.len() != np || valid_mask.len() != np
        {
            bail!("HIP starless tile inputs have invalid dimensions or lengths");
        }
        if np == 0 {
            return Ok(Vec::new());
        }
        let required_bytes = total
            .checked_mul(12)
            .and_then(|n| n.checked_add(np.checked_mul(8)?))
            .context("HIP starless working set overflow")?;
        if required_bytes > self.budget || total > u32::MAX as usize {
            bail!("HIP starless tile exceeds device memory budget");
        }
        self.api
            .check(unsafe { (self.api.set_device)(0) }, "hipSetDevice")?;
        if self
            .starless_buffers
            .as_ref()
            .is_none_or(|b| b.pixels < np || b.channels < channels)
        {
            self.starless_buffers = None;
            self.starless_buffers = Some(StarlessBuffers::new(&self.api, np, channels)?);
        }
        let b = self.starless_buffers.as_ref().expect("HIP starless buffers allocated");
        let total_bytes = total * 4;
        let mask_bytes = np * 4;
        for (dst, src, len, label) in [
            (b.orig.ptr, orig_values.as_ptr().cast(), total_bytes, "HIP starless orig upload"),
            (b.ping.ptr, seed_values.as_ptr().cast(), total_bytes, "HIP starless seed upload"),
            (b.star_mask.ptr, star_mask.as_ptr().cast(), mask_bytes, "HIP starless mask upload"),
            (b.valid_mask.ptr, valid_mask.as_ptr().cast(), mask_bytes, "HIP starless valid upload"),
        ] {
            self.api
                .check(unsafe { (self.api.memcpy)(dst, src, len, 1) }, label)?;
        }
        let blocks = u32::try_from(total.div_ceil(THREADS))
            .context("HIP starless dispatch exceeds u32 workgroups")?;
        let mut arg_orig = b.orig.ptr;
        let mut arg_star_mask = b.star_mask.ptr;
        let mut arg_valid_mask = b.valid_mask.ptr;
        let mut arg_w = width as u32;
        let mut arg_h = height as u32;
        let mut arg_c = channels as u32;
        for (iter, &step) in crate::stars::STARLESS_DIFFUSION_STEPS.iter().enumerate() {
            let (mut arg_src, mut arg_dst) = if iter % 2 == 0 {
                (b.ping.ptr, b.pong.ptr)
            } else {
                (b.pong.ptr, b.ping.ptr)
            };
            let mut arg_step = step;
            let mut args: [*mut c_void; 9] = [
                (&mut arg_orig as *mut DevicePtr).cast(),
                (&mut arg_src as *mut DevicePtr).cast(),
                (&mut arg_star_mask as *mut DevicePtr).cast(),
                (&mut arg_valid_mask as *mut DevicePtr).cast(),
                (&mut arg_dst as *mut DevicePtr).cast(),
                (&mut arg_w as *mut u32).cast(),
                (&mut arg_h as *mut u32).cast(),
                (&mut arg_c as *mut u32).cast(),
                (&mut arg_step as *mut u32).cast(),
            ];
            self.api.check(
                unsafe {
                    (self.api.launch)(
                        self.starless_kernel,
                        blocks,
                        1,
                        1,
                        THREADS as u32,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        args.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "hipModuleLaunchKernel (starless_diffuse_step)",
            )?;
        }
        self.api.check(
            unsafe { (self.api.synchronize)() },
            "hipDeviceSynchronize (starless_diffuse_step)",
        )?;
        let mut out = vec![0.0f32; total];
        self.api.check(
            unsafe { (self.api.memcpy)(out.as_mut_ptr().cast(), b.ping.ptr, total_bytes, 2) },
            "HIP starless readback",
        )?;
        Ok(out)
    }
}

impl Drop for HipStacker {
    fn drop(&mut self) {
        // Set the owning device before releasing allocations (Drop may run on
        // a different worker thread). Preserve the libraries until after cleanup.
        unsafe {
            (self.api.set_device)(0);
        }
        self.buffers = None;
        self.starless_buffers = None;
        if !self.module.is_null() {
            unsafe {
                (self.api.module_unload)(self.module);
            }
        }
    }
}
