import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
except ImportError as exc:
    sys.stderr.write(
        f"Missing optional Python dependency for --restore ({exc}).\n"
        "Install optional dependencies with: pip install torch numpy Pillow tifffile\n"
        "And install verified model weights with: photonloom models install\n"
    )
    sys.exit(2)

EXPECTED_SHARPEN_SHA256 = "c480bdf8522b2e3797525217cad9e04058c59b2a32eb8f7d1f7e90d99aafb9be"
EXPECTED_DENOISE_SHA256 = "9867c7d822223c922d37a92b6f2d1713d37f943d963b8cfbef43fe21918b405c"


def verify_sha256(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weight file '{path}' not found. Run `photonloom models install` first."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest().lower()
    if digest != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for '{path}': expected {expected}, got {digest}"
        )
    return digest


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class DeepResidualCNN(nn.Module):
    """Matches Cosmic Clarity deep_sharp_stellar_cnn.pth architecture (revision 423acc442724)."""

    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        d5 = self.decoder5(torch.cat([e5, e4], dim=1))
        d4 = self.decoder4(torch.cat([d5, e3], dim=1))
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))
        return self.decoder1(d2)


class SequentialCNN(nn.Module):
    """Matches Cosmic Clarity denoise_cnn.pth architecture (revision 423acc442724)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def select_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        label = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return torch.device("cuda"), label
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def load_verified_model(weight_path: Path, expected_sha: str, device: torch.device) -> nn.Module:
    verify_sha256(weight_path, expected_sha)
    state_dict = torch.load(str(weight_path), map_location=device, weights_only=True)
    if "encoder1.0.weight" in state_dict:
        model: nn.Module = DeepResidualCNN()
    elif "encoder.0.weight" in state_dict:
        model = SequentialCNN()
    else:
        raise ValueError(f"Unrecognized state_dict structure in '{weight_path}'")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_feather_window(h: int, w: int, overlap: int) -> np.ndarray:
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    ov_y = min(overlap, h // 2)
    ov_x = min(overlap, w // 2)
    if ov_y > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, ov_y + 1, dtype=np.float32) / (ov_y + 1)))
        wy[:ov_y] = ramp
        wy[-ov_y:] = ramp[::-1]
    if ov_x > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, ov_x + 1, dtype=np.float32) / (ov_x + 1)))
        wx[:ov_x] = ramp
        wx[-ov_x:] = ramp[::-1]
    return np.outer(wy, wx)[None, :, :]


def run_tiled_inference(
    model: nn.Module,
    chw_img: np.ndarray,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    _, h, w = chw_img.shape
    stride = max(tile_size - overlap, 32)
    acc = np.zeros_like(chw_img, dtype=np.float32)
    w_acc = np.zeros((1, h, w), dtype=np.float32)

    ys = list(range(0, max(h - tile_size, 0) + 1, stride))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    with torch.no_grad():
        for y0 in ys:
            y1 = min(y0 + tile_size, h)
            th = y1 - y0
            for x0 in xs:
                x1 = min(x0 + tile_size, w)
                tw = x1 - x0
                patch = chw_img[:, y0:y1, x0:x1]
                tensor_in = torch.from_numpy(patch).unsqueeze(0).to(device)
                pred = model(tensor_in).squeeze(0).detach().cpu().numpy()
                win = build_feather_window(th, tw, overlap)
                acc[:, y0:y1, x0:x1] += pred * win
                w_acc[:, y0:y1, x0:x1] += win

    return np.clip(acc / np.maximum(w_acc, 1e-6), 0.0, 1.0)


def read_rgb_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except ImportError:
        with Image.open(path) as im:
            arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return np.clip(np.transpose(arr, (2, 0, 1)), 0.0, 1.0)


def write_outputs(chw_img: np.ndarray, out_tiff: Path, out_png: Path) -> None:
    hwc = np.clip(np.transpose(chw_img, (1, 2, 0)), 0.0, 1.0)
    u16_arr = np.clip(hwc * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)

    try:
        import tifffile

        tifffile.imwrite(str(out_tiff), u16_arr, photometric="rgb")
    except ImportError:
        # Fallback if tifffile not installed: write 8-bit RGB via Pillow
        u8_tiff = np.clip(hwc * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(u8_tiff, mode="RGB").save(str(out_tiff), format="TIFF")

    # Convert linear/stretched [0,1] to sRGB u8 PNG preview
    srgb = np.where(
        hwc <= 0.0031308,
        12.92 * hwc,
        1.055 * np.power(np.maximum(hwc, 0.0), 1.0 / 2.4) - 0.055,
    )
    u8_png = np.clip(srgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(u8_png, mode="RGB").save(str(out_png), format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Photonloom Cosmic Clarity restoration helper")
    parser.add_argument("--input-tiff", required=True, type=Path)
    parser.add_argument("--output-tiff", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--sharpen-weights", required=True, type=Path)
    parser.add_argument("--denoise-weights", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=0.35)
    args = parser.parse_args()

    strength = float(np.clip(args.strength, 0.05, 0.65))
    device, device_label = select_device()

    sharpen_model = load_verified_model(args.sharpen_weights, EXPECTED_SHARPEN_SHA256, device)
    denoise_model = load_verified_model(args.denoise_weights, EXPECTED_DENOISE_SHA256, device)

    orig_chw = read_rgb_tiff(args.input_tiff)
    denoised = run_tiled_inference(denoise_model, orig_chw, device)
    sharpened = run_tiled_inference(sharpen_model, denoised, device)

    # Low-strength conservative blend to avoid manufactured tiny stars
    restored = np.clip((1.0 - strength) * orig_chw + strength * sharpened, 0.0, 1.0)
    write_outputs(restored, args.output_tiff, args.output_png)

    print(
        json.dumps(
            {
                "status": "ok",
                "device": device_label,
                "strength": strength,
                "sharpen_sha256": EXPECTED_SHARPEN_SHA256,
                "denoise_sha256": EXPECTED_DENOISE_SHA256,
            }
        )
    )


if __name__ == "__main__":
    main()
