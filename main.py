from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qrcode
from PIL import Image, ImageFilter
from qrcode.constants import ERROR_CORRECT_H


def make_qr_matrix(payload: str, version: int = 3) -> np.ndarray:
    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_CORRECT_H,
        box_size=1,
        border=0,
    )
    qr.add_data(payload)
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=bool)


def expand_modules(matrix: np.ndarray, cell_px: int) -> np.ndarray:
    expanded = np.kron(matrix.astype(np.uint8), np.ones((cell_px, cell_px), dtype=np.uint8))
    return np.where(expanded, 0, 255).astype(np.uint8)


def protected_region_mask(module_count: int, preserve_timing: bool = False) -> np.ndarray:
    mask = np.zeros((module_count, module_count), dtype=bool)

    def mark(r0: int, c0: int, h: int, w: int) -> None:
        r1, c1 = max(0, r0), max(0, c0)
        r2, c2 = min(module_count, r0 + h), min(module_count, c0 + w)
        mask[r1:r2, c1:c2] = True

    # Finder patterns plus one-module separators.
    mark(-1, -1, 9, 9)
    mark(-1, module_count - 8, 9, 9)
    mark(module_count - 8, -1, 9, 9)

    # Version 3 has an alignment pattern centered at module (22, 22).
    if module_count == 29:
        mark(20, 20, 5, 5)

    if preserve_timing:
        mask[6, :] = True
        mask[:, 6] = True
    return mask


def make_refined_qr(matrix: np.ndarray, cell_px: int, point_px: int, preserve_timing: bool) -> np.ndarray:
    module_count = matrix.shape[0]
    protected = protected_region_mask(module_count, preserve_timing)
    refined = np.full((module_count * cell_px, module_count * cell_px), 255, dtype=np.uint8)
    half = point_px // 2

    for r in range(module_count):
        for c in range(module_count):
            value = 0 if matrix[r, c] else 255
            y0, x0 = r * cell_px, c * cell_px
            if protected[r, c]:
                refined[y0 : y0 + cell_px, x0 : x0 + cell_px] = value
            else:
                cy = y0 + cell_px // 2
                cx = x0 + cell_px // 2
                refined[cy - half : cy - half + point_px, cx - half : cx - half + point_px] = value
    return refined


def gaussian_texture(size: int, mean: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = rng.normal(mean, sigma, (size, size))
    return np.clip(texture, 0, 255).astype(np.uint8)


def bilinear_resize(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(image, mode="L")
    return np.array(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def floyd_steinberg_halftone(image: np.ndarray) -> np.ndarray:
    work = image.astype(np.float32).copy()
    h, w = work.shape
    for y in range(h):
        for x in range(w):
            old = work[y, x]
            new = 255.0 if old >= 128.0 else 0.0
            work[y, x] = new
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * 7 / 16
            if y + 1 < h and x > 0:
                work[y + 1, x - 1] += err * 3 / 16
            if y + 1 < h:
                work[y + 1, x] += err * 5 / 16
            if y + 1 < h and x + 1 < w:
                work[y + 1, x + 1] += err * 1 / 16
    return np.clip(work, 0, 255).astype(np.uint8)


def fuse_texture_and_qr(
    matrix: np.ndarray,
    halftone: np.ndarray,
    cell_px: int,
    point_px: int,
    preserve_timing: bool,
) -> np.ndarray:
    module_count = matrix.shape[0]
    protected = protected_region_mask(module_count, preserve_timing)
    fused = halftone.copy()
    half = point_px // 2

    for r in range(module_count):
        for c in range(module_count):
            value = 0 if matrix[r, c] else 255
            y0, x0 = r * cell_px, c * cell_px
            if protected[r, c]:
                fused[y0 : y0 + cell_px, x0 : x0 + cell_px] = value
            else:
                cy = y0 + cell_px // 2
                cx = x0 + cell_px // 2
                fused[cy - half : cy - half + point_px, cx - half : cx - half + point_px] = value
    return fused


def add_quiet_zone(image: np.ndarray, cell_px: int, modules: int = 4) -> np.ndarray:
    border = cell_px * modules
    return np.pad(image, ((border, border), (border, border)), constant_values=255).astype(np.uint8)


def spectrum(image: np.ndarray) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))
    spec = np.log1p(np.abs(fft))
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
    return (spec * 255).astype(np.uint8)


def simulate_copy(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image, mode="L")
    small = pil.resize((image.shape[1] // 2, image.shape[0] // 2), Image.Resampling.BILINEAR)
    copied = small.resize((image.shape[1], image.shape[0]), Image.Resampling.BILINEAR)
    copied = copied.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.array(copied, dtype=np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)


def save_generation_montage(
    path: Path,
    original_qr: np.ndarray,
    refined_qr: np.ndarray,
    gaussian: np.ndarray,
    interpolated: np.ndarray,
    halftone: np.ndarray,
    fused: np.ndarray,
) -> None:
    items = [
        ("Original QR code", original_qr),
        ("Refined QR code", refined_qr),
        ("Gaussian texture", gaussian),
        ("Bilinear texture", interpolated),
        ("Halftone texture", halftone),
        ("Texture-hidden QR", fused),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, image) in zip(axes.ravel(), items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_spectrum_montage(path: Path, fused: np.ndarray, copied: np.ndarray) -> None:
    items = [
        ("Generated code", fused),
        ("Generated spectrum", spectrum(fused)),
        ("Simulated copied code", copied),
        ("Copied spectrum", spectrum(copied)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (title, image) in zip(axes.ravel(), items):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_outputs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = make_qr_matrix(args.payload, args.version)
    module_count = matrix.shape[0]
    target_size = module_count * args.cell_px

    original_qr = expand_modules(matrix, args.cell_px)
    refined_qr = make_refined_qr(matrix, args.cell_px, args.point_px, args.preserve_timing)
    raw_texture = gaussian_texture(args.texture_size, args.gaussian_mean, args.gaussian_sigma, args.seed)
    interpolated = bilinear_resize(raw_texture, target_size)
    halftone = floyd_steinberg_halftone(interpolated)
    fused = fuse_texture_and_qr(matrix, halftone, args.cell_px, args.point_px, args.preserve_timing)
    fused_quiet = add_quiet_zone(fused, args.cell_px)
    copied = simulate_copy(fused)

    save_image(output_dir / "01_original_qr.png", original_qr)
    save_image(output_dir / "02_refined_qr.png", refined_qr)
    save_image(output_dir / "03_gaussian_texture.png", raw_texture)
    save_image(output_dir / "04_bilinear_texture.png", interpolated)
    save_image(output_dir / "05_halftone_texture.png", halftone)
    save_image(output_dir / "06_texture_hidden_qr.png", fused)
    save_image(output_dir / "07_texture_hidden_qr_with_quiet_zone.png", fused_quiet)
    save_image(output_dir / "08_simulated_copied_qr.png", copied)
    save_image(output_dir / "09_texture_hidden_qr_spectrum.png", spectrum(fused))
    save_image(output_dir / "10_simulated_copied_qr_spectrum.png", spectrum(copied))
    save_generation_montage(output_dir / "generation_process_montage.png", original_qr, refined_qr, raw_texture, interpolated, halftone, fused)
    save_spectrum_montage(output_dir / "dft_spectrum_comparison.png", fused, copied)

    print(f"Payload: {args.payload}")
    print(f"QR version: {args.version}, modules: {module_count} x {module_count}")
    print(f"Image size: {target_size} x {target_size}px")
    print(f"Code point size: {args.point_px} x {args.point_px}px")
    print(f"Gaussian mean/sigma: {args.gaussian_mean}/{args.gaussian_sigma}")
    print(f"Preserve timing pattern: {args.preserve_timing}")
    print(f"Wrote outputs to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the texture-hidden anti-counterfeiting QR generation process.")
    parser.add_argument("--payload", default="TextureQR-CDP1-20260708", help="Semantic payload encoded by the QR code.")
    parser.add_argument("--version", type=int, default=3, help="QR version. The paper uses version 3 in its parameter experiments.")
    parser.add_argument("--cell-px", type=int, default=20, help="Pixel size of one QR module. Version 3 with 20px modules yields 580px.")
    parser.add_argument("--point-px", type=int, default=5, help="Reduced code point size, matching the paper's selected 5x5 setting.")
    parser.add_argument("--texture-size", type=int, default=145, help="Initial Gaussian texture size before bilinear upsampling.")
    parser.add_argument("--gaussian-mean", type=float, default=120.0, help="Gaussian texture mean, matching the paper's selected mu=120.")
    parser.add_argument("--gaussian-sigma", type=float, default=50.0, help="Gaussian texture standard deviation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable texture generation.")
    parser.add_argument("--output-dir", default="outputs/texture_qr_generation", help="Directory for generated images.")
    parser.add_argument("--preserve-timing", action="store_true", help="Keep timing patterns as full QR modules. Useful for software decoder checks, less visually hidden.")
    return parser.parse_args()


if __name__ == "__main__":
    build_outputs(parse_args())
