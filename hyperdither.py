#!/usr/bin/env python
import argparse
import io
import math
import os
import re
import sys
from dataclasses import dataclass

import numba
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ENHANCE_DEFAULT = 1.0
# pins: ESC * mode, dots per pass, horizontal dpi, vertical dpi
ESCP_MODES = {9: (1, 8, 120, 72), 24: (39, 24, 180, 180)}
VALUE_OPTIONS = {"-c", "--contrast", "-s", "--sharpness"}
NEGATIVE_NUMBER = re.compile(r"-(\d+\.?\d*|\.\d+)")
DEVICE_DIR = "/dev/"


@numba.jit(cache=True)
def dither(num: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Atkinson-dither an 8-bit array in one pass, in place.

    thresh: values at or above this become white (255), the rest black (0).
    """
    height, width = num.shape
    stride = width + 3
    # Pad columns and rows absorb error that leaves the image, so no edge checks.
    carried = np.zeros((height + 2) * stride)
    i = 1
    for y in range(height):
        for x in range(width):
            value = carried[i] + num[y, x]
            if value >= thresh:
                num[y, x] = 255
                share = (value - 255.0) / 8.0
            else:
                num[y, x] = 0
                share = value / 8.0
            carried[i + 1] += share
            carried[i + 2] += share
            carried[i + stride - 1] += share
            carried[i + stride] += share
            carried[i + stride + 1] += share
            carried[i + 2 * stride] += share
            i += 1
        i += 3
    return num


def escp_stream(dots: np.ndarray, pins: int) -> bytes:
    """Encode a boolean array, True where a dot fires, as unidirectional ESC/P bit-image passes."""
    mode, pass_height, _, _ = ESCP_MODES[pins]
    stream = bytearray(b"\x1b@\x1bU\x01\x1b3\x18")
    for top in range(0, dots.shape[0], pass_height):
        band = np.zeros((pass_height, dots.shape[1]), bool)
        rows = dots[top : top + pass_height]
        band[: rows.shape[0]] = rows
        columns = np.packbits(band, axis=0).T
        used = np.flatnonzero(columns.any(axis=1))
        if used.size:
            count = int(used[-1]) + 1
            stream += bytes((0x1B, 0x2A, mode, count & 255, count >> 8))
            stream += columns[:count].tobytes()
        stream += b"\r\n"
    stream += b"\x1b@"
    return bytes(stream)


@dataclass(frozen=True)
class FileTarget:
    """Write the dithered image to an image file."""

    path: str
    resize: int | None
    double: bool
    mono: bool
    dpi: int

    def fit(self, img: Image.Image) -> Image.Image:
        if self.resize:
            img.thumbnail((self.resize, self.resize), Image.Resampling.BICUBIC)
        return img

    def send(self, out: Image.Image) -> None:
        if self.double:
            out = out.resize((out.width * 2, out.height * 2), Image.Resampling.NEAREST)
        if self.mono:
            out = out.convert("1", dither=Image.Dither.NONE)
        out.save(self.path, dpi=(self.dpi, self.dpi))


@dataclass(frozen=True)
class PrintTarget:
    """Send the dithered image to a dot-matrix printer as ESC/P graphics."""

    device: str
    pins: int
    inches: float

    def fit(self, img: Image.Image) -> Image.Image:
        _, _, x_dpi, y_dpi = ESCP_MODES[self.pins]
        columns = round(self.inches * x_dpi)
        rows = max(1, round(columns * img.height / img.width * y_dpi / x_dpi))
        return img.resize((columns, rows), Image.Resampling.LANCZOS)

    def send(self, out: Image.Image) -> None:
        stream = escp_stream(np.array(out) == 0, self.pins)
        if self.device == "-":
            sys.stdout.buffer.write(stream)
            sys.stdout.buffer.flush()
        else:
            with open(self.device, "wb") as printer:
                printer.write(stream)


def positive(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text} is not a positive number")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dither an image using Atkinson 'hyperdither'."
    )
    parser.add_argument("filename")
    parser.add_argument(
        "-1", "--mono", action="store_true", help="1-bit black and white file"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=127,
        help="pixels at or above this become white (default 127)",
    )
    parser.add_argument(
        "-d", "--dpi", type=positive_int, help="dpi of output file (default 72)"
    )
    parser.add_argument(
        "--print",
        metavar="DEVICE",
        help="send Epson ESC/P graphics to DEVICE, or - for stdout, instead of writing an image",
    )
    parser.add_argument(
        "--pins",
        type=int,
        choices=sorted(ESCP_MODES),
        help="print head pins, with --print (default 9)",
    )
    parser.add_argument(
        "--page-width",
        type=positive,
        metavar="INCHES",
        help="printed width in inches, with --print (default 8)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("-e", "--ext", help="output filetype (default png)")
    output.add_argument(
        "-o", "--output", help="output path (default <input>_out.<ext>)"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite the output file if it exists",
    )
    parser.add_argument(
        "-b",
        "--bottom",
        action="store_true",
        help="start at bottom left instead of top left",
    )
    parser.add_argument(
        "-c",
        "--contrast",
        type=float,
        default=ENHANCE_DEFAULT,
        help="boost contrast by specified factor (default 1)",
    )
    parser.add_argument(
        "-s",
        "--sharpness",
        type=float,
        default=ENHANCE_DEFAULT,
        help="boost sharpness by specified factor (default 1)",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=positive,
        default=ENHANCE_DEFAULT,
        help="tone curve applied before dithering; above 1 lightens (default 1)",
    )
    parser.add_argument(
        "-r",
        "--resize",
        type=positive_int,
        help="resize pre-dither image on longest dimension",
    )
    parser.add_argument(
        "-2",
        "--double",
        action="store_true",
        help="double post-dither image using nearest neighbors",
    )
    return parser


def attach_negative_values(argv: list[str]) -> list[str]:
    """Join `-c -1` into `-c=-1`, because the -1 flag makes argparse read negative numbers as options."""
    joined: list[str] = []
    for token in argv:
        if joined and joined[-1] in VALUE_OPTIONS and NEGATIVE_NUMBER.fullmatch(token):
            joined[-1] = f"{joined[-1]}={token}"
        else:
            joined.append(token)
    return joined


def resolve_output(
    args: argparse.Namespace, filename: str, parser: argparse.ArgumentParser
) -> str:
    """Choose the output path and refuse an unusable one before any image work."""
    if args.output:
        outfn = os.path.expanduser(args.output)
    else:
        outfn = os.path.splitext(filename)[0] + "_out." + (args.ext or "png")
    Image.init()
    ext = os.path.splitext(outfn)[1].lower()
    if not format_can_write(ext, "1" if args.mono else "L"):
        parser.error(f"cannot write files with extension '{ext}'")
    if not os.access(os.path.dirname(outfn) or ".", os.W_OK):
        parser.error(f"output directory is missing or read-only: {outfn}")
    if os.path.exists(outfn):
        if not args.force:
            parser.error(f"{outfn} exists; pass -f to overwrite it")
        if not os.access(outfn, os.W_OK):
            parser.error(f"cannot overwrite read-only file: {outfn}")
    return outfn


def format_can_write(ext: str, mode: str) -> bool:
    """Try a small save, because a format listed as writable can still reject a mode."""
    image_format = Image.EXTENSION.get(ext)
    if image_format not in Image.SAVE:
        return False
    try:
        Image.new(mode, (16, 16)).save(io.BytesIO(), format=image_format)
    except (OSError, ValueError):
        return False
    return True


def resolve_printing(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> PrintTarget | None:
    """Return the print target, or None for file output, refusing an unusable one before any work."""
    if args.print is None:
        if args.pins is not None or args.page_width is not None:
            parser.error("--pins and --page-width need --print")
        return None
    file_only = {
        "-o": args.output,
        "-e": args.ext,
        "-2": args.double,
        "-r": args.resize,
        "-d": args.dpi,
    }
    conflicts = [flag for flag, value in file_only.items() if value]
    if conflicts:
        parser.error(f"--print cannot be combined with {' '.join(conflicts)}")
    device = "-" if args.print == "-" else os.path.expanduser(args.print)
    if device.startswith(DEVICE_DIR) and os.path.isfile(device):
        parser.error(
            f"{device} is a regular file, not a printer device; "
            "remove it with sudo rm, then reconnect the printer"
        )
    if os.path.exists(device):
        if os.path.isfile(device) and not args.force:
            parser.error(f"{device} exists; pass -f to overwrite it")
        if not os.access(device, os.W_OK):
            parser.error(
                f"cannot write to {device}; on Linux your user usually needs the lp group"
            )
    elif device != "-" and (
        device.startswith(DEVICE_DIR)
        or not os.access(os.path.dirname(device) or ".", os.W_OK)
    ):
        parser.error(f"no such device: {device}")
    return PrintTarget(device, args.pins or 9, args.page_width or 8.0)


def resolve_target(
    args: argparse.Namespace, filename: str, parser: argparse.ArgumentParser
) -> FileTarget | PrintTarget:
    return resolve_printing(args, parser) or FileTarget(
        resolve_output(args, filename, parser),
        args.resize,
        args.double,
        args.mono,
        args.dpi or 72,
    )


def load_gray(filename: str) -> Image.Image:
    """Open an image upright, with transparency on white paper and 16-bit data scaled to 8."""
    img = ImageOps.exif_transpose(Image.open(filename))
    if img.mode in ("I", "F"):
        raise OSError(f"pixel mode {img.mode} has no known brightness range")
    if img.mode.startswith("I;16"):
        return Image.fromarray(np.uint8(np.clip(np.asarray(img) / 257, 0, 255)))
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return img.convert("L")


def prepare(
    img: Image.Image, args: argparse.Namespace, target: FileTarget | PrintTarget
) -> Image.Image:
    """Apply contrast, sizing, sharpness and gamma, in that order."""
    if args.contrast != ENHANCE_DEFAULT:
        img = ImageEnhance.Contrast(img).enhance(args.contrast)
    img = target.fit(img)
    if args.sharpness != ENHANCE_DEFAULT:
        img = ImageEnhance.Sharpness(img).enhance(args.sharpness)
    if args.gamma != ENHANCE_DEFAULT:
        img = img.point(
            [round(255 * (v / 255) ** (1 / args.gamma)) for v in range(256)]
        )
    return img


def main(argv: list[str] | None = None) -> None:
    """Dither one image and send it to a file or a printer."""
    parser = build_parser()
    args = parser.parse_args(
        attach_negative_values(sys.argv[1:] if argv is None else argv)
    )
    filename = os.path.expanduser(args.filename)
    if not os.path.isfile(filename):
        print("Must supply a valid file.", file=sys.stderr)
        sys.exit(1)
    target = resolve_target(args, filename, parser)
    try:
        img = load_gray(filename)
    except (OSError, Image.DecompressionBombError) as error:
        parser.error(f"cannot read {filename}: {error}")
    scan = -1 if args.bottom else 1
    gray = np.array(prepare(img, args, target))[::scan]
    target.send(Image.fromarray(dither(gray, args.threshold)[::scan]))


if __name__ == "__main__":
    sys.exit(main())
