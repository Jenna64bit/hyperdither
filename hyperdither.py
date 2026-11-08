#!/usr/bin/env python
import argparse
import math
import os
import re
import sys

import numba
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ENHANCE_DEFAULT = 1.0
# pins: ESC * mode, dots per pass, horizontal dpi, vertical dpi
ESCP_MODES = {9: (1, 8, 120, 72), 24: (39, 24, 180, 180)}
VALUE_OPTIONS = {"-c", "--contrast", "-s", "--sharpness"}
NEGATIVE_NUMBER = re.compile(r"-(\d+\.?\d*|\.\d+)")


@numba.jit(cache=True)
def dither(num: np.ndarray, thresh: int = 127) -> np.ndarray:
    """
    Applies Atkinson's style dithering process to a given intensity image represented as a NumPy array.

    The function runs an error diffusion filter in a single pass (using neighbor information)
    to spread quantization errors, resulting in a more visually pleasing dithered output when
    converting continuous tones to black and white (1-bit).

    Args:
        num (np.ndarray): The input intensity array of shape (height, width), containing
                           pixel values between 0 and 255. This array is modified in place
                           to hold the dithered output.
        thresh (int, optional): The black/white threshold value. Values at or above this become white
                                 and values below it become black. Defaults to 127.

    Returns:
        np.ndarray: The final dithered array, scaled back to the full 0-255 range and reversed along axis 0 (rows).
    """
    derr = np.zeros(tuple(num.shape), dtype=np.float64)

    div = 8.0
    for y in range(num.shape[0]):
        for x in range(num.shape[1]):
            newval = derr[y, x] + num[y, x]
            if newval >= thresh:
                errval = newval - 255
                num[y, x] = 1.0
            else:
                errval = newval
                num[y, x] = 0.0
            if x + 1 < num.shape[1]:
                derr[y, x + 1] += errval / div
                if x + 2 < num.shape[1]:
                    derr[y, x + 2] += errval / div
            if y + 1 < num.shape[0]:
                if x > 0:
                    derr[y + 1, x - 1] += errval / div
                derr[y + 1, x] += errval / div
                if y + 2 < num.shape[0]:
                    derr[y + 2, x] += errval / div
                if x + 1 < num.shape[1]:
                    derr[y + 1, x + 1] += errval / div
    return num[::-1, :] * 255


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
    if Image.EXTENSION.get(ext) not in Image.SAVE:
        parser.error(f"cannot write files with extension '{ext}'")
    if not os.path.isdir(os.path.dirname(outfn) or "."):
        parser.error(f"output directory does not exist: {outfn}")
    if os.path.exists(outfn) and not args.force:
        parser.error(f"{outfn} exists; pass -f to overwrite it")
    return outfn


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
            stream += bytes((0x1b, 0x2a, mode, count & 255, count >> 8))
            stream += columns[:count].tobytes()
        stream += b"\r\n"
    stream += b"\x1b@"
    return bytes(stream)


def resolve_printing(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> str | None:
    """Return the print target, or None for file output, refusing an unusable one before any work."""
    if args.print_to is None:
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
    args.pins = args.pins or 9
    args.page_width = 8.0 if args.page_width is None else args.page_width
    if not (math.isfinite(args.page_width) and args.page_width > 0):
        parser.error("--page-width must be a positive number of inches")
    if args.print_to == "-":
        return "-"
    device = os.path.expanduser(args.print_to)
    if os.path.exists(device):
        if os.path.isfile(device) and not args.force:
            parser.error(f"{device} exists; pass -f to overwrite it")
        if not os.access(device, os.W_OK):
            parser.error(f"cannot write to {device}; on Linux your user usually needs the lp group")
    elif device.startswith("/dev/") or not os.path.isdir(os.path.dirname(device) or "."):
        parser.error(f"no such device: {device}")
    return device


def fit_to_page(img: Image.Image, pins: int, inches: float) -> Image.Image:
    """Resize to the page in printer dots, so non-square dots still print the right shape."""
    _, _, x_dpi, y_dpi = ESCP_MODES[pins]
    columns = round(inches * x_dpi)
    rows = max(1, round(columns * img.height / img.width * y_dpi / x_dpi))
    return img.resize((columns, rows), Image.Resampling.LANCZOS)


def load_gray(filename: str) -> Image.Image:
    """Open an image upright, with transparency on white paper and 16-bit data scaled to 8."""
    img = ImageOps.exif_transpose(Image.open(filename))
    if img.mode.startswith("I"):
        return Image.fromarray(np.uint8(np.clip(np.asarray(img) / 257, 0, 255)))
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return img.convert("L")


def main(argv: list[str] | None = None) -> None:
    """
    Main function to parse command-line arguments and orchestrate the dithering process.

    This function initializes argparse with all necessary options:

    Args:
        argv (list[str], optional): Arguments without the program name (defaults to sys.argv[1:]).

    Returns:
        None: Executes the program logic and handles file output/exit.

    Parser Arguments Summary:
        filename (Positional, required): The path to the image file to process.
        --mono (-1) (bool): If set, saves the output as a 1-bit black and white image.
        --threshold (-t) (int): The intensity threshold for dithering (default 127).
        --dpi (-d) (int): DPI setting for output files (default 72).
        --ext (-e) (str): Output file extension type (default png).
        --output (-o) (str): Output path, instead of <input>_out.<ext>.
        --force (-f) (bool): Overwrite the output file if it exists.
        --bottom (-b) (bool): Processes the image starting from the bottom left.
        --contrast (-c) (float): Contrast boost factor (1 is no change, negative for inversion).
        --sharpness (-s) (float): Sharpness boost factor (1 is no change).
        --print (str): Send ESC/P graphics to a device, or - for stdout, instead of writing an image.
        --pins (int): Print head pins with --print, 9 or 24 (default 9).
        --page-width (float): Printed width in inches with --print (default 8).
        --gamma (-g) (float): Tone curve before dithering; above 1 lightens (default 1).
        --resize (-r) (int): Resizes the image on its longest dimension before dithering.
        --double (-2) (bool): Doubles the output resolution using nearest neighbors.
    """
    parser = argparse.ArgumentParser(
        description="Dither image an image using Atkinson 'hyperdither'."
    )
    parser.add_argument("filename")
    parser.add_argument(
        "-1",
        "--mono",
        dest="mono",
        action="store_true",
        default=False,
        help="1-bit black and white file",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        type=int,
        action="store",
        default=127,
        help="pixels at or above this become white (default 127)",
    )
    parser.add_argument(
        "-d",
        "--dpi",
        dest="dpi",
        type=int,
        action="store",
        default=None,
        help="dpi of output file (default 72)",
    )
    parser.add_argument(
        "--print",
        dest="print_to",
        metavar="DEVICE",
        default=None,
        help="send Epson ESC/P graphics to DEVICE, or - for stdout, instead of writing an image",
    )
    parser.add_argument(
        "--pins",
        dest="pins",
        type=int,
        choices=sorted(ESCP_MODES),
        default=None,
        help="print head pins, with --print (default 9)",
    )
    parser.add_argument(
        "--page-width",
        dest="page_width",
        type=float,
        metavar="INCHES",
        default=None,
        help="printed width in inches, with --print (default 8)",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-e",
        "--ext",
        dest="ext",
        action="store",
        default=None,
        help="output filetype (default png)",
    )
    target.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        default=None,
        help="output path (default <input>_out.<ext>)",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="overwrite the output file if it exists",
    )
    parser.add_argument(
        "-b",
        "--bottom",
        dest="bottom",
        action="store_true",
        default=False,
        help="start at bottom left instead of top left",
    )
    parser.add_argument(
        "-c",
        "--contrast",
        dest="contrast",
        action="store",
        default=ENHANCE_DEFAULT,
        type=float,
        help="boost contrast by specified factor (default 1)",
    )
    parser.add_argument(
        "-s",
        "--sharpness",
        dest="sharpness",
        action="store",
        default=ENHANCE_DEFAULT,
        type=float,
        help="boost sharpness by specified factor (default 1)",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        dest="gamma",
        action="store",
        default=1.0,
        type=float,
        help="tone curve applied before dithering; above 1 lightens (default 1)",
    )
    parser.add_argument(
        "-r",
        "--resize",
        dest="resize",
        action="store",
        default=None,
        type=int,
        help="""resize pre-dither image on longest dimension""",
    )
    parser.add_argument(
        "-2",
        "--double",
        dest="double",
        action="store_true",
        default=False,
        help="double post-dither image using nearest neighbors",
    )

    args = parser.parse_args(
        attach_negative_values(sys.argv[1:] if argv is None else argv)
    )

    if not (math.isfinite(args.gamma) and args.gamma > 0):
        parser.error("gamma must be a positive number")
    filename = os.path.expanduser(args.filename)
    if not os.path.isfile(filename):
        print("Must supply a valid file.")
        sys.exit(1)
    device = resolve_printing(args, parser)
    outfn = None if device else resolve_output(args, filename, parser)
    img = load_gray(filename)
    if args.contrast != ENHANCE_DEFAULT:
        img = ImageEnhance.Contrast(img).enhance(args.contrast)
    if device:
        img = fit_to_page(img, args.pins, args.page_width)
    if args.resize:
        img.thumbnail((args.resize,) * 2, Image.Resampling.BICUBIC)
    if args.sharpness != ENHANCE_DEFAULT:
        img = ImageEnhance.Sharpness(img).enhance(args.sharpness)
    if args.gamma != 1:
        img = img.point([round(255 * (v / 255) ** (1 / args.gamma)) for v in range(256)])

    if args.bottom:
        m = np.array(img)[::-1, :]
        m2 = dither(m, thresh=args.threshold)
        out = Image.fromarray(m2[:, :].astype(np.uint8))
    else:
        m = np.array(img)[:, :]
        m2 = dither(m, thresh=args.threshold)
        out = Image.fromarray(m2[::-1, :].astype(np.uint8))
    if device:
        stream = escp_stream(np.array(out) == 0, args.pins)
        if device == "-":
            sys.stdout.buffer.write(stream)
            sys.stdout.buffer.flush()
        else:
            with open(device, "wb") as printer:
                printer.write(stream)
        return
    if args.double:
        out = out.resize((out.width * 2, out.height * 2), Image.Resampling.NEAREST)
    dpi = (args.dpi or 72,) * 2
    if args.mono:
        out.convert("1", dither=Image.Dither.NONE).save(outfn, dpi=dpi)
    else:
        out.save(outfn, dpi=dpi)


if __name__ == "__main__":
    sys.exit(main())
