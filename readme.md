# Hyperdither.py

Simple Atkinson style dithering using Python.

To speed up the processing, this script requires numpy and numba. This fork has been updated both for modern Python programming standards (IE, Typehinting and Ruff formatting) and has been updated to function under Python 3.14.

## Example

Input image:

![cat](images/cat.jpg)

Command:

    hyperdither.py -1 cat.jpg

Output:

![out cat](images/cat_out.png)

Reduced contrast:

    hyperdither.py -1 cat.jpg -c .85 -o cat_out2.png

![out cat](images/cat_out2.png)

## Options

- `-1`: 1-bit black and white file.  Otherwise the output is encoded as a grayscale file.
- `-t`: the black/white threshold.  Pixels at or above it become white.  The maximum is 255.
- `-d`: dpi out output file.  Only for printing purposes.  Default is 72 dpi.
- `-e`: output filetype.  Must be a format Pillow can write.  Default is png.
- `-o`: output path, instead of `<input>_out.<ext>`.  Its extension selects the format, so it cannot be combined with `-e`.
- `-f`: overwrite the output file if it exists.  Without it, an existing output file is left alone and the run stops.
- `-b`: start at bottom left instead of top left with the dithering.
- `-c`: boost contrast by specified amount (in terms of [pillow's][pillow] [ImageEnhancers][enhance] factor: 1 is no change).
- `-s`: boost sharpness by specified amount (in terms of [pillow's][pillow] [ImageEnhancers][enhance] factor: 1 is no change).
- `-g`: tone curve applied before dithering.  Numbers above 1 lighten the midtones, which helps printers that print dark.  Default is 1.
- `-r`: resize pre-dither image on longest dimension.  Makes the image smaller only, pre-dithering.
- `-2`: double post-dither image using nearest neighbors.  Makes output image twice as large as the pre-dither image.  If image is scaled from the resized pre-dither image if the `-r` option was specified.

For contrast and sharpness, 1 is default and does not need to be specified.  Numbers larger than 1 increase sharpness/contrast, numbers below 1 reduce it.  Negative numbers for contrast invert the image, for example `-c -1`.

I found values of 0.8 to 1.2 are good for contrast if you want lighten or darken specific areas.

## Requirements

- [pillow][] 
- [numpy](http://www.numpy.org)
- [numba][]

Dependencies are declared in `pyproject.toml`.  With [uv](https://docs.astral.sh/uv/), run `uv run hyperdither.py -1 cat.jpg`, and `uv run pytest` for the tests.

[numba]: http://numba.pydata.org 
[pillow]: http://python-pillow.org
[enhance]: https://pillow.readthedocs.io/en/4.0.x/reference/ImageEnhance.html

## Links

- <http://www.tannerhelland.com/4660/dithering-eleven-algorithms-source-code/>
- <https://en.wikipedia.org/wiki/Bill_Atkinson>
