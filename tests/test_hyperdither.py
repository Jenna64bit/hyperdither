import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance

import hyperdither as h

ROOT = Path(__file__).resolve().parent.parent


def third_party_imports():
    tree = ast.parse((ROOT / "hyperdither.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names)


def test_manifest_declares_every_third_party_import():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.match(r"[A-Za-z0-9_.-]+", dep).group().lower()
        for dep in manifest["project"]["dependencies"]
    }
    wanted = {
        {"PIL": "pillow"}.get(name, name).lower() for name in third_party_imports()
    }
    assert wanted and wanted <= declared


def write_source(path, size=(64, 48), seed=7):
    width, height = size
    rng = np.random.default_rng(seed)
    ramp = np.tile(np.linspace(0, 255, width), (height, 1))
    pixels = np.clip(ramp + rng.normal(0, 25, ramp.shape), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def pixels(path):
    return np.array(Image.open(path).convert("L"))


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "in.png"
    write_source(path)
    return path


def output_of(source):
    return source.with_name(source.stem + "_out.png")


def test_main_reads_the_argv_it_is_given(source, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "/nonexistent/from-sys-argv.png"])
    h.main([str(source)])
    assert output_of(source).is_file()


def test_main_argv_values_change_the_result(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    write_source(first)
    write_source(second)
    h.main([str(first), "-t", "60"])
    h.main([str(second), "-t", "200"])
    assert not np.array_equal(pixels(output_of(first)), pixels(output_of(second)))


def test_main_without_argv_reads_the_command_line(source, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", str(source)])
    h.main()
    assert output_of(source).is_file()


def kernel_reference(array, threshold=127):
    values = array.astype(float)
    height, width = values.shape
    carried = np.zeros(values.shape)
    result = np.zeros(values.shape, np.uint8)
    for y in range(height):
        for x in range(width):
            value = carried[y, x] + values[y, x]
            if value >= threshold:
                error = value - 255
                result[y, x] = 255
            else:
                error = value
            for dy, dx in ((0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= nx < width and ny < height:
                    carried[ny, nx] += error / 8
    return result


def run_kernel(array):
    return h.dither(array.copy())[::-1, :]


def test_left_edge_error_does_not_reach_the_right_edge():
    array = np.zeros((2, 4), np.uint8)
    array[0, 0] = 100
    array[1, 3] = 120
    assert run_kernel(array)[1, 3] == 0


def test_error_on_the_left_edge_changes_only_nearby_columns():
    wide = np.zeros((3, 40), np.uint8)
    wide[0, 0] = 126
    wide[1, 39] = 126
    with_left_error = run_kernel(wide)
    wide[0, 0] = 0
    without_left_error = run_kernel(wide)
    assert np.array_equal(with_left_error[:, 10:], without_left_error[:, 10:])


def test_dark_flat_gray_keeps_its_highlights():
    flat = np.full((200, 200), 32, np.uint8)
    assert np.array_equal(run_kernel(flat), kernel_reference(flat))
    assert (run_kernel(flat) == 255).any()


def test_kernel_matches_an_independent_float_reference():
    rng = np.random.default_rng(11)
    array = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    assert np.array_equal(run_kernel(array), kernel_reference(array))


def test_kernel_reference_is_sensitive_to_input():
    rng = np.random.default_rng(11)
    array = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    changed = array.copy()
    changed[0, 0] ^= 0x80
    assert not np.array_equal(kernel_reference(array), kernel_reference(changed))


def test_plain_output_is_exactly_the_dithered_bits(source):
    h.main([str(source)])
    expected = run_kernel(pixels(source))
    actual = pixels(output_of(source))
    assert np.array_equal(actual, expected)
    assert set(np.unique(actual)) == {0, 255}


def test_mono_output_is_not_dithered_a_second_time(source):
    h.main(["-1", str(source)])
    saved = Image.open(output_of(source))
    assert saved.mode == "1"
    assert np.array_equal(np.array(saved.convert("L")), run_kernel(pixels(source)))


def test_bottom_start_output_is_exactly_the_dithered_bits(source):
    h.main(["-b", str(source)])
    expected = h.dither(pixels(source)[::-1, :].copy())
    assert np.array_equal(pixels(output_of(source)), expected)


def test_double_is_nearest_neighbor(tmp_path):
    single = tmp_path / "single.png"
    doubled = tmp_path / "doubled.png"
    write_source(single)
    write_source(doubled)
    h.main([str(single)])
    h.main(["-2", str(doubled)])
    once = pixels(output_of(single))
    assert np.array_equal(
        pixels(output_of(doubled)), once.repeat(2, axis=0).repeat(2, axis=1)
    )


def test_double_with_mono_stays_nearest_neighbor(tmp_path):
    single = tmp_path / "single.png"
    doubled = tmp_path / "doubled.png"
    write_source(single)
    write_source(doubled)
    h.main(["-1", str(single)])
    h.main(["-1", "-2", str(doubled)])
    once = pixels(output_of(single))
    assert np.array_equal(
        pixels(output_of(doubled)), once.repeat(2, axis=0).repeat(2, axis=1)
    )


def enhanced_expectation(source, enhancer, factor):
    return run_kernel(
        np.array(enhancer(Image.open(source).convert("L")).enhance(factor))
    )


@pytest.mark.parametrize(
    ("args", "enhancer", "factor"),
    [
        (["-c", "-1"], ImageEnhance.Contrast, -1.0),
        (["--contrast", "-1"], ImageEnhance.Contrast, -1.0),
        (["-c", "-0.5"], ImageEnhance.Contrast, -0.5),
        (["-c", "-.5"], ImageEnhance.Contrast, -0.5),
        (["-s", "-2"], ImageEnhance.Sharpness, -2.0),
        (["--sharpness", "-2"], ImageEnhance.Sharpness, -2.0),
        (["-c-1"], ImageEnhance.Contrast, -1.0),
    ],
)
def test_negative_enhancement_values_parse(source, args, enhancer, factor):
    h.main([*args, str(source)])
    assert np.array_equal(
        pixels(output_of(source)), enhanced_expectation(source, enhancer, factor)
    )


def test_mono_flag_still_works_after_a_contrast_value(source):
    h.main(["-c", "1.2", "-1", str(source)])
    assert Image.open(output_of(source)).mode == "1"


def test_mono_flag_alone_is_not_taken_for_a_value(source):
    h.main(["-1", str(source)])
    assert Image.open(output_of(source)).mode == "1"


@pytest.mark.parametrize(
    ("flag", "enhancer"),
    [("-c", ImageEnhance.Contrast), ("-s", ImageEnhance.Sharpness)],
)
def test_zero_enhancement_is_applied(source, flag, enhancer):
    h.main([flag, "0", str(source)])
    assert np.array_equal(
        pixels(output_of(source)), enhanced_expectation(source, enhancer, 0.0)
    )


@pytest.mark.parametrize("flag", ["-c", "-s"])
def test_factor_one_equals_no_flag(tmp_path, flag):
    plain = tmp_path / "plain.png"
    explicit = tmp_path / "explicit.png"
    write_source(plain)
    write_source(explicit)
    h.main([str(plain)])
    h.main([flag, "1", str(explicit)])
    assert np.array_equal(pixels(output_of(plain)), pixels(output_of(explicit)))


def test_help_states_the_real_default(capsys):
    with pytest.raises(SystemExit):
        h.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert h.ENHANCE_DEFAULT == 1.0
    assert text.count("(default 1)") == 3


def test_literal_tilde_input_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_source(tmp_path / "in.png")
    h.main(["~/in.png"])
    assert (tmp_path / "in_out.png").is_file()


def test_literal_tilde_output_stays_out_of_a_directory_named_tilde(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    write_source(tmp_path / "in.png")
    h.main(["~/in.png"])
    assert not (tmp_path / "~").exists()


@pytest.fixture
def dither_calls(monkeypatch):
    calls = []
    real = h.dither

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(h, "dither", spy)
    return calls


def refused(args):
    with pytest.raises(SystemExit) as caught:
        h.main(args)
    return caught.value.code


def test_spy_observes_a_normal_run(source, dither_calls):
    h.main([str(source)])
    assert len(dither_calls) == 1


def test_output_flag_chooses_the_path(source, tmp_path):
    chosen = tmp_path / "chosen.png"
    h.main(["-o", str(chosen), str(source)])
    assert chosen.is_file()
    assert not output_of(source).exists()


def test_output_flag_expands_a_tilde(source, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    h.main(["-o", "~/chosen.png", str(source)])
    assert (tmp_path / "chosen.png").is_file()


def test_output_extension_selects_the_format(source, tmp_path):
    chosen = tmp_path / "chosen.bmp"
    h.main(["-o", str(chosen), str(source)])
    assert Image.open(chosen).format == "BMP"


def test_existing_output_is_kept_and_no_work_is_done(source, dither_calls, capsys):
    output_of(source).write_bytes(b"keep")
    assert refused([str(source)]) == 2
    assert output_of(source).read_bytes() == b"keep"
    assert dither_calls == []
    assert "exists" in capsys.readouterr().err


def test_force_overwrites(source):
    output_of(source).write_bytes(b"keep")
    h.main(["-f", str(source)])
    assert output_of(source).read_bytes() != b"keep"


@pytest.mark.parametrize("ext", ["nope", "psd"])
def test_unwritable_extension_is_refused_before_work(source, dither_calls, ext, capsys):
    assert refused(["-e", ext, str(source)]) == 2
    assert dither_calls == []
    assert ext in capsys.readouterr().err


def test_writable_extension_is_accepted(source):
    h.main(["-e", "bmp", str(source)])
    assert Image.open(source.with_name("in_out.bmp")).format == "BMP"


def test_unwritable_output_path_is_refused_before_work(source, tmp_path, dither_calls):
    assert refused(["-o", str(tmp_path / "chosen.nope"), str(source)]) == 2
    assert dither_calls == []


def test_missing_output_directory_is_refused_before_work(
    source, tmp_path, dither_calls
):
    assert refused(["-o", str(tmp_path / "no" / "chosen.png"), str(source)]) == 2
    assert dither_calls == []


@pytest.mark.parametrize("ext", ["png", "bmp"])
def test_output_and_ext_conflict(source, tmp_path, ext):
    assert refused(["-e", ext, "-o", str(tmp_path / "chosen.png"), str(source)]) == 2


def readme_option(flag):
    text = (ROOT / "readme.md").read_text(encoding="utf-8")
    return next(line for line in text.splitlines() if line.startswith(f"- `{flag}`"))


def dither_docstring():
    tree = ast.parse((ROOT / "hyperdither.py").read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dither"
    )
    return " ".join(ast.get_docstring(node).split())


@pytest.mark.parametrize(("value", "bit"), [(200, 255), (100, 0)])
def test_values_above_the_threshold_are_white(value, bit):
    assert h.dither(np.array([[value]], np.uint8), thresh=127)[0, 0] == bit


def test_threshold_docs_say_what_the_code_does(capsys):
    assert re.search(r"at or above.*white", readme_option("-t"))
    assert "black" not in readme_option("-t").split("threshold", 1)[1]
    assert re.search(r"at or above this become white", dither_docstring())
    with pytest.raises(SystemExit):
        h.main(["--help"])
    assert re.search(
        r"at or above this become white", " ".join(capsys.readouterr().out.split())
    )


def test_readme_names_the_real_plain_output_mode(source):
    h.main([str(source)])
    assert Image.open(output_of(source)).mode == "L"
    assert "grayscale" in readme_option("-1")
    assert "RGB" not in readme_option("-1")


def test_second_process_reuses_the_compiled_kernel(tmp_path):
    copy = tmp_path / "hyperdither.py"
    copy.write_text(
        (ROOT / "hyperdither.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    probe = (
        "import numpy as np, hyperdither as h;"
        "h.dither(np.zeros((2, 3), np.uint8));"
        "print(sum(h.dither.stats.cache_hits.values()))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    env.pop("NUMBA_CACHE_DIR", None)
    hits = [
        subprocess.run(
            [sys.executable, "-c", probe],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert hits == ["0", "1"]


def source_text():
    return (ROOT / "hyperdither.py").read_text(encoding="utf-8")


def readme_text():
    return (ROOT / "readme.md").read_text(encoding="utf-8")


def test_stale_migration_plan_is_gone():
    assert not (ROOT / "PY3_UPDATE.MD").exists()


def test_thumbnail_names_its_resample_mode():
    calls = [
        node
        for node in ast.walk(ast.parse(source_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "thumbnail"
    ]
    assert calls
    assert not any(isinstance(arg, ast.Constant) for call in calls for arg in call.args)


def test_no_commented_out_assignments():
    assert not re.search(r"^\s*#\s*\w+\s*=", source_text(), re.MULTILINE)


def test_commented_out_scan_fires_on_a_positive_fixture():
    assert re.search(r"^\s*#\s*\w+\s*=", "    # img = 1\n", re.MULTILINE)
    assert not re.search(r"^\s*#\s*\w+\s*=", "    # a plain note\n", re.MULTILINE)


def test_docstring_describes_one_pass():
    assert "multiple passes" not in dither_docstring()


def test_readme_drops_the_negative_value_workaround():
    assert '-c"-1"' not in readme_text()


def test_readme_lists_every_short_flag():
    flags = {
        arg.value
        for node in ast.walk(ast.parse(source_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and re.fullmatch(r"-\w", arg.value)
    }
    assert {"-1", "-o", "-f"} <= flags
    assert not [flag for flag in sorted(flags) if f"- `{flag}`" not in readme_text()]


@pytest.mark.parametrize(
    ("sample", "args", "command"),
    [
        ("cat_out.png", ["-1"], "hyperdither.py -1 cat.jpg"),
        (
            "cat_out2.png",
            ["-1", "-c", ".85"],
            "hyperdither.py -1 cat.jpg -c .85 -o cat_out2.png",
        ),
    ],
)
def test_committed_samples_match_their_documented_commands(
    tmp_path, sample, args, command
):
    assert command in readme_text()
    produced = tmp_path / sample
    h.main([*args, "-o", str(produced), str(ROOT / "images" / "cat.jpg")])
    assert np.array_equal(pixels(produced), pixels(ROOT / "images" / sample))


def test_exif_orientation_is_applied(tmp_path):
    src = tmp_path / "in.jpg"
    array = np.zeros((20, 40), np.uint8)
    array[:, 20:] = 255
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(array).save(src, exif=exif, quality=95)
    h.main([str(src)])
    assert Image.open(output_of(src)).size == (20, 40)
    result = pixels(output_of(src))
    assert result[:20].mean() < result[20:].mean()


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_transparent_pixels_stay_white(tmp_path, mode):
    src = tmp_path / "in.png"
    array = np.zeros((32, 32, 4), np.uint8)
    array[8:24, 8:24] = (0, 0, 0, 255)
    Image.fromarray(array).convert(mode).save(src)
    h.main([str(src)])
    result = pixels(output_of(src))
    assert result[:4].min() == 255
    assert result[12:20, 12:20].max() == 0


def test_palette_transparency_stays_white(tmp_path):
    src = tmp_path / "in.png"
    array = np.zeros((16, 16), np.uint8)
    image = Image.fromarray(array, "L").convert("P")
    image.save(src, transparency=int(np.array(image)[0, 0]))
    h.main([str(src)])
    assert pixels(output_of(src)).min() == 255


def test_sixteen_bit_gray_keeps_its_range(tmp_path):
    src = tmp_path / "in.png"
    ramp = np.tile(np.linspace(0, 65535, 256).astype(np.uint16), (64, 1))
    Image.fromarray(ramp).save(src)
    assert Image.open(src).mode.startswith("I")
    h.main([str(src)])
    result = pixels(output_of(src))
    assert abs((result > 0).mean() - 0.5) < 0.06
    assert result[:, :64].mean() < result[:, -64:].mean()


def gamma_expectation(source, gamma):
    table = np.array([round(255 * (v / 255) ** (1 / gamma)) for v in range(256)], np.uint8)
    return run_kernel(table[pixels(source)])


@pytest.mark.parametrize("gamma", [0.6, 1.8, 2.2])
def test_gamma_applies_the_tone_curve(source, gamma):
    h.main(["-g", str(gamma), str(source)])
    assert np.array_equal(pixels(output_of(source)), gamma_expectation(source, gamma))


def test_gamma_above_one_lightens(tmp_path):
    plain = tmp_path / "plain.png"
    light = tmp_path / "light.png"
    write_source(plain)
    write_source(light)
    h.main([str(plain)])
    h.main(["-g", "2", str(light)])
    assert (pixels(output_of(light)) > 0).mean() > (pixels(output_of(plain)) > 0).mean()


def test_gamma_one_changes_nothing(tmp_path):
    plain = tmp_path / "plain.png"
    unit = tmp_path / "unit.png"
    write_source(plain)
    write_source(unit)
    h.main([str(plain)])
    h.main(["--gamma", "1", str(unit)])
    assert np.array_equal(pixels(output_of(plain)), pixels(output_of(unit)))


@pytest.mark.parametrize("gamma", ["0", "nan", "inf"])
def test_unusable_gamma_is_refused_before_work(source, dither_calls, gamma):
    assert refused(["-g", gamma, str(source)]) == 2
    assert dither_calls == []


ESCP_GEOMETRY = {9: (1, 1, 216, 72, 8), 24: (39, 3, 180, 180, 24)}


def decode_escp(stream, pins, shape):
    mode, bytes_per_column, unit, rows_per_inch, pass_height = ESCP_GEOMETRY[pins]
    dots = np.zeros(shape, bool)
    state = {"resets": 0, "unidirectional": False, "passes": 0, "spacing": None}
    i = y = 0
    while i < len(stream):
        byte = stream[i]
        if byte == 0x0D:
            i += 1
        elif byte == 0x0A:
            advance = state["spacing"] * rows_per_inch / unit
            assert advance == pass_height
            y += pass_height
            i += 1
        elif byte == 0x1B and stream[i + 1] == ord("@"):
            state["resets"] += 1
            i += 2
        elif byte == 0x1B and stream[i + 1] == ord("U"):
            state["unidirectional"] = stream[i + 2] == 1
            i += 3
        elif byte == 0x1B and stream[i + 1] == ord("3"):
            state["spacing"] = stream[i + 2]
            i += 3
        elif byte == 0x1B and stream[i + 1] == ord("*"):
            assert stream[i + 2] == mode
            columns = stream[i + 3] + 256 * stream[i + 4]
            body = stream[i + 5 : i + 5 + columns * bytes_per_column]
            assert len(body) == columns * bytes_per_column
            for column in range(columns):
                word = int.from_bytes(body[column * bytes_per_column : (column + 1) * bytes_per_column])
                for pin in range(pass_height):
                    if word >> (pass_height - 1 - pin) & 1:
                        dots[y + pin, column] = True
            state["passes"] += 1
            i += 5 + columns * bytes_per_column
        else:
            raise AssertionError(f"unexpected byte {byte:#x} at offset {i}")
    return dots, state


@pytest.mark.parametrize("pins", [9, 24])
@pytest.mark.parametrize("shape", [(23, 50), (48, 300), (1, 1), (24, 7)])
def test_escp_round_trips_through_an_independent_decoder(pins, shape):
    rng = np.random.default_rng(pins + shape[0])
    dots = rng.random(shape) < 0.4
    decoded, state = decode_escp(h.escp_stream(dots, pins), pins, shape)
    assert np.array_equal(decoded, dots)
    assert state["unidirectional"]


def test_escp_exact_bytes_for_one_dot():
    dots = np.zeros((8, 1), bool)
    dots[0, 0] = True
    assert h.escp_stream(dots, 9) == (
        b"\x1b@\x1bU\x01\x1b3\x18\x1b*\x01\x01\x00\x80\r\n\x1b@"
    )


def test_escp_top_pin_is_the_high_bit_on_24_pins():
    dots = np.zeros((24, 1), bool)
    dots[23, 0] = True
    assert b"\x1b*\x27\x01\x00\x00\x00\x01\r\n" in h.escp_stream(dots, 24)


def test_escp_column_count_uses_both_bytes():
    assert b"\x1b*\x01\x2c\x01" in h.escp_stream(np.ones((8, 300), bool), 9)


def test_escp_blank_band_only_feeds_paper():
    dots = np.zeros((24, 10), bool)
    dots[16:, 0] = True
    decoded, state = decode_escp(h.escp_stream(dots, 9), 9, dots.shape)
    assert state["passes"] == 1
    assert np.array_equal(decoded, dots)


def test_escp_trims_blank_right_margin():
    dots = np.zeros((8, 500), bool)
    dots[0, 3] = True
    assert b"\x1b*\x01\x04\x00" in h.escp_stream(dots, 9)


def test_escp_resets_the_printer_when_done():
    assert h.escp_stream(np.ones((8, 2), bool), 9).endswith(b"\x1b@")


def disc_source(path, size=400, radius=150):
    yy, xx = np.mgrid[:size, :size]
    array = np.where((yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2, 0, 255)
    Image.fromarray(array.astype(np.uint8)).save(path)


@pytest.mark.parametrize(("pins", "size"), [(9, (960, 576)), (24, (1440, 1440))])
def test_print_sends_the_dithered_dots_one_for_one(tmp_path, pins, size):
    src = tmp_path / "in.png"
    capture = tmp_path / "job.prn"
    write_source(src, size=size)
    h.main(["--print", str(capture), "--pins", str(pins), str(src)])
    expected = run_kernel(pixels(src)) == 0
    decoded, state = decode_escp(capture.read_bytes(), pins, expected.shape)
    assert np.array_equal(decoded, expected)
    assert state["unidirectional"]


def test_print_defaults_to_nine_pins(tmp_path):
    src = tmp_path / "in.png"
    capture = tmp_path / "job.prn"
    write_source(src, size=(960, 576))
    h.main(["--print", str(capture), str(src)])
    assert b"\x1b*\x01" in capture.read_bytes()


@pytest.mark.parametrize(("pins", "x_dpi", "y_dpi", "shape"), [(9, 120, 72, (576, 960)), (24, 180, 180, (1440, 1440))])
def test_a_circle_prints_round(tmp_path, pins, x_dpi, y_dpi, shape):
    src = tmp_path / "in.png"
    capture = tmp_path / "job.prn"
    disc_source(src)
    h.main(["--print", str(capture), "--pins", str(pins), str(src)])
    dots, _ = decode_escp(capture.read_bytes(), pins, shape)
    rows, columns = np.nonzero(dots)
    width_inches = (columns.max() - columns.min() + 1) / x_dpi
    height_inches = (rows.max() - rows.min() + 1) / y_dpi
    assert width_inches == pytest.approx(6.0, rel=0.03)
    assert height_inches == pytest.approx(width_inches, rel=0.03)


def test_page_width_limits_the_printed_columns(tmp_path):
    src = tmp_path / "in.png"
    capture = tmp_path / "job.prn"
    Image.new("L", (100, 100), 0).save(src)
    h.main(["--print", str(capture), "--page-width", "4", str(src)])
    dots, _ = decode_escp(capture.read_bytes(), 9, (288, 480))
    assert dots[:, 479].any()


def test_print_to_stdout(source, tmp_path, capsysbinary):
    capture = tmp_path / "job.prn"
    h.main(["--print", str(capture), str(source)])
    h.main(["--print", "-", str(source)])
    assert capsysbinary.readouterr().out == capture.read_bytes()


def test_print_writes_no_image_file(source, tmp_path):
    h.main(["--print", str(tmp_path / "job.prn"), str(source)])
    assert not output_of(source).exists()


def test_print_honours_gamma(tmp_path):
    src = tmp_path / "in.png"
    plain = tmp_path / "plain.prn"
    light = tmp_path / "light.prn"
    write_source(src, size=(960, 576))
    h.main(["--print", str(plain), str(src)])
    h.main(["--print", str(light), "-g", "2", str(src)])
    assert plain.read_bytes() != light.read_bytes()


def test_print_existing_capture_needs_force(source, tmp_path, dither_calls):
    capture = tmp_path / "job.prn"
    capture.write_bytes(b"keep")
    assert refused(["--print", str(capture), str(source)]) == 2
    assert capture.read_bytes() == b"keep"
    assert dither_calls == []
    h.main(["--print", str(capture), "-f", str(source)])
    assert capture.read_bytes() != b"keep"


@pytest.mark.parametrize("device", ["/dev/hyperdither-no-such", "/no/such/dir/lp0"])
def test_missing_device_is_refused_before_work(source, dither_calls, device):
    assert refused(["--print", device, str(source)]) == 2
    assert dither_calls == []


def test_unwritable_device_names_the_lp_group(source, tmp_path, dither_calls, capsys):
    device = tmp_path / "lp0"
    device.write_bytes(b"")
    device.chmod(0o444)
    assert refused(["--print", str(device), "-f", str(source)]) == 2
    assert dither_calls == []
    assert "lp group" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra", [["-o", "x.png"], ["-e", "bmp"], ["-2"], ["-r", "100"], ["-d", "100"]]
)
def test_file_only_options_conflict_with_print(source, tmp_path, dither_calls, extra):
    assert refused(["--print", str(tmp_path / "job.prn"), *extra, str(source)]) == 2
    assert dither_calls == []


@pytest.mark.parametrize("extra", [["--pins", "24"], ["--page-width", "4"]])
def test_print_options_need_print(source, dither_calls, extra):
    assert refused([*extra, str(source)]) == 2
    assert dither_calls == []


@pytest.mark.parametrize("width", ["0", "nan", "inf"])
def test_unusable_page_width_is_refused(source, tmp_path, dither_calls, width):
    assert refused(["--print", str(tmp_path / "j.prn"), "--page-width", width, str(source)]) == 2
    assert dither_calls == []


def test_unknown_pin_count_is_refused(source, tmp_path, dither_calls):
    assert refused(["--print", str(tmp_path / "j.prn"), "--pins", "12", str(source)]) == 2
    assert dither_calls == []


def test_file_output_keeps_its_default_dpi(source):
    h.main([str(source)])
    assert Image.open(output_of(source)).info["dpi"] == pytest.approx((72, 72), abs=0.01)
