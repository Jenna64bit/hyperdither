import ast
import re
import sys
import tomllib
from pathlib import Path

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
    wanted = {{"PIL": "pillow"}.get(name, name).lower() for name in third_party_imports()}
    assert wanted and wanted <= declared


import numpy as np
import pytest
from PIL import Image, ImageEnhance

import hyperdither as h


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
    assert np.array_equal(pixels(output_of(doubled)), once.repeat(2, axis=0).repeat(2, axis=1))


def test_double_with_mono_stays_nearest_neighbor(tmp_path):
    single = tmp_path / "single.png"
    doubled = tmp_path / "doubled.png"
    write_source(single)
    write_source(doubled)
    h.main(["-1", str(single)])
    h.main(["-1", "-2", str(doubled)])
    once = pixels(output_of(single))
    assert np.array_equal(pixels(output_of(doubled)), once.repeat(2, axis=0).repeat(2, axis=1))


def enhanced_expectation(source, enhancer, factor):
    return run_kernel(np.array(enhancer(Image.open(source).convert("L")).enhance(factor)))


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
    assert np.array_equal(pixels(output_of(source)), enhanced_expectation(source, enhancer, factor))


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
    assert np.array_equal(pixels(output_of(source)), enhanced_expectation(source, enhancer, 0.0))


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
    assert text.count("(default 1)") == 2


def test_literal_tilde_input_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_source(tmp_path / "in.png")
    h.main(["~/in.png"])
    assert (tmp_path / "in_out.png").is_file()


def test_literal_tilde_output_stays_out_of_a_directory_named_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    write_source(tmp_path / "in.png")
    h.main(["~/in.png"])
    assert not (tmp_path / "~").exists()
