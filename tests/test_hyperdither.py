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
