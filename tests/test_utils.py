"""Unit tests for the lightweight utility modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.scoring import normalize_scores, rank_top_k, smooth_scores
from src.utils.io_utils import ensure_dir, new_run_dir, write_results_csv


def test_normalize_scores_basic():
    s = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    n = normalize_scores(s)
    assert n.min() == pytest.approx(0.0)
    assert n.max() == pytest.approx(1.0)
    assert n[0] < n[-1]


def test_normalize_scores_constant():
    s = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    n = normalize_scores(s)
    assert np.allclose(n, 0.0)


def test_normalize_scores_empty():
    s = np.array([], dtype=np.float32)
    n = normalize_scores(s)
    assert n.shape == (0,)


def test_smooth_scores_identity():
    s = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(smooth_scores(s, window=1), s)
    assert np.allclose(smooth_scores(s, window=0), s)


def test_smooth_scores_window():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    out = smooth_scores(s, window=3)
    assert out.shape == s.shape
    # Interior point should be the plain mean of its 3-window.
    assert out[2] == pytest.approx(3.0)


def test_rank_top_k_basic():
    s = np.array([0.1, 0.9, 0.3, 0.8, 0.2], dtype=np.float32)
    top = rank_top_k(s, k=2)
    assert top == [1, 3]


def test_rank_top_k_min_gap_dedup():
    # Adjacent near-duplicates at indices 1 and 2 — with gap=1 we should skip one.
    s = np.array([0.1, 0.95, 0.94, 0.2, 0.85], dtype=np.float32)
    top = rank_top_k(s, k=2, min_gap=1)
    assert top[0] == 1
    assert 2 not in top  # dedup'd
    assert top[1] == 4


def test_rank_top_k_k_zero_or_empty():
    assert rank_top_k(np.array([1.0]), k=0) == []
    assert rank_top_k(np.array([], dtype=np.float32), k=5) == []


def test_ensure_dir_and_run_dir(tmp_path: Path):
    d = ensure_dir(tmp_path / "nested" / "thing")
    assert d.exists()

    run = new_run_dir(tmp_path, prefix="testrun")
    assert run.exists()
    assert (run / "frames").exists()


def test_write_results_csv(tmp_path: Path):
    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    out = write_results_csv(rows, tmp_path / "out.csv")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "a,b" in text.splitlines()[0]
    assert "1,x" in text
    assert "2,y" in text
