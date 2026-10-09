# -*- coding: utf-8 -*-
"""
Thong ke THEO NHOM TRUY VAN -- dung chung cho group_analysis_issue.py va
group_analysis_text.py.

problem.txt: "phan tich theo nhom truy van de chi ro nhom nao lexical chiem uu
the (ma loi, exception, dinh danh service), nhom nao semantic chiem uu the, va
hybrid bu duoc bao nhieu".

NHOM -- chot truoc khi do (2026-09-14, HANDOFF.md §7 "Follow-up analyses")
  co_tin_hieu    : truy van chua it nhat mot tin hieu phan biet cua build_corpus
                   (DISCRIMINATIVE_SIGNALS: ma loi, funcCode..., exception, class,
                   duong dan API, HTTP 4xx/5xx; bo ma issue Jira) -- keyword co cho bam
  khong_tin_hieu : con lai -- mo ta thuan ngon ngu tu nhien
Luong text tinh tren CHINH chuoi truy van; luong issue dung `lexical_group` cua
issue truy van (cung dinh nghia, tinh tren mo ta goc day du).

SO SANH
  ci          : trung binh + CI 95% cua tung cau hinh trong tung nhom
  vs_baseline : cau hinh - keyword, bootstrap cap theo truy van, trong tung nhom
  gap         : (cau hinh - keyword) o co_tin_hieu TRU (cau hinh - keyword) o
                khong_tin_hieu, bootstrap doc lap hai nhom. AM = loi the so voi
                keyword LON HON o truy van ngon ngu tu nhien.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import DISCRIMINATIVE_SIGNALS, extract_signals  # noqa: E402
from eval_paraphrase import ci, paired                             # noqa: E402

GROUPS = ("co_tin_hieu", "khong_tin_hieu")


def has_signal(text: str) -> bool:
    """Chuoi co tin hieu lexical phan biet khong (ma issue Jira RO-xxx khong tinh)."""
    sig = extract_signals(text or "", project_keys=("RO",))
    return any(k in sig for k in DISCRIMINATIVE_SIGNALS)


def group_label(flag: bool) -> str:
    return GROUPS[0] if flag else GROUPS[1]


def diff_of_means(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int = 17) -> dict:
    """mean(a) - mean(b) cho hai nhom DOC LAP: bootstrap rieng tung nhom."""
    rng = np.random.default_rng(seed)
    ba = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    bb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return {"delta": round(float(a.mean() - b.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "significant": bool(lo > 0 or hi < 0)}


def group_report(scores: dict, labels, baseline: str, metrics, n_boot: int) -> dict:
    """scores: {cau_hinh: {chi_so: mang theo truy van}}; labels: nhan nhom, cung thu tu."""
    labels = np.asarray(labels)
    idx = {g: np.flatnonzero(labels == g) for g in GROUPS}
    out = {"n": {g: int(len(i)) for g, i in idx.items()}, "ci": {}, "vs_baseline": {}, "gap": {}}
    for g, i in idx.items():
        if not len(i):
            continue
        out["ci"][g] = {c: {m: ci(s[m][i], n_boot) for m in metrics} for c, s in scores.items()}
        out["vs_baseline"][g] = {c: {m: paired(s[m][i], scores[baseline][m][i], n_boot)
                                     for m in metrics}
                                 for c, s in scores.items() if c != baseline}
    if all(len(i) for i in idx.values()):
        a, b = idx[GROUPS[0]], idx[GROUPS[1]]
        out["gap"] = {c: {m: diff_of_means(s[m][a] - scores[baseline][m][a],
                                           s[m][b] - scores[baseline][m][b], n_boot)
                          for m in metrics}
                      for c, s in scores.items() if c != baseline}
    return out


def _fmt(d: dict) -> str:
    return f"{d['delta']:+.3f}[{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['significant'] else ' '}"


def print_report(title: str, rep: dict, metrics=("recall@10", "ndcg@10")) -> None:
    print(f"\n== {title} ==  n = {rep['n']}")
    for g in GROUPS:
        if g not in rep["ci"]:
            continue
        print(f"  [{g}]")
        for c, ms in rep["ci"][g].items():
            line = f"    {c:<34}" + "  ".join(f"{m} {ms[m]['mean']:.3f}" for m in metrics)
            d = rep["vs_baseline"][g].get(c)
            if d:
                line += " | vs keyword " + "  ".join(_fmt(d[m]) for m in metrics)
            print(line)
    if rep["gap"]:
        print("  gap (co_tin_hieu - khong_tin_hieu) cua [cau hinh - keyword]; am = loi hon o truy van tu nhien:")
        for c, ms in rep["gap"].items():
            print(f"    {c:<34}" + "  ".join(f"{m} {_fmt(ms[m])}" for m in metrics))
