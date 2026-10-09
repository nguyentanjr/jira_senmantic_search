# -*- coding: utf-8 -*-
"""Ha diem ung vien NGUOC CHIEU ngay luc xep hang (hau xu ly, khong train).

Do contrastive dev 2026-09-22: luong issue xep dung chieu 4/8, tuc bang tung
dong xu. Khai thac cap de train chi thu duoc 87 cap sach - qua it de doi duoc
trong so model. Nhung chinh bo luat gan chieu do (ky su cham tay: 82% tren 40
cap, 91% sau khi siet) van dung duoc truc tiep o buoc xep hang:

    neu TRUY VAN co chieu ro rang tren mot truc,
    thi ung vien mang chieu NGUOC LAI tren dung truc do bi day xuong duoi.

Co y giu it tham so de khong phai chinh theo diem:
- chi tac dong trong ``window`` ket qua dau (mac dinh 10): ngoai do khong anh
  huong toi nguoi dung;
- day xuong duoi CUOI window chu khong xoa: neu bo phan loai sai chieu thi ket
  qua van con do, chi tut hang;
- truy van khong ro chieu -> khong lam gi.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_direction_pairs import AXES, classify, contains, judging_text

Label = tuple[str, str]


def doc_labels(docs: list[dict]) -> dict[str, set[Label]]:
    """key -> {(truc, ben)} theo dung luat da duoc cham tay nghiem thu."""
    out: dict[str, set[Label]] = {}
    for doc in docs:
        labels = classify(judging_text(doc), doc.get("summary", ""))
        if labels:
            out[str(doc.get("key", "")).upper()] = set(labels)
    return out


def llm_labels(path: str) -> dict[str, set[Label]]:
    """Nhan chieu do LLM gan (llm_direction_label.py). Phu ~31% corpus so voi
    7,4% cua luat tu vung. Dong loi va dong "none" bi bo qua."""
    import json

    out: dict[str, set[Label]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            axis, side = row.get("axis", ""), row.get("side", "")
            if axis in AXES and side in AXES.get(axis, {}):
                out[str(row["key"]).upper()] = {(axis, side)}
    return out


def load_mined(path: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """Tu dien bo sung do mine_direction_lexicon.py hoc tu nhan LLM."""
    import json

    with open(path, encoding="utf-8") as f:
        raw = json.load(f).get("lexicon", {})
    return {axis: {side: tuple(phrases) for side, phrases in sides.items()}
            for axis, sides in raw.items()}


def query_labels(query: str,
                 mined: dict[str, dict[str, tuple[str, ...]]] | None = None
                 ) -> set[Label]:
    """Truy van tu do khong co tieu de rieng, nen ca cau dong vai than bai.

    ``mined`` la tu dien hoc tu nhan LLM: tu dien viet tay chi khop 12/41 truy
    van that cua pilot, nen neu khong mo rong thi bo loc gan nhu khong chay.
    """
    out = set(classify(query, ""))
    if not mined:
        return out
    low = query.casefold()
    for axis, sides in mined.items():
        if any(axis == a for a, _ in out):
            continue                     # tu dien viet tay da quyet dinh
        hit = [side for side, phrases in sides.items()
               if any(contains(low, p) for p in phrases)]
        if len(hit) == 1:
            out.add((axis, hit[0]))
    return out


def is_opposite(query: set[Label], doc: set[Label]) -> bool:
    """Cung truc nhung khac ben. Cung ben, hoac khac truc -> khong phat."""
    by_axis = {axis: side for axis, side in query}
    return any(axis in by_axis and by_axis[axis] != side for axis, side in doc)


def demote(query: str, ranked_keys: list[str], labels: dict[str, set[Label]],
           window: int = 10, source_labels: set[Label] | None = None,
           mined: dict | None = None) -> list[str]:
    """Tra ve thu tu moi. ``source_labels`` dung cho luong issue: truy van la
    mot issue nen lay nhan cua chinh issue do thay vi doc lai van ban."""
    qlab = (source_labels if source_labels is not None
            else query_labels(query, mined))
    if not qlab or window <= 1:
        return list(ranked_keys)
    head = list(ranked_keys[:window])
    flagged = [k for k in head if is_opposite(qlab, labels.get(k.upper(), set()))]
    if not flagged or len(flagged) == len(head):
        return list(ranked_keys)
    keep = [k for k in head if k not in set(flagged)]
    return keep + flagged + list(ranked_keys[window:])
