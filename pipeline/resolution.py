# -*- coding: utf-8 -*-
"""
Trich phan XU LY DA GHI NHAN (root cause / solution) tu comment.

Duoi truc "truy hoi issue tuong dong de tai su dung tri thuc", day la phan mang
gia tri nghiep vu that: he thong dua issue cu len KHONG phai de noi "trung lap,
dong di", ma de noi "cai nay giong RO-xxx, hoi do nguyen nhan la X, sua the nay".

Do tren corpus 10.925 bug:
    co tu khoa root cause / solution : 1.492  (13,7%)
      - chi tro sang issue khac      :    25   ("Root Cause tuong tu RO-11923")
      - qua ngan, khong co noi dung  :   485   ("Fixed", "OK", "da fix")
      - CO NOI DUNG THAT             :   982  ( 9,0%)   median 156 ky tu

9% la tran gia tri ma du lieu hien tai cho phep. Con so nay phai duoc bao cao,
vi no noi rang muon he thong huu ich hon thi phai cai thien ky luat ghi
resolution, khong phai cai thien model.
"""
from __future__ import annotations

import re

RC_RE = re.compile(
    r"(root\s*cause|nguyên nhân|nguyen nhan|solution|cách xử lý|đã fix|fix:)",
    re.I)
XREF_RE = re.compile(r"\b[A-Z]{2,10}-\d+\b")
CUE_RE = re.compile(r"trùng|tương tự|giống", re.I)

#: duoi nguong nay coi nhu khong co noi dung ("Fixed", "OK", "da fix xong")
MIN_BODY = 40


def extract(comments: str, max_chars: int = 600) -> str:
    """-> doan xu ly dung duoc, hoac chuoi rong.

    Tra ve rong trong hai truong hop, deu la 'co chu nhung khong co thong tin':
      1. doan do chi tro sang issue khac  ("Root Cause va Solution tuong tu
         RO-11923") -- day vua la bang chung gia, vua chinh la cau cue sinh ra
         nhan gold, nen phai loai;
      2. doan qua ngan de mang noi dung nao.
    """
    c = comments or ""
    m = RC_RE.search(c)
    if not m:
        return ""
    seg = c[m.start(): m.start() + max_chars]
    head = seg[:120]
    if XREF_RE.search(head) and CUE_RE.search(head):
        return ""
    body = seg[len(m.group(0)):].strip(" :–-\n")
    return seg.strip() if len(body) >= MIN_BODY else ""


def has(doc: dict) -> bool:
    return bool(extract(doc.get("text_comments", "")))
