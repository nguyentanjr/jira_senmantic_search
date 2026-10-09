# -*- coding: utf-8 -*-
"""Khai thac cap NGUOC CHIEU tu corpus de day model phan biet loai loi.

Vi sao: do contrastive dev ngay 2026-09-22 cho thay luong issue chi xep dung
chieu 4/8 (bang tung dong xu), va pilot co 23 chip "nguoc chieu" + 99 chip
"sai loai loi". Du lieu train hien tai chi day "tim dung issue", chua bao gio
day "hai issue cung chu de nhung nguoc chieu thi KHAC nhau".

Cach lam, co y giu don gian va kiem duoc bang mat:

1. Moi ISSUE duoc gan mot *truc* (axis) va mot *ben* (side) bang luat tu vung,
   vi du truc ``send_count`` co ben ``missing`` ("khong gui sms") va ben
   ``extra`` ("gui double"). Cham tu vung cua CA HAI ben -> bo, vi cau chua
   ro chieu.
2. Issue duoc gom theo *chu de* = the trong ngoac vuong dau summary
   ("[CMS merchant][request delete]" -> "cms merchant"). Cung truc + cung chu
   de + KHAC ben => cap nguoc chieu.
3. Cap train: anchor + positive (cung ben, cung chu de) + negatives (khac ben,
   cung chu de). Khong co positive thi bo ca anchor: mot cap chi co negative
   khong day duoc gi cho MultipleNegativesRankingLoss.

Ba rao chan:
- Moi issue thuoc bat ky bo do nao (gold, pooling, contrastive, paraphrase
  sach) deu bi loai, dung ``build_feedback_pairs.existing_eval_keys``.
- Cap co van ban gan trung nhau (jaccard >= ``--dup-jaccard``) bi loai: day la
  bai "tim ban trung", khong phai bai phan biet chieu.
- Doi lai, cap qua khac nhau (jaccard < ``--min-jaccard``) cung bi loai: khong
  cung chu de thi khong phai negative kho.

Dau ra: JSONL dung schema cua ``task_pairs_v2`` (key/kind/qv/query/positive/
negatives) de ``train_task_models.py`` dung duoc ngay, kem mot CSV mau de ky su
cham tay truoc khi train.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_feedback_pairs import existing_eval_keys

SEED = 20260922

# Moi truc: ten -> {ben: [cum tu]}. Cum tu duoc so khop tren summary + actual.
AXES: dict[str, dict[str, tuple[str, ...]]] = {
    "send_count": {
        "missing": ("không gửi sms", "chưa gửi sms", "không gửi tin nhắn",
                    "chưa gửi tin nhắn", "không nhận được sms", "thiếu sms",
                    "gửi thiếu sms", "không thấy sms", "chưa nhận được sms",
                    "không có sms", "chưa gửi thông báo", "không gửi noti",
                    "chưa gửi noti", "không bắn sms", "gửi thiếu tin nhắn",
                    "thiếu tin nhắn", "gửi thiếu"),
        "extra": ("gửi double", "bị double", "gửi thừa", "thừa sms", "sms thừa",
                  "gửi 2 tin nhắn", "gửi 2 sms", "gửi 3 sms", "gửi lặp", "sms bị lặp",
                  "double sms", "sms double", "gửi trùng sms", "bắn 2 sms",
                  "nhận được 2 sms", "nhận 2 tin nhắn", "gửi nhiều sms"),
    },
    "permission": {
        "blocked": ("có quyền", "được cấp quyền", "được phân quyền",
                    "theo file phân quyền"),
        "not_enforced": ("không có quyền vẫn", "chưa check phân quyền",
                         "chưa thực hiện phân quyền", "chưa phân quyền",
                         "không có quyền nhưng vẫn", "vẫn thực hiện được"),
    },
    "button_state": {
        "enable_wrong": ("vẫn enable", "luôn enable", "đang enable",
                         "enable button", "không disable", "vẫn cho phép",
                         "vẫn đang cho phép", "vẫn cho sửa", "vẫn cho nhập"),
        "disable_wrong": ("bị disable", "đang disable", "disable button",
                          "disable sai", "không cho nhập"),
    },
    "limit_enforce": {
        "not_blocked": ("vượt hạn mức nhưng", "vi phạm hạn mức không",
                        "vi phạm hạn mức vẫn", "không báo lỗi hạn mức",
                        "không bắt lỗi", "vẫn thành công"),
        "over_blocked": ("báo lỗi vi phạm hạn mức", "báo lỗi hạn mức",
                         "er30019", "vượt quá hạn mức", "không thực hiện được"),
    },
    "otp_required": {
        "missing_otp": ("không bắt nhập otp", "không yêu cầu otp",
                        "chưa bắt nhập otp", "không bắt otp"),
        "extra_otp": ("vẫn yêu cầu nhập otp", "vẫn bắt nhập otp",
                      "yêu cầu nhập otp", "vẫn generate otp"),
    },
    "result_status": {
        "false_success": ("thất bại nhưng vẫn", "thất bại nhưng báo thành công",
                          "không thành công nhưng", "lỗi nhưng vẫn thành công",
                          "vẫn trừ tiền", "vẫn hạch toán"),
        "false_failure": ("thành công nhưng báo lỗi", "thành công nhưng hiển thị lỗi",
                          "đã thành công nhưng", "thành công nhưng không",
                          "system busy nhưng"),
    },
}

TAG = re.compile(r"\[([^\]]{2,60})\]")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
STOP = {"lỗi", "bug", "issue", "test", "khi", "của", "và", "các", "cho", "với",
        "không", "vẫn", "bị", "là", "có", "được", "trong", "tại", "ở", "thì"}


def judging_text(doc: dict) -> str:
    """Van ban duoc phep quyet dinh chieu: summary + ``actual``, KHONG lay
    ``steps``. Phan steps ke thao tac nen thuong chua ca hai trang thai
    ("chon disable... kiem tra enable"), gan nhan tu do se cho chieu nguoc."""
    sec = doc.get("sections") or {}
    actual = sec.get("actual", "") if isinstance(sec, dict) else ""
    return f"{doc.get('summary', '')}\n{actual}".strip()


def text_of(doc: dict) -> str:
    """Van ban dung de train: dung dung truong ma index dang luu."""
    return (doc.get("text_dense") or judging_text(doc)).strip()


def topic_of(doc: dict) -> str:
    """Chu de = the ngoac vuong dau tien, chuan hoa. Khong co the -> rong."""
    tags = TAG.findall(doc.get("summary", ""))
    if not tags:
        return ""
    tag = re.sub(r"\s+", " ", tags[0]).strip().casefold()
    return re.sub(r"[_\-/]+", " ", tag)


NEGATORS = ("không ", "khong ", "chưa ", "chua ", "ko ", "k ", "kh ")


def contains(low: str, phrase: str) -> bool:
    """So khop cum tu nhung BO QUA lan xuat hien nam sau tu phu dinh.

    Bat buoc vi cum "co quyen" nam TRONG "khong co quyen": RO-13385 ("role
    khong co quyen van create merchant thanh cong") tung bi gan nham thanh
    chieu "co quyen ma bi chan" - dung chieu nguoc lai.
    """
    start = low.find(phrase)
    while start >= 0:
        # Cua so 16 ky tu chu khong phai 8: phu dinh co the cach cum tu vai chu
        # ("khong BI yeu cau xac thuc OTP" - tung bi hieu thanh "co hoi OTP").
        head = low[max(0, start - 16):start]
        if not any(neg in head for neg in NEGATORS):
            return True
        start = low.find(phrase, start + 1)
    return False


def sides_in(text: str, sides: dict) -> list[str]:
    low = text.casefold()
    return [side for side, phrases in sides.items()
            if any(contains(low, p) for p in phrases)]


def classify(text: str, summary: str = "") -> list[tuple[str, str]]:
    """Tra ve [(truc, ben)] theo thu tu uu tien TIEU DE.

    Cum tu chi chieu con nam ca trong phan "expected" hoac trong mo ta cac
    trang thai khac, nen so khop tren toan van cho nhan nguoc: RO-21196 tieu de
    la "Van dang cho phep chinh sua" (enable_wrong) nhung than bai co "dang
    disable" nen bi gan disable_wrong. Vi vay:

    - tieu de chi ro mot ben  -> lay ben do;
    - tieu de im lang         -> lay ben duy nhat cua phan than;
    - hai nguon mau thuan, hoac mot nguon cham ca hai ben -> BO (mo ho).
    """
    body = text if not summary else text[len(summary):]
    out = []
    for axis, sides in AXES.items():
        head_hit = sides_in(summary, sides) if summary else []
        body_hit = sides_in(body, sides)
        if len(head_hit) == 1:
            if len(body_hit) == 1 and body_hit[0] != head_hit[0]:
                continue          # tieu de va than bai noi nguoc nhau
            out.append((axis, head_hit[0]))
        elif not head_hit and len(body_hit) == 1:
            out.append((axis, body_hit[0]))
    return out


def tokens(text: str) -> set[str]:
    return {w for w in (m.group().casefold() for m in WORD.finditer(text))
            if len(w) > 1 and w not in STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def build_idf(token_sets: list[set[str]]) -> dict[str, float]:
    """IDF tren chinh corpus. Tu pho bien ("giao dich", "thanh cong", "sms")
    gan nhu khong dong gop, nen do giong se do ten module quyet dinh."""
    df = Counter()
    for toks in token_sets:
        df.update(toks)
    n = max(1, len(token_sets))
    return {tok: math.log(n / c) for tok, c in df.items()}


def idf_cosine(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    inter = sum(idf.get(t, 0.0) ** 2 for t in a & b)
    na = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in a))
    nb = math.sqrt(sum(idf.get(t, 0.0) ** 2 for t in b))
    return inter / (na * nb) if na and nb else 0.0


def shared_rare(a: set[str], b: set[str], idf: dict[str, float],
                min_idf: float) -> set[str]:
    """Tu chung DAC TRUNG: ten module, ten chuc nang. Khong co tu nao nhu vay
    thi hai issue chi trung nhau o tu nghiep vu chung chung."""
    return {t for t in a & b if idf.get(t, 0.0) >= min_idf}


def llm_label_source(path: str, axes: set[str] | None = None,
                     summary_guard: bool = True):
    """Doc nhan LLM da siet lam nguon gan truc, thay cho luat tu vung.

    Luat tu vung chi phu 7,4% corpus va mang dung khuon sai da do duoc
    (2026-09-23: "gui thieu THONG TIN trong sms" khop cum "gui thieu"). Nhan
    LLM sau khi siet dat 14/14 tren send_count va 4/6 tren permission o bo giu
    rieng, nen day moi la nguon nen dung de sinh bo ba.

    Moi issue chi co DUNG MOT (truc, ben) - LLM tra mot nhan - khac luat tu
    vung co the tra nhieu truc cho mot issue.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            axis, side = row.get("axis", ""), row.get("side", "")
            key = str(row.get("key", "")).upper()
            if axis in AXES and side in AXES.get(axis, {}) and (not axes or axis in axes):
                out[key] = [(axis, side)]
            else:
                out.pop(key, None)      # dong sau ghi de dong truoc

    def label(doc: dict) -> list[tuple[str, str]]:
        """Bo nhan khi TIEU DE noi nguoc lai nhan LLM.

        ``classify`` von co rao nay ("hai nguon mau thuan -> BO"); chuyen sang
        nhan LLM la mat no. Rao van can vi prompt bao LLM chi doc ``actual``,
        ma mot so issue viet HANH VI MONG MUON vao ``actual``: RO-16380 tieu de
        "he thong KHONG gui SMS" nhung actual "he thong thuc hien gui SMS" ->
        LLM gan ``extra``, nguoc han. Chi 12 issue dinh loi nay nhung chung bi
        dung lai lam negative nhieu nen cham 17% so bo ba.
        """
        labels = out.get(str(doc.get("key", "")).upper(), [])
        if not summary_guard:
            return labels
        summary = doc.get("summary", "")
        keep = []
        for axis, side in labels:
            head = sides_in(summary, AXES[axis])
            if len(head) == 1 and head[0] != side:
                continue
            keep.append((axis, side))
        return keep
    return label


def inventory(pairs: list[dict], pool_sizes: Counter) -> dict:
    """Kiem ke de biet hang nghin bo ba co phai chi la vai chuc issue lap lai.

    Tong so bo ba KHONG noi len luong du lieu that: mot anchor sinh mot bo ba
    nhung mot issue co the xuat hien lam negative cua hang chuc anchor khac.
    Vi vay dem RIENG so issue duy nhat o tung vai.
    """
    report = {}
    axes = sorted({p["axis"] for p in pairs}) if pairs else []
    for axis in axes:
        sub = [p for p in pairs if p["axis"] == axis]
        anchors = {p["key"] for p in sub}
        positives = {p["positive_key"] for p in sub}
        negatives = {k for p in sub for k in p["neg_keys"]}
        have = sum(v for (a, _s), v in pool_sizes.items() if a == axis)
        report[axis] = {
            "issue_co_nhan": have,
            "issue_ghep_duoc": len(anchors),
            "ti_le_ghep_duoc": round(len(anchors) / have, 3) if have else 0.0,
            "bo_ba": len(sub),
            "anchor_duy_nhat": len(anchors),
            "positive_duy_nhat": len(positives),
            "negative_duy_nhat": len(negatives),
            "issue_duy_nhat_cham_vao": len(anchors | positives | negatives),
            "theo_ben": dict(Counter(p["side"] for p in sub)),
        }
    return report


def near_dup_to_eval(docs: list[dict], excluded: set[str], threshold: float,
                     candidates: set[str] | None = None) -> set[str]:
    """Issue train GAN TRUNG mot issue cua bo do, du khac key.

    Loai theo key khong du: RO-14303 va RO-14296 cung mot loi, chi khac the
    module dau summary, jaccard 0,966. Train tren cai nay roi do tren cai kia
    la ro ri kieu cu ma du an da dinh mot lan (xem memory finetune-readiness).

    So khop qua CHI MUC NGUOC thay vi quet doi: jaccard >= t bat buoc
    |A giao B| >= t * |A|, nen chi can xet nhung issue bo do co du so tu chung.
    Quet doi la 42 trieu phep so va khong chay noi.
    """
    if threshold <= 0:
        return set()
    by_key = {str(d.get("key", "")).upper(): d for d in docs}
    index: dict[str, list[str]] = defaultdict(list)
    eval_toks: dict[str, set[str]] = {}
    for key, doc in by_key.items():
        if key not in excluded:
            continue
        toks = tokens(text_of(doc))
        eval_toks[key] = toks
        for tok in toks:
            index[tok].append(key)
    hit = set()
    for key, doc in by_key.items():
        if key in excluded or (candidates is not None and key not in candidates):
            continue
        toks = tokens(text_of(doc))
        if not toks:
            continue
        overlap = Counter(k for tok in toks for k in index.get(tok, ()))
        need = threshold * len(toks)
        for other, n in overlap.items():
            if n >= need and jaccard(toks, eval_toks[other]) >= threshold:
                hit.add(key)
                break
    return hit


def e5_scorer(texts: dict[str, str], model_path: str, topic_bonus: float,
              topics: dict[str, str]):
    """Do giong bang CHINH mo hinh dang dung, thay cho cosine IDF.

    Ly do doi: 2026-09-23 do tren 106 bo ba da cham tay, cosine IDF chi bat duoc
    15/24 dong ma e5 that su xep sai, va bao dong gia 12 dong. Negative it trung
    tu van lam e5 nham chieu - do chinh la loi goc cua du an ("khong hien thi
    icon DELETE" bi xep gan "hien thi icon edit" hon ca ban dong nghia). Muon
    negative KHO voi mo hinh thi phai hoi mo hinh, khong hoi tu vung.

    ``topic_bonus`` cong them cho ung vien cung the module: no khong quyet dinh
    thay cosine, chi uu tien khi diem xap xi nhau - vi cham tay cho thay khac
    the module van co the cung nghiep vu ("Unlock Investigation" va "Reset Pin"
    deu la "gui SMS cho checker khi tao request").
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device="cpu")
    keys = sorted(texts)
    # Luong issue<->issue dung 'query: ' ca hai phia, dung nhu eval_similar.py
    # va dung nhu luc fine-tune task_pairs_v2.
    vecs = model.encode(["query: " + texts[k] for k in keys], batch_size=16,
                        normalize_embeddings=True, show_progress_bar=False)
    emb = dict(zip(keys, vecs))

    def score(a_key: str, b_key: str) -> float:
        base = float(emb[a_key] @ emb[b_key])
        if topic_bonus and topics.get(a_key) and topics.get(a_key) == topics.get(b_key):
            base += topic_bonus
        return base
    return score


def build(docs: list[dict], excluded: set[str], args,
          label_of=None, scorer=None) -> tuple[list[dict], list[dict], Counter]:
    """``label_of(doc) -> [(truc, ben)]``. Mac dinh la luat tu vung; truyen
    vao ham khac de dung nhan LLM da siet (xem ``--llm-labels``)."""
    label_of = label_of or (
        lambda d: classify(judging_text(d), d.get("summary", "")))
    stats = Counter()
    pool = defaultdict(list)          # (axis, topic, side) -> [doc]
    corpus_tokens: list[set[str]] = []
    for doc in docs:
        key = str(doc.get("key", "")).upper()
        if key in excluded:
            stats["bo_vi_nam_trong_bo_do"] += 1
            continue
        text = text_of(doc)
        topic = topic_of(doc)
        if not topic:
            stats["bo_vi_khong_co_the_chu_de"] += 1
            continue
        corpus_tokens.append(tokens(text_of(doc)))
        labels = label_of(doc)
        if not labels:
            continue
        for axis, side in labels:
            pool[(axis, topic, side)].append({"key": key, "text": text,
                                              "tok": tokens(text)})
            stats[f"gan_nhan:{axis}:{side}"] += 1

    # IDF tinh tren TOAN BO corpus du dieu kien, khong phai tren rieng nhom da
    # gan nhan: neu tinh tren nhom thi thang IDF doi theo kich thuoc nhom va
    # nguong --rare-idf mat y nghia giua cac lan chay.
    idf = build_idf(corpus_tokens)
    rng = random.Random(SEED)
    pairs, review = [], []
    topics = {(axis, topic) for axis, topic, _ in pool}
    if args.topic_mode == "none":
        # Chu de khong con la dieu kien cung: nguong jaccard ben duoi da bao dam
        # hai issue cung mang. Gop het ve mot "chu de" duy nhat cho moi truc.
        merged = defaultdict(list)
        for (axis, _topic, side), items in pool.items():
            merged[(axis, "*", side)].extend(items)
        pool = merged
        topics = {(axis, "*") for axis, _, _ in pool}
    for axis, topic in sorted(topics):
        sides = [s for s in AXES[axis] if pool.get((axis, topic, s))]
        if len(sides) < 2:
            stats["chu_de_chi_co_mot_ben"] += 1
            continue
        for side in sides:
            others = [s for s in sides if s != side]
            same = pool[(axis, topic, side)]
            opp = [d for s in others for d in pool[(axis, topic, s)]]

            def related(d, anchor=None):
                """Cung mang hay khong. Jaccard tran khong du: "giao dich",
                "thanh cong", "sms" xuat hien khap noi nen hai issue khac han
                module van vuot nguong (ky su cham tay 2026-09-22: truc
                button_state chi dung 4/8). Vi vay doi THEM cosine co trong so
                IDF va it nhat vai tu chung dac trung (ten module/chuc nang)."""
                j = jaccard(anchor["tok"], d["tok"])
                if not (args.min_jaccard <= j < args.dup_jaccard):
                    return None
                sim = (scorer(anchor["key"], d["key"]) if scorer
                       else idf_cosine(anchor["tok"], d["tok"], idf))
                if sim < args.min_sim:
                    return None
                if scorer is None and len(shared_rare(
                        anchor["tok"], d["tok"], idf,
                        args.rare_idf)) < args.min_shared_rare:
                    # Rao "tu chung dac trung" la rao TU VUNG: khi cham diem
                    # bang e5 no se loai dung nhung negative kho nhat (it trung
                    # tu ma van bi xep nham chieu).
                    return None
                return sim

            for anchor in same:
                scored_pos = [(related(d, anchor), d) for d in same
                              if d["key"] != anchor["key"]]
                scored_neg = [(related(d, anchor), d) for d in opp]
                positives = [(s, d) for s, d in scored_pos if s is not None]
                negatives = [(s, d) for s, d in scored_neg if s is not None]
                if not positives:
                    stats["bo_vi_khong_co_positive"] += 1
                    continue
                if not negatives:
                    stats["bo_vi_khong_co_negative"] += 1
                    continue
                positive = max(positives, key=lambda x: x[0])[1]
                negatives = [d for _, d in sorted(negatives, key=lambda x: -x[0])
                             ][:args.max_negatives]
                pairs.append({
                    "key": anchor["key"], "kind": f"direction_{axis}", "qv": "goc",
                    "query": anchor["text"], "positive": positive["text"],
                    "negatives": [d["text"] for d in negatives],
                    "positive_key": positive["key"],
                    "neg_keys": [d["key"] for d in negatives],
                    "axis": axis, "side": side, "topic": topic,
                    "source": "mined_direction_pairs",
                })
                stats["cap_train"] += 1
                # Hien HET negative: 54% dong co nhieu hon mot, cham moi negative
                # dau tien thi mot dong "dung" van co the mang negative thu 2-4
                # sai ma khong ai thay.
                review.append({
                    "axis": axis, "topic": topic, "side": side, "ok": "",
                    "anchor_key": anchor["key"], "anchor": anchor["text"][:300],
                    "positive_key": positive["key"], "positive": positive["text"][:300],
                    "so_negative": len(negatives),
                    "negative_key": negatives[0]["key"],
                    "negative": negatives[0]["text"][:300],
                    "negative_khac": " || ".join(
                        f'[{d["key"]}] {" ".join(d["text"].split())[:170]}'
                        for d in negatives[1:]),
                })
    rng.shuffle(review)
    return pairs, review, stats          # main quyet dinh rut mau the nao


def sample_review(review: list[dict], args) -> list[dict]:
    """Rut mau de cham tay. Mac dinh gop chung; --review-per-axis rut deu.

    Rut gop chung khong doc duoc ti le dung cua tung truc: truc dong se chiem
    phan lon mau. Va trong moi truc con can ca hai ben, vi loi co the chi nam o
    mot chieu. Cot cham de trong CO Y tach ba cau hoi khac nhau - nhan chieu,
    do tuong dong cua positive, do nguoc chieu cua negative - vi mot bo ba co
    the co nhan dung ma positive van lech.
    """
    rng = random.Random(SEED)
    if args.review_per_axis:
        picked: list[dict] = []
        by_axis: dict[str, list[dict]] = defaultdict(list)
        for row in review:
            by_axis[row["axis"]].append(row)
        for axis in sorted(by_axis):
            by_side: dict[str, list[dict]] = defaultdict(list)
            for row in by_axis[axis]:
                by_side[row["side"]].append(row)
            for side in by_side:
                rng.shuffle(by_side[side])
            quota = args.review_per_axis
            sides = sorted(by_side, key=lambda s: len(by_side[s]))
            take: list[dict] = []
            for i, side in enumerate(sides):        # ben it truoc, chia deu phan con lai
                share = min(len(by_side[side]), -(-(quota - len(take)) // (len(sides) - i)))
                take += by_side[side][:share]
            picked += take[:quota]
        review = picked
    else:
        review = review[:args.review_sample]
    cols = ["nhan_anchor", "nhan_positive", "nhan_negative",
            "positive_tuong_dong", "negative_nguoc_chieu",
            "negative_khac_nguoc_chieu", "ghi_chu"]
    out = []
    for row in review:
        new = {k: v for k, v in row.items() if k != "ok"}
        for c in cols:
            new[c] = ""
        out.append(new)
    return out


def main() -> None:
    d = "data/processed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=f"{d}/corpus.jsonl")
    ap.add_argument("--gold-eval", default=f"{d}/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default=f"{d}/pooling_judgments_v4_llm.csv")
    ap.add_argument("--contrast", default=f"{d}/contrastive_jira_review.csv")
    ap.add_argument("--clean-split", default=f"{d}/finetune_split_clean.json")
    ap.add_argument("--min-jaccard", type=float, default=0.20,
                    help="duoi nguong nay thi khong cung chu de -> negative de")
    ap.add_argument("--dup-jaccard", type=float, default=0.80,
                    help="tu nguong nay tro len coi la ban trung -> bo")
    ap.add_argument("--max-negatives", type=int, default=4)
    ap.add_argument("--min-sim", type=float, default=0.0,
                    help="cosine co trong so IDF toi thieu giua anchor va doi tac")
    ap.add_argument("--min-shared-rare", type=int, default=1,
                    help="so tu chung dac trung toi thieu (ten module/chuc nang)")
    ap.add_argument("--rare-idf", type=float, default=4.0,
                    help="nguong IDF de coi mot tu la dac trung")
    ap.add_argument("--review-sample", type=int, default=40)
    ap.add_argument("--review-per-axis", type=int, default=0,
                    help="rut N cap MOI TRUC, can bang hai ben, thay vi N cap gop "
                         "chung. Mau gop chung bi truc dong lan at nen khong doc "
                         "duoc ti le dung cua tung truc")
    ap.add_argument("--topic-mode", choices=["tag", "none"], default="none",
                    help="tag: bat buoc cung the ngoac vuong; none: chi dua vao jaccard")
    ap.add_argument("--llm-labels", default="",
                    help="dung nhan LLM da siet lam nguon gan truc thay cho luat "
                         "tu vung, vd data/processed/direction_labels_llm.jsonl")
    ap.add_argument("--axes", default="",
                    help="chi lay cac truc nay (ngan cach dau phay); de trong = tat ca. "
                         "Dung de loai truc chua cham do tin cay ra khoi lot train dau")
    ap.add_argument("--extra-exclude",
                    default=f"{d}/direction_axis_check_dev.csv,"
                            f"{d}/direction_axis_check_holdout.csv",
                    help="CSV co cot key, loai them khoi train (bo cham chat luong nhan)")
    ap.add_argument("--no-summary-guard", action="store_true",
                    help="tat rao chan 'tieu de nguoc nhan'. Cham tay 2026-09-23: "
                         "rao nay chi dung 6/12 ca - no bi chinh bo tu vung lam "
                         "hong (cum 'thanh cong nhung khong' cua false_failure "
                         "khop ca cau false_success). Bat/tat de do lai")
    ap.add_argument("--e5-model", default="",
                    help="cham diem do giong bang chinh checkpoint nay thay cho "
                         "cosine IDF, vd models/e5-base-task2")
    ap.add_argument("--topic-bonus", type=float, default=0.05,
                    help="cong them cho ung vien cung the module khi dung --e5-model")
    ap.add_argument("--eval-dup-jaccard", type=float, default=0.70,
                    help="loai issue train gan trung mot issue bo do o nguong nay; "
                         "0 = tat. Loai theo key khong bat duoc ban clone khac the")
    ap.add_argument("--report-out", default=f"{d}/direction_pairs_inventory.json")
    ap.add_argument("--out", default=f"{d}/direction_pairs.jsonl")
    ap.add_argument("--review-out", default=f"{d}/direction_pairs_review.csv")
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    excluded = existing_eval_keys(args.gold_eval, args.pooling, args.contrast,
                                  args.clean_split)
    # Bo cham CHAT LUONG NHAN cung phai ra khoi train: neu train tren chinh cac
    # issue dung de do do chinh xac nhan thi con so do khong con doc duoc nua.
    for path in (args.extra_exclude or "").split(","):
        path = path.strip()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8-sig", newline="") as f:
                excluded |= {str(r["key"]).strip().upper()
                             for r in csv.DictReader(f) if r.get("key")}
    axes = {a.strip() for a in args.axes.split(",") if a.strip()}
    unknown = axes - set(AXES)
    if unknown:
        raise SystemExit(f"truc khong co that: {sorted(unknown)}")
    label_of = None
    if args.llm_labels:
        label_of = llm_label_source(args.llm_labels, axes or None,
                                    not args.no_summary_guard)
    elif axes:
        def label_of(doc, _axes=axes):
            return [(a, sd) for a, sd in
                    classify(judging_text(doc), doc.get("summary", ""))
                    if a in _axes]
    labelled = ({str(d.get("key", "")).upper() for d in docs if label_of(d)}
                if label_of else None)
    docs_by_key = {str(d.get("key", "")).upper(): d for d in docs}
    near = near_dup_to_eval(docs, excluded, args.eval_dup_jaccard, labelled)
    if near:
        print(f"loai them {len(near)} issue gan trung bo do "
              f"(jaccard >= {args.eval_dup_jaccard})")
        excluded |= near
    scorer = None
    if args.e5_model:
        cands = {str(d.get("key", "")).upper(): text_of(d) for d in docs
                 if str(d.get("key", "")).upper() not in excluded and label_of
                 and label_of(d) and topic_of(d)}
        print(f"cham diem bang {args.e5_model} tren {len(cands)} ung vien...")
        scorer = e5_scorer(cands, args.e5_model, args.topic_bonus,
                           {k: topic_of(docs_by_key[k]) for k in cands})
    pairs, review, stats = build(docs, excluded, args, label_of, scorer)
    if not pairs:
        raise SystemExit("Khong khai thac duoc cap nao; kiem tra lai tu vung AXES.")

    leaked = sorted({p["key"] for p in pairs} & excluded)
    if leaked:
        raise SystemExit(f"RO RI: {len(leaked)} issue cua bo do lot vao train, vd {leaked[:5]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    review = sample_review(review, args)
    with open(args.review_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0]))
        w.writeheader(); w.writerows(review)

    pool_sizes = Counter()
    for name, n in stats.items():
        if name.startswith("gan_nhan:"):
            _, axis, side = name.split(":")
            pool_sizes[(axis, side)] = n
    report = {
        "nguon_nhan": args.llm_labels or "luat tu vung (classify)",
        "truc_lay": sorted(axes) if axes else "tat ca",
        "loai_tru_bo_do": len(excluded),
        "cap_train": len(pairs),
        "issue_khac_nhau": len({p["key"] for p in pairs}),
        "theo_truc": inventory(pairs, pool_sizes),
        "thong_ke": dict(stats),
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print()
    print(f"khong co issue nao cua bo do lot vao train ({len(excluded)} key bi loai)")
    print(f"-> {args.out}\n-> {args.review_out} ({len(review)} cap de cham tay)"
          f"\n-> {args.report_out}")


if __name__ == "__main__":
    main()
