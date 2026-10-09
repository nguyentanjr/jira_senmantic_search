"""Diem cross-encoder tinh truoc cho luong issue (bang tra thay cho CE tren CPU).

Quy trinh:
  1. python pipeline/dump_issue_ce_pairs.py              -> colab/issue_ce_offline_bundle/data
  2. Colab: colab/issue_ce_offline_colab.ipynb            -> issue_ce_offline_result.zip
  3. python pipeline/ce_offline.py import --result issue_ce_offline_result.zip
         [--check-predictions data/processed/relation_independent_eval_result_20260923/predictions.jsonl]
                                                         -> data/processed/ce_issue_offline/

Khi tim: lay top-100 hybrid TRUC TIEP (khong dung pool da dump, de issue moi them
vao corpus van duoc xet), roi tra diem tung cap (truy van, ung vien). Mot cap chi
hop le khi van ban CA HAI phia con dung nhu luc cham (so sanh text_sha). Thieu
du mot cap -> tra None, nguoi goi giu thu tu hybrid: tron diem CE voi diem
tinh lai online o may khac/fp khac la khong dang tin.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import load_jsonl                                    # noqa: E402

DEFAULT_STORE = "data/processed/ce_issue_offline"
DEFAULT_BUNDLE_DATA = "colab/issue_ce_offline_bundle/data"


def text_sha(text: str) -> str:
    """Van tay van ban cap; lech -> diem cu khong con dung cho issue do."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class OfflineCEStore:
    """Bang diem CE (truy van, ung vien) gan voi corpus hien tai qua khoa issue."""

    def __init__(self, keys: list[str], shas: list[str], pools: np.ndarray,
                 scores: np.ndarray, manifest: dict | None = None):
        if pools.shape != scores.shape or len(keys) != len(shas):
            raise ValueError("store hong: kich thuoc khong khop")
        self.keys, self.shas = keys, shas
        self.pools, self.scores = pools, scores
        self.manifest = manifest or {}
        self.key_to_row = {k: i for i, k in enumerate(keys)}
        self._doc_to_store: np.ndarray | None = None
        self._valid: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STORE) -> "OfflineCEStore":
        path = Path(path)
        z = np.load(path / "ce_scores.npz", allow_pickle=False)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return cls([str(k) for k in z["keys"]], [str(s) for s in z["text_sha"]],
                   z["pools"], z["scores"], manifest)

    def bind(self, docs: list[dict], field: str = "text_dense") -> dict:
        """Noi store voi corpus dang dung. Tra so issue con hop le / da doi / moi."""
        n_store = len(self.keys)
        self._doc_to_store = np.full(len(docs), -1, dtype=np.int64)
        self._valid = np.zeros(n_store, dtype=bool)
        changed = new = 0
        for i, d in enumerate(docs):
            r = self.key_to_row.get(d["key"])
            if r is None:
                new += 1
                continue
            if text_sha(d.get(field, "")) != self.shas[r]:
                changed += 1
                continue
            self._doc_to_store[i] = r
            self._valid[r] = True
        return {"valid": int(self._valid.sum()), "changed": changed, "new": new}

    def lookup(self, query_doc: int, cand_docs: np.ndarray) -> np.ndarray | None:
        """Diem CE cho tung ung vien (cung thu tu), None neu thieu bat ky cap nao."""
        if self._doc_to_store is None:
            raise RuntimeError("goi bind(docs) truoc")
        qr = self._doc_to_store[query_doc]
        if qr < 0:
            return None
        row_scores = {int(c): float(s) for c, s in zip(self.pools[qr], self.scores[qr])
                      if c >= 0 and self._valid[c]}
        out = np.empty(len(cand_docs), dtype=np.float32)
        for j, d in enumerate(cand_docs):
            sr = self._doc_to_store[int(d)]
            if sr < 0 or int(sr) not in row_scores:
                return None
            out[j] = row_scores[int(sr)]
        return out

    def rerank(self, query_doc: int, ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Xep `ranked` (thu tu hybrid) theo diem CE, giu thu tu hybrid khi hoa diem."""
        s = self.lookup(query_doc, ranked)
        if s is None:
            return None
        order = np.argsort(-s, kind="stable")
        return ranked[order], s[order]


def _read_result(result: Path) -> tuple[np.ndarray, dict]:
    if result.is_dir():
        return (np.load(result / "ce_scores.npy"),
                json.loads((result / "manifest.json").read_text(encoding="utf-8")))
    with zipfile.ZipFile(result) as zf:
        scores = np.load(io.BytesIO(zf.read("ce_scores.npy")))
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    return scores, manifest


def import_result(result: Path, bundle_data: Path, out: Path) -> dict:
    scores, rman = _read_result(result)
    pools = np.load(bundle_data / "pools.npy")
    pools_sha = hashlib.sha256((bundle_data / "pools.npy").read_bytes()).hexdigest()
    if rman.get("pools_sha256") != pools_sha:
        raise SystemExit("Ket qua Colab duoc cham tren pools.npy KHAC bundle hien tai")
    if scores.shape != pools.shape:
        raise SystemExit(f"shape {scores.shape} != pools {pools.shape}")
    valid = pools >= 0
    if np.isnan(scores[valid]).any() or not np.isfinite(scores[valid]).all():
        raise SystemExit("Thieu diem hoac diem khong huu han")
    if rman.get("labels_used") or rman.get("training"):
        raise SystemExit("Ket qua khong phai inference thuan")
    docs = load_jsonl(str(bundle_data / "docs.jsonl"))
    if len(docs) != pools.shape[0]:
        raise SystemExit("Bundle chi dump mot phan truy van (--limit); khong nhap")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "ce_scores.npz",
                        keys=np.array([d["key"] for d in docs]),
                        text_sha=np.array([d["text_sha"] for d in docs]),
                        pools=pools, scores=scores.astype(np.float32))
    manifest = {"source": str(result), "colab": rman,
                "policy": "issue flow: live hybrid top-100, reorder by CE (replace); "
                          "any missing/stale pair -> keep hybrid order"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return manifest


def check_predictions(store: OfflineCEStore, path: Path) -> dict:
    """So voi diem CE da cham truoc day (vd. 10 family) -- cung model, cung cap."""
    diffs, same_top10, n = [], 0, 0
    for r in load_jsonl(str(path)):
        qr = store.key_to_row[r["query_key"]]
        mine = {store.keys[c]: s for c, s in zip(store.pools[qr], store.scores[qr]) if c >= 0}
        diffs += [abs(mine[k] - s) for k, s in zip(r["reranked_keys"], r["reranked_scores"])]
        order = sorted(mine, key=lambda k: -mine[k])[:10]
        same_top10 += set(order) == set(r["reranked_keys"][:10])
        n += 1
    return {"queries": n, "pairs": len(diffs), "max_abs_diff": float(max(diffs)),
            "mean_abs_diff": float(np.mean(diffs)), "same_top10_set": same_top10}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="nhap ket qua Colab thanh store")
    imp.add_argument("--result", required=True, help="issue_ce_offline_result.zip hoac thu muc")
    imp.add_argument("--bundle-data", default=DEFAULT_BUNDLE_DATA)
    imp.add_argument("--out", default=DEFAULT_STORE)
    imp.add_argument("--check-predictions", default="")
    st = sub.add_parser("status", help="do store voi corpus hien tai")
    st.add_argument("--store", default=DEFAULT_STORE)
    st.add_argument("--corpus", default="data/processed/corpus.jsonl")
    args = ap.parse_args()

    if args.cmd == "import":
        m = import_result(Path(args.result), Path(args.bundle_data), Path(args.out))
        print(json.dumps({"scores_shape": m["colab"]["scores_shape"],
                          "scorer": m["colab"]["scorer"]}, ensure_ascii=False, indent=2))
        if args.check_predictions:
            store = OfflineCEStore.load(args.out)
            print(json.dumps(check_predictions(store, Path(args.check_predictions)), indent=2))
    else:
        store = OfflineCEStore.load(args.store)
        print(json.dumps(store.bind(load_jsonl(args.corpus)), indent=2))


if __name__ == "__main__":
    main()
