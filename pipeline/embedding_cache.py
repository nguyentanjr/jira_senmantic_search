"""Content-addressed embedding caches. Legacy prefix-only caches are never reused."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile

SCHEMA = 2


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_hash(path):
    """Hash weights, tokenizer, pooling, configs and custom code in full.

    Documentation and training reports are excluded; all inference file types
    used by the local SentenceTransformer models are included recursively.
    Deliberately not memoized by mtime/size: weights can change at the same path.
    """
    root = Path(path).resolve()
    suffixes = {".json", ".safetensors", ".bin", ".pt", ".pth", ".model",
                ".txt", ".vocab", ".merges", ".py", ".onnx", ".xml"}
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in suffixes
                   and not any(x.startswith(".") for x in p.relative_to(root).parts))
    if not files:
        raise ValueError(f"No model inference files in {root}")
    entries = [(p.relative_to(root).as_posix(), file_hash(p)) for p in files]
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode()).hexdigest()


def identity(texts, model_identity, query_prefix, passage_prefix, max_seq_length):
    digest = hashlib.sha256()
    for text in texts:
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    runtime = {}
    for name in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers"):
        try:
            runtime[name] = version(name)
        except PackageNotFoundError:
            runtime[name] = None
    return {"schema": SCHEMA, "model": model_identity, "runtime": runtime,
            "texts_sha256": digest.hexdigest(), "rows": len(texts),
            "query_prefix": query_prefix, "passage_prefix": passage_prefix,
            "max_seq_length": max_seq_length, "normalize_embeddings": True,
            "dtype": "float32"}


def cache_path(cache_dir, tag, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return os.path.join(cache_dir, f"{tag}_v{SCHEMA}_{digest}.npy")


def load(path, metadata):
    import numpy as np
    try:
        with open(path + ".json", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("identity") != metadata or manifest.get("sha256") != file_hash(path):
            return None
        array = np.load(path, allow_pickle=False)
        if (array.ndim != 2 or array.shape[0] != metadata["rows"]
                or array.shape[1] < 1 or array.dtype != np.float32
                or not np.isfinite(array).all()
                or list(array.shape) != manifest.get("shape")):
            return None
        return array
    except (OSError, ValueError, EOFError, TypeError, AttributeError):
        return None


def save(path, metadata, array):
    """Atomic files; a partial/concurrent pair fails closed via its checksum."""
    import numpy as np
    if (array.ndim != 2 or array.shape[0] != metadata["rows"]
            or array.shape[1] < 1 or array.dtype != np.float32
            or not np.isfinite(array).all()):
        raise ValueError("Invalid embedding matrix; cache not written")
    temp_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".npy",
                                         delete=False) as stream:
            temp_paths.append(stream.name)
            np.save(stream, array, allow_pickle=False)
        manifest = {"identity": metadata, "shape": list(array.shape),
                    "sha256": file_hash(temp_paths[0])}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=os.path.dirname(path), suffix=".json",
                                         delete=False) as stream:
            temp_paths.append(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        os.replace(temp_paths[0], path)
        os.replace(temp_paths[1], path + ".json")
    finally:
        for temp in temp_paths:
            if os.path.exists(temp):
                os.unlink(temp)
