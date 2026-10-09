"""Đo: issue RO -> repo code. Tập kiểm = issue RO có commit mang key (ground truth mức repo).

Phương pháp:
  0. majority: luôn đoán repo phổ biến nhất trong tập kiểm (baseline tầm thường)
  1. lexical: mỏ neo trong text issue (mã lỗi, mã API, URL, CamelCase, UPPER_SNAKE, tên repo)
     khớp token trong code/config của repo, trọng số idf theo số repo chứa token
  2. neighbor: cộng phiếu lexical của top-k issue tương tự (TF-IDF trên text issue, loại chính nó)
  3. lexical+neighbor: lexical trước, neighbor phá thế hòa / bù khi lexical rỗng
Chỉ đọc; output ghi cạnh file này.
"""
import os, re, json, math, collections, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = r"D:/viettel/microservice"
ISSUES = r"D:/viettel/semantic/data/processed/all_issues.jsonl"
GT_TSV = r"C:/Users/Lenovo/AppData/Local/Temp/claude/D--viettel-semantic/b22e9c75-5970-464b-98a6-ef228d80d4aa/scratchpad/ro_commits.tsv"
OUT = os.path.dirname(os.path.abspath(__file__))
SKIP = {"target", ".git", "node_modules", "local-lab", "legacy-reference", "test", ".idea"}
EXCLUDE_REPOS = {"app-config"}
KEY_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")

# ---------- code index: token -> set(repo)
tok_re = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")
path_re = re.compile(r'"(/[A-Za-z0-9_\-/{}.$]+)"')
repos = sorted(d for d in os.listdir(ROOT)
               if os.path.isdir(os.path.join(ROOT, d)) and d not in EXCLUDE_REPOS and not d.endswith("local-lab"))
tok_repos = collections.defaultdict(set)
repo_paths = collections.defaultdict(set)
for r in repos:
    for dp, dn, fn in os.walk(os.path.join(ROOT, r)):
        dn[:] = [x for x in dn if x not in SKIP]
        for f in fn:
            if not f.endswith((".java", ".properties", ".yml", ".yaml", ".xml", ".sql", ".proto")):
                continue
            try:
                s = open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for t in set(tok_re.findall(s)):
                tok_repos[t.upper()].add(r)
            for p in path_re.findall(s):
                for seg in p.strip("/").split("/"):
                    if len(seg) >= 5 and "{" not in seg and "$" not in seg:
                        repo_paths[seg.lower()].add(r)
NR = len(repos)

# tên repo -> alias (người viết issue hay ghi "[Adapter]", "UAA", "notification"...)
alias = {}
GENERIC = {"transaction", "query", "routing", "business", "common", "base", "core", "service", "management",
           "payment", "process", "sync", "source", "money", "order", "file", "error", "task", "processor",
           "api", "gateway", "registry", "event", "backbone", "timeout", "refund", "merchant", "bank", "spi"}
for r in repos:
    name = r.replace("ewallet-", "")
    for v in {name, name.replace("-", " "), name.replace("-", "")}:
        alias.setdefault(v.lower(), set()).add(r)
    for p in name.split("-"):
        if len(p) >= 3 and p not in GENERIC and p not in ("v2",):
            alias.setdefault(p.lower(), set()).add(r)

pats = [
    re.compile(r"\b[A-Z]{2,4}_?\d{3,4}\b"),                 # mã lỗi
    re.compile(r"\b(?:API|IBR)[ _-]?(\d{2,4})\b", re.I),    # mã API -> chuẩn hóa
    re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b"),      # CamelCase
    re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]{2,})+\b"),         # UPPER_SNAKE
]

ALIAS_W = float(os.environ.get("ALIAS_W", "1"))
def lexical_scores(text):
    sc = collections.Counter()
    toks = set()
    for p in pats:
        for m in p.finditer(text):
            t = m.group(0)
            if p is pats[1]:
                n = m.group(1).zfill(4)
                toks.update({f"API_{n}", f"API{n}", f"IBR_{n}", f"IBR{n}"})
            else:
                toks.add(t.upper())
    for t in toks:
        if KEY_RE.match(t):
            continue
        rs = tok_repos.get(t)
        if rs and len(rs) <= NR // 2:
            w = math.log(NR / len(rs))
            for r in rs:
                sc[r] += w
    for p in re.findall(r"(?:/[a-zA-Z][a-zA-Z0-9\-_]{2,}){2,}", text):
        for seg in p.strip("/").split("/"):
            rs = repo_paths.get(seg.lower())
            if rs and len(rs) <= NR // 2:
                w = math.log(NR / len(rs))
                for r in rs:
                    sc[r] += w
    low = text.lower()
    for a, rs in alias.items():
        if re.search(r"\b" + re.escape(a) + r"\b", low):
            for r in rs:
                sc[r] += 2.0 * ALIAS_W
    return sc

# ---------- issues + GT
issues = [json.loads(l) for l in open(ISSUES, encoding="utf-8")]
idx = {d["key"]: i for i, d in enumerate(issues)}
def text_of(d):
    return "\n".join([d.get("summary", ""), d.get("text_business", ""), d.get("text_technical", "")])
gt = collections.defaultdict(set)
for line in open(GT_TSV, encoding="utf-8"):
    r, k = line.rstrip("\n").split("\t")
    if k in idx and r not in EXCLUDE_REPOS:
        gt[k].add(r)
test = sorted(gt)
print("repos indexed:", NR, "| test issues (GT không tính app-config):", len(test))
print("GT repo distribution:", collections.Counter(r for k in test for r in gt[k]).most_common())

# ---------- neighbors (TF-IDF, cả corpus 20k)
texts = [text_of(d) for d in issues]
vec = TfidfVectorizer(max_features=200000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(texts)
lex_cache = {}
def lex(i):
    if i not in lex_cache:
        lex_cache[i] = lexical_scores(texts[i])
    return lex_cache[i]

def neighbor_scores(i, k=20):
    sims = linear_kernel(X[i], X).ravel()
    sims[i] = -1
    top = np.argsort(-sims)[:k]
    sc = collections.Counter()
    for j in top:
        if sims[j] <= 0:
            break
        l = lex(j)
        tot = sum(l.values())
        if tot <= 0:
            continue
        for r, v in l.items():
            sc[r] += sims[j] * v / tot
    return sc

def rank(sc):
    return [r for r, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]

majority = collections.Counter(r for k in test for r in gt[k]).most_common(1)[0][0]
methods = {"0_majority": [], "1_lexical": [], "2_neighbor": [], "3_lexical+neighbor": []}
rows = []
for k in test:
    i = idx[k]
    L, N = lex(i), neighbor_scores(i)
    comb = collections.Counter({r: v + 1e-3 * N.get(r, 0) for r, v in L.items()}) if L else N
    preds = {"0_majority": [majority] + [r for r in repos if r != majority],
             "1_lexical": rank(L), "2_neighbor": rank(N), "3_lexical+neighbor": rank(comb)}
    for m, p in preds.items():
        methods[m].append((k, p))
    rows.append({"key": k, "gt": sorted(gt[k]), "lexical_top3": rank(L)[:3], "neighbor_top3": rank(N)[:3],
                 "combined_top3": rank(comb)[:3], "summary": issues[i]["summary"][:120]})

def boot_ci(hits, B=2000, seed=20260928):
    rng = random.Random(seed); n = len(hits)
    vals = sorted(sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B)]

report = {}
print("\nmethod                 cover   top1 [95% CI]          top3 [95% CI]")
for m, lst in methods.items():
    h1 = [1 if p and p[0] in gt[k] else 0 for k, p in lst]
    h3 = [1 if set(p[:3]) & gt[k] else 0 for k, p in lst]
    cov = sum(1 for _, p in lst if p) / len(lst)
    c1, c3 = boot_ci(h1), boot_ci(h3)
    report[m] = {"coverage": cov, "top1": sum(h1) / len(h1), "top1_ci": c1, "top3": sum(h3) / len(h3), "top3_ci": c3}
    print(f"{m:22s} {cov:5.2f}   {sum(h1)/len(h1):.3f} [{c1[0]:.2f},{c1[1]:.2f}]   {sum(h3)/len(h3):.3f} [{c3[0]:.2f},{c3[1]:.2f}]")

# ngoài business-adapter (tránh bị repo chiếm đa số kéo số lên)
non_ba = [k for k in test if gt[k] != {"ewallet-business-adapter"}]
print(f"\nChỉ issue KHÔNG thuần business-adapter (n={len(non_ba)}):")
for m, lst in methods.items():
    d = dict(lst)
    h1 = sum(1 for k in non_ba if d[k] and d[k][0] in gt[k]) / len(non_ba)
    h3 = sum(1 for k in non_ba if set(d[k][:3]) & gt[k]) / len(non_ba)
    report[m]["non_ba_top1"], report[m]["non_ba_top3"] = h1, h3
    print(f"  {m:22s} top1={h1:.3f} top3={h3:.3f}")

# độ phủ trên toàn corpus: bao nhiêu issue có dự đoán lexical
cov_all = sum(1 for i in range(len(issues)) if lex(i)) / len(issues)
print(f"\nToàn corpus RO: {cov_all:.1%} issue có ít nhất 1 mỏ neo lexical chấm được repo")
report["corpus_lexical_coverage"] = cov_all
json.dump({"report": report, "rows": rows}, open(os.path.join(OUT, os.environ.get("OUTF","result_v2.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
