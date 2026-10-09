"""Consumers per resolved production topic value (reference aid, read-only)."""
import os, re, json, sys

ROOT = r"D:/viettel/microservice"
CFG = os.path.join(ROOT, "app-config")
SKIP = {"target", ".git", "node_modules", "local-lab", "legacy-reference", "test"}


def load_props(service):
    props = {}
    d = os.path.join(CFG, service, "production")
    if not os.path.isdir(d):
        return props
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if f.endswith(".properties"):
            for line in open(p, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    props[k.strip()] = v.strip()
        elif f.endswith((".yml", ".yaml")):
            stack = []
            for line in open(p, encoding="utf-8", errors="ignore"):
                if not line.strip() or line.strip().startswith("#") or line.strip().startswith("-"):
                    continue
                ind = len(line) - len(line.lstrip())
                m = re.match(r"\s*([\w.\-]+)\s*:\s*(.*)$", line.rstrip())
                if not m:
                    continue
                while stack and stack[-1][0] >= ind:
                    stack.pop()
                key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
                stack.append((ind, key))
                if val:
                    props[".".join(k for _, k in stack)] = val
    return props


def resolve(expr, props):
    expr = expr.strip()
    m = re.fullmatch(r"#\{'?\$\{([^}]+)\}'?\}", expr)
    if m:
        expr = "${" + m.group(1) + "}"
    m = re.fullmatch(r"\$\{([^:}]+)(?::([^}]*))?\}", expr)
    if m:
        k, default = m.group(1), m.group(2)
        if k in props:
            return props[k], "config"
        if default is not None:
            return default, "default"
        return None, "missing"
    return expr.strip('"'), "literal"


out = []
for repo in sorted(os.listdir(ROOT)):
    rp = os.path.join(ROOT, repo)
    if not os.path.isdir(rp) or repo == "app-config" or repo.endswith("local-lab"):
        continue
    props = load_props(repo)
    for dp, dn, fn in os.walk(rp):
        dn[:] = [x for x in dn if x not in SKIP]
        for f in fn:
            if not f.endswith(".java"):
                continue
            path = os.path.join(dp, f)
            s = open(path, encoding="utf-8", errors="ignore").read()
            for m in re.finditer(r"@KafkaListener\s*\((.*?)\)\s*(?:public|private|protected)?[^;{]*?(\w+)\s*\(", s, re.S):
                args, meth = m.group(1), m.group(2)
                t = re.search(r"topics\s*=\s*(\{(?:\s*\"[^\"]*\"\s*,?)+\s*\}|\"[^\"]*\")", args, re.S)
                if not t:
                    continue
                exprs = re.findall(r"\"([^\"]*)\"", t.group(1))
                line = s[: m.start()].count("\n") + 1
                for e in exprs:
                    val, how = resolve(e, props)
                    out.append({"topic": val, "how": how, "expr": e, "repo": repo,
                                "method": f"{f[:-5]}#{meth}",
                                "file": os.path.relpath(path, ROOT).replace("\\", "/"), "line": line,
                                "has_prod_config": bool(props)})
json.dump(out, open(os.path.join(os.path.dirname(__file__), "kafka_consumers.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
q = sys.argv[1:]
for r in out:
    if not q or any(x in (r["topic"] or "") for x in q):
        print(r["topic"], "|", r["how"], "|", r["repo"], r["method"], f'{r["file"]}:{r["line"]}')
