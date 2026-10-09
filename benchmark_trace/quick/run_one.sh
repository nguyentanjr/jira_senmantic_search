#!/bin/bash
# usage: run_one.sh FLOWID RUN
Q=/d/viettel/semantic/benchmark_trace/quick
C=/c/Users/Lenovo/AppData/Local/OpenAI/Codex/bin/faa963e871dd422c/codex.exe
F=$1; R=$2; D=$Q/runs/${F}_r$R; mkdir -p $D
SPEC=$(grep "^$F|" $Q/flows.txt | cut -d'|' -f2-)
python - "$Q/prompt_template.md" "$D/prompt.md" "$F: $SPEC" <<'PY'
import sys; t=open(sys.argv[1],encoding='utf-8').read(); open(sys.argv[2],'w',encoding='utf-8').write(t.replace('{{FLOW}}',sys.argv[3]))
PY
start=$(date +%s)
timeout 2700 "$C" exec -m gpt-6-astra -c model_reasoning_effort='"xhigh"' -s workspace-write \
  -C "$(cygpath -w $D)" --skip-git-repo-check --json -o "$(cygpath -w $D/final.json)" - < $D/prompt.md > $D/events.jsonl 2> $D/stderr.txt
echo "exit=$? seconds=$(( $(date +%s)-start ))" > $D/meta.txt
