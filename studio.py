"""
CodebaseLM Studio: Sleek Dark-Mode Web Visualizer & Playground
for CodebaseLlama (5.87M) AST Context Grounding, Fast Repo-Brain Adaptation,
and Hallucination Guard.

Run:
    python studio.py
Then open http://localhost:8050 in your browser.
"""

import ast
import difflib
import json
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.nano_llama import CodebaseLlama, CodebaseLlamaConfig
from repo_context.builder import IGNORE_DIRS, analyze_python_file

TOKENIZER_PATH = ROOT / "data" / "tokenizer_llama_4k.json"
CKPT_PATH = ROOT / "checkpoints" / "codebase_llama_6m.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "repo_path": str(ROOT),
    "files": [],
    "symbols": {},
    "tasks": [],
    "training_blocks": [],
    "adapted": False,
    "adapt_time": 0.0,
}


def init_engine():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    cfg = CodebaseLlamaConfig(**ckpt["config"])
    model = CodebaseLlama(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    STATE["config"] = cfg
    scan_repo(str(ROOT))


def scan_repo(repo_str: str):
    repo_path = Path(repo_str).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Directory not found: {repo_str}")

    py_files = [
        p for p in sorted(repo_path.rglob("*.py"))
        if not any(part in IGNORE_DIRS for part in p.parts)
    ]

    files_data = []
    all_symbols = {}
    tasks = []
    training_blocks = []

    for path in py_files:
        rel_path = path.relative_to(repo_path).as_posix()
        info = analyze_python_file(path)
        if not info:
            continue

        file_syms = []
        for cls in info["classes"]:
            cname = cls.split(" ")[0]
            file_syms.append(cname)
            all_symbols[cname] = rel_path
        for fn in info["functions"]:
            fname = fn.split("(")[0]
            if fname != "main":
                file_syms.append(fname)
                all_symbols[fname] = rel_path

        files_data.append(
            {
                "file": rel_path,
                "classes": info["classes"],
                "functions": info["functions"],
                "symbols": file_syms,
            }
        )

        if file_syms:
            header = f"# repo: {', '.join(file_syms[:4])}\n"
            primary = file_syms[0]
            tasks.append(
                {
                    "file": rel_path,
                    "header": header.strip(),
                    "prefix": "def caller(x):\n    return ",
                    "target": f"{primary}(x)",
                    "symbol": primary,
                }
            )
            for sym in file_syms:
                training_blocks.append(f"{header}def caller(x):\n    return {sym}(x)\n")

    STATE["repo_path"] = str(repo_path)
    STATE["files"] = files_data
    STATE["symbols"] = all_symbols
    STATE["tasks"] = tasks
    STATE["training_blocks"] = training_blocks


@torch.no_grad()
def analyze_single_prompt(header: str, prefix: str, target: str):
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    max_len = STATE["config"].max_seq_len
    model.eval()

    results = {}
    for mode in ("no_ctx", "with_ctx"):
        full_prefix = prefix if mode == "no_ctx" else (header.strip() + "\n" + prefix)
        p_ids = tokenizer.encode(full_prefix).ids
        t_ids = tokenizer.encode(target).ids
        seq = p_ids + t_ids

        loss_sum = 0.0
        correct = 0
        token_breakdown = []

        for idx in range(len(p_ids), len(seq)):
            start = max(0, idx - max_len)
            x = torch.tensor([seq[start:idx]], dtype=torch.long, device=DEVICE)
            target_id = seq[idx]
            y = torch.tensor([target_id], dtype=torch.long, device=DEVICE)

            logits = model(x)[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            loss_sum += F.cross_entropy(logits.unsqueeze(0), y).item()

            pred_id = int(torch.argmax(probs).item())
            if pred_id == target_id:
                correct += 1

            token_breakdown.append(
                {
                    "target_tok": tokenizer.decode([target_id]),
                    "pred_tok": tokenizer.decode([pred_id]),
                    "target_prob": round(float(probs[target_id].item()) * 100, 1),
                    "match": pred_id == target_id,
                }
            )

        count = max(len(t_ids), 1)
        avg_loss = loss_sum / count
        ppl = round(math.exp(min(avg_loss, 12)), 2)
        acc = round((correct / count) * 100.0, 2)

        # Also generate greedy completion from full_prefix
        gen_ids = list(p_ids)
        for _ in range(len(t_ids) + 2):
            gx = torch.tensor([gen_ids[-max_len:]], dtype=torch.long, device=DEVICE)
            next_id = int(torch.argmax(model(gx)[0, -1, :]).item())
            gen_ids.append(next_id)
            if tokenizer.decode([next_id]).endswith("\n"):
                break
        completion = tokenizer.decode(gen_ids[len(p_ids) :]).splitlines()[0]

        results[mode] = {
            "perplexity": ppl,
            "accuracy": acc,
            "completion": completion,
            "tokens": token_breakdown,
        }

    return results


def fast_batched_adapt(steps: int = 35):
    """Batched GPU tensor adaptation that finishes in ~2 seconds on RTX 2050."""
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    blocks = STATE["training_blocks"]
    if not blocks:
        return 0.0

    pad_id = tokenizer.token_to_id("<pad>") or 0
    encoded = [tokenizer.encode(b).ids for b in blocks]
    max_l = max(len(seq) for seq in encoded)

    x_batch, y_batch = [], []
    for seq in encoded:
        padded = seq + [pad_id] * (max_l - len(seq))
        x_batch.append(padded[:-1])
        y_subs = padded[1:]
        y_masked = [
            tok if idx < len(seq) - 1 else -100 for idx, tok in enumerate(y_subs)
        ]
        y_batch.append(y_masked)

    x_tensor = torch.tensor(x_batch, dtype=torch.long, device=DEVICE)
    y_tensor = torch.tensor(y_batch, dtype=torch.long, device=DEVICE)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(x_tensor)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_tensor.view(-1),
                ignore_index=-100,
            )
        loss.backward()
        optimizer.step()

    model.eval()
    elapsed = round(time.time() - t0, 2)
    STATE["adapted"] = True
    STATE["adapt_time"] = elapsed
    return elapsed


def verify_snippet(snippet: str):
    all_symbols = STATE["symbols"]
    builtins_set = set(dir(__builtins__)) | {"print", "len", "range", "isinstance", "max", "min"}
    findings = []
    try:
        tree = ast.parse(snippet)
    except SyntaxError as e:
        return [{"status": "ERROR", "call": "SyntaxError", "detail": str(e)}]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn in all_symbols:
                findings.append(
                    {"status": "VERIFIED", "call": f"{fn}()", "detail": f"Defined in {all_symbols[fn]}"}
                )
            elif fn in builtins_set:
                findings.append(
                    {"status": "BUILTIN", "call": f"{fn}()", "detail": "Python Standard Builtin"}
                )
            else:
                matches = difflib.get_close_matches(fn, all_symbols.keys(), n=1, cutoff=0.45)
                fix = f"Did you mean `{matches[0]}()` in `{all_symbols[matches[0]]}`?" if matches else "No matching symbol in repo."
                findings.append({"status": "HALLUCINATED", "call": f"{fn}()", "detail": fix})
    return findings


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CodebaseLM Studio — 5.87M Neural Repo Engine</title>
<style>
  :root {
    --bg: #09090b; --panel: #121215; --border: #27272a;
    --text: #f4f4f5; --muted: #a1a1aa; --accent: #10b981;
    --danger: #ef4444; --blue: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 20px; }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; }
  .top-bar { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 18px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .badge { background: rgba(16,185,129,0.15); color: var(--accent); border: 1px solid rgba(16,185,129,0.3); padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .grid { display: grid; grid-template-columns: 340px 1fr; gap: 18px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 14px; }
  input, textarea { width: 100%; background: #09090b; border: 1px solid var(--border); color: var(--text); padding: 9px 12px; border-radius: 6px; margin-bottom: 10px; }
  button { background: var(--text); color: #09090b; border: none; padding: 9px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.15s; }
  button:hover { opacity: 0.9; }
  .btn-accent { background: var(--accent); color: #052e16; }
  .file-item { padding: 9px 11px; border: 1px solid var(--border); border-radius: 6px; margin-bottom: 8px; cursor: pointer; transition: 0.15s; }
  .file-item:hover { border-color: var(--accent); background: rgba(16,185,129,0.05); }
  .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
  .metric-box { background: #09090b; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .big-num { font-size: 28px; font-weight: 700; margin: 6px 0; }
  .tok-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 12px; }
  .bar-bg { flex: 1; height: 8px; background: #27272a; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); }
  .pill-ok { color: var(--accent); font-weight: 600; }
  .pill-err { color: var(--danger); font-weight: 600; }
</style>
</head>
<body>
  <div class="top-bar">
    <div class="brand">
      <strong style="font-size:18px;">CodebaseLM Studio</strong>
      <span class="badge mono" id="model-badge">CodebaseLlama 5.87M • CUDA</span>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span class="mono" style="color:var(--muted);" id="adapt-status">Zero-Shot Mode</span>
      <button class="btn-accent" onclick="adaptRepo()">⚡ Instant Repo-Brain Adapt</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>1. Repository AST Graph</h2>
      <div style="display:flex; gap:8px; margin-bottom:12px;">
        <input type="text" id="repo-input" class="mono" value="." style="margin:0;">
        <button onclick="scanRepo()">Scan</button>
      </div>
      <div id="repo-stats" class="mono" style="color:var(--muted); margin-bottom:12px; font-size:12px;"></div>
      <div id="file-list" style="max-height:520px; overflow-y:auto;"></div>
    </div>

    <div style="display:flex; flex-direction:column; gap:18px;">
      <div class="card">
        <h2>2. Live AST Context Grounding vs. Uncontextualized Baseline</h2>
        <label class="mono" style="color:var(--muted);">AST Context Header (`# repo:`):</label>
        <input type="text" id="header-input" class="mono">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
          <div>
            <label class="mono" style="color:var(--muted);">Code Prefix:</label>
            <input type="text" id="prefix-input" class="mono">
          </div>
          <div>
            <label class="mono" style="color:var(--muted);">Target Repository Call:</label>
            <input type="text" id="target-input" class="mono">
          </div>
        </div>
        <button onclick="runComparison()">Run Neural Evaluation</button>

        <div class="compare-grid">
          <div class="metric-box">
            <div class="mono" style="color:var(--muted);">WITHOUT AST CONTEXT (BLIND)</div>
            <div class="big-num" id="no-ctx-ppl">—</div>
            <div class="mono" id="no-ctx-acc" style="color:var(--muted);">Accuracy: —</div>
            <div class="mono" id="no-ctx-gen" style="margin-top:8px; color:#f87171;"></div>
            <div id="no-ctx-bars" style="margin-top:10px;"></div>
          </div>
          <div class="metric-box" style="border-color: rgba(16,185,129,0.4);">
            <div class="mono" style="color:var(--accent);">WITH `# repo:` AST CONTEXT</div>
            <div class="big-num" id="with-ctx-ppl" style="color:var(--accent);">—</div>
            <div class="mono" id="with-ctx-acc" style="color:var(--text);">Accuracy: —</div>
            <div class="mono" id="with-ctx-gen" style="margin-top:8px; color:var(--accent);"></div>
            <div id="with-ctx-bars" style="margin-top:10px;"></div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>3. Hallucination Guard — Live AST Code Verifier</h2>
        <div style="display:flex; gap:10px;">
          <input type="text" id="verify-input" class="mono" value="res = load_split_dataset(tokenizer, split_path)" style="margin:0;">
          <button onclick="runVerify()">Verify Calls</button>
        </div>
        <div id="verify-output" class="mono" style="margin-top:12px;"></div>
      </div>
    </div>
  </div>

<script>
let tasks = [];

async function loadState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  document.getElementById('repo-input').value = data.repo_path;
  document.getElementById('repo-stats').innerText = `${data.files.length} Python files | ${Object.keys(data.symbols).length} AST symbols indexed`;
  tasks = data.tasks;

  const list = document.getElementById('file-list');
  list.innerHTML = '';
  data.tasks.forEach((t, i) => {
    const div = document.createElement('div');
    div.className = 'file-item mono';
    div.innerHTML = `<div style="font-weight:600;">${t.file}</div><div style="color:var(--accent); font-size:11px; margin-top:3px;">${t.header}</div>`;
    div.onclick = () => selectTask(i);
    list.appendChild(div);
  });
  if (tasks.length > 0) selectTask(1 < tasks.length ? 1 : 0);
  runVerify();
}

function selectTask(i) {
  const t = tasks[i];
  document.getElementById('header-input').value = t.header;
  document.getElementById('prefix-input').value = t.prefix;
  document.getElementById('target-input').value = t.target;
  runComparison();
}

function renderBars(containerId, tokens) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  tokens.forEach(t => {
    const color = t.match ? '#10b981' : '#ef4444';
    el.innerHTML += `
      <div class="tok-row mono">
        <span style="width:95px; overflow:hidden;">'${t.target_tok}'</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.min(t.target_prob, 100)}%; background:${color};"></div></div>
        <span style="width:50px; text-align:right;">${t.target_prob}%</span>
      </div>`;
  });
}

async function runComparison() {
  const payload = {
    header: document.getElementById('header-input').value,
    prefix: document.getElementById('prefix-input').value,
    target: document.getElementById('target-input').value,
  };
  const res = await fetch('/api/compare', { method: 'POST', body: JSON.stringify(payload) });
  const data = await res.json();

  document.getElementById('no-ctx-ppl').innerText = `PPL: ${data.no_ctx.perplexity}`;
  document.getElementById('no-ctx-acc').innerText = `Token Accuracy: ${data.no_ctx.accuracy}%`;
  document.getElementById('no-ctx-gen').innerText = `Pred: ${data.no_ctx.completion}`;
  renderBars('no-ctx-bars', data.no_ctx.tokens);

  document.getElementById('with-ctx-ppl').innerText = `PPL: ${data.with_ctx.perplexity}`;
  document.getElementById('with-ctx-acc').innerText = `Token Accuracy: ${data.with_ctx.accuracy}%`;
  document.getElementById('with-ctx-gen').innerText = `Pred: ${data.with_ctx.completion}`;
  renderBars('with-ctx-bars', data.with_ctx.tokens);
}

async function adaptRepo() {
  const status = document.getElementById('adapt-status');
  status.innerText = 'Adapting on GPU...';
  const res = await fetch('/api/adapt', { method: 'POST' });
  const data = await res.json();
  status.innerText = `⚡ Repo-Brain Adapted (${data.elapsed}s)`;
  status.style.color = '#10b981';
  runComparison();
}

async function scanRepo() {
  const path = document.getElementById('repo-input').value;
  await fetch('/api/scan', { method: 'POST', body: JSON.stringify({ repo_path: path }) });
  loadState();
}

async function runVerify() {
  const snippet = document.getElementById('verify-input').value;
  const res = await fetch('/api/verify', { method: 'POST', body: JSON.stringify({ snippet }) });
  const data = await res.json();
  const out = document.getElementById('verify-output');
  out.innerHTML = data.findings.map(f => {
    const cls = f.status === 'HALLUCINATED' ? 'pill-err' : 'pill-ok';
    return `<div style="margin-bottom:6px;"><span class="${cls}">[${f.status}]</span> <strong>${f.call}</strong> — ${f.detail}</div>`;
  }).join('');
}

loadState();
</script>
</body>
</html>
"""


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._send_json(
                {
                    "repo_path": STATE["repo_path"],
                    "files": STATE["files"],
                    "symbols": STATE["symbols"],
                    "tasks": STATE["tasks"],
                    "adapted": STATE["adapted"],
                }
            )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(raw)

        if self.path == "/api/compare":
            res = analyze_single_prompt(
                payload.get("header", ""),
                payload.get("prefix", ""),
                payload.get("target", ""),
            )
            self._send_json(res)
        elif self.path == "/api/adapt":
            elapsed = fast_batched_adapt(steps=35)
            self._send_json({"elapsed": elapsed})
        elif self.path == "/api/scan":
            scan_repo(payload.get("repo_path", "."))
            self._send_json({"ok": True})
        elif self.path == "/api/verify":
            findings = verify_snippet(payload.get("snippet", ""))
            self._send_json({"findings": findings})


def main():
    print("Loading CodebaseLlama (5.87M) into GPU memory...")
    init_engine()
    port = 8050
    server = HTTPServer(("127.0.0.1", port), StudioHandler)
    print("=" * 70)
    print(f"  CODEBASE-LM STUDIO LIVE AT: http://localhost:{port}")
    print("=" * 70)
    print("Open http://localhost:8050 in Chrome/Edge to launch the visual playground!")
    server.serve_forever()


if __name__ == "__main__":
    main()