"""
cblm: Full-Codebase AST Analyzer, Neural QA Engine & Terminal Shell
powered by CodebaseLlama (5.87M).

Commands inside `cblm>`:
    Just type any question! e.g.:
      What files are in this codebase?
      Where is CodebaseLlama defined and what are its methods?
      Which files call TinyTransformer?
      What does train_llama.py do?
      What does this codebase do?
    Or prefix commands:
      @filename prompt         -> File-contextualized code autocompletion
      :verify <code_snippet>   -> AST Hallucination Guard
      :audit                   -> Full repo cross-file call integrity audit
      :map                     -> Print full codebase AST architecture map
      :quit                    -> Exit
"""

import ast
import copy
import difflib
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.nano_llama import CodebaseLlama, CodebaseLlamaConfig
from repo_context.builder import IGNORE_DIRS

TOKENIZER_PATH = ROOT / "data" / "tokenizer_llama_4k.json"
CKPT_PATH = ROOT / "checkpoints" / "codebase_llama_6m.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def deep_parse_python_file(file_path: Path, repo_dir: Path) -> dict | None:
    """Read and deeply analyze a Python file (imports, classes, methods, functions, calls, docstrings)."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    rel_path = file_path.relative_to(repo_dir).as_posix()
    module_doc = ast.get_docstring(tree) or ""
    module_doc_short = " ".join(module_doc.strip().split())[:140]

    imports = []
    classes = []
    functions = []
    calls = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [n.name for n in node.names]
            imports.append(f"{mod}: {', '.join(names)}" if mod else ", ".join(names))
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            c_doc = " ".join((ast.get_docstring(node) or "").split())[:100]
            classes.append({"name": node.name, "methods": methods, "doc": c_doc, "line": node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            f_doc = " ".join((ast.get_docstring(node) or "").split())[:100]
            functions.append(
                {
                    "name": node.name,
                    "sig": f"{node.name}({', '.join(args)})",
                    "args": args,
                    "doc": f_doc,
                    "line": node.lineno,
                }
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    return {
        "file": rel_path,
        "path": file_path,
        "lines": len(source.splitlines()),
        "doc": module_doc_short,
        "imports": imports,
        "classes": classes,
        "functions": functions,
        "calls": sorted(calls),
    }


class RepoEngine:
    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir.resolve()
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        self.config = CodebaseLlamaConfig(**ckpt["config"])

        self.model = CodebaseLlama(self.config).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.base_model = copy.deepcopy(self.model)
        self.base_model.eval()

        self.files = []
        self.symbols = {}
        self.symbol_details = {}
        self.file_headers = {}
        self.training_blocks = []
        self.qa_pairs = []
        self.global_header = ""
        self.scan()

    def scan(self):
        py_files = [
            p for p in sorted(self.repo_dir.rglob("*.py"))
            if not any(part in IGNORE_DIRS for part in p.parts)
        ]
        self.files = []
        self.symbols = {}
        self.symbol_details = {}
        self.file_headers = {}
        self.training_blocks = []
        self.qa_pairs = []

        for path in py_files:
            info = deep_parse_python_file(path, self.repo_dir)
            if not info:
                continue

            rel = info["file"]
            syms = []

            for c in info["classes"]:
                cname = c["name"]
                syms.append(cname)
                self.symbols[cname] = rel
                m_str = ", ".join(c["methods"]) if c["methods"] else "none"
                self.symbol_details[cname] = {
                    "type": "class",
                    "file": rel,
                    "line": c["line"],
                    "signature": f"class {cname} (methods: {m_str})",
                    "doc": c["doc"],
                }
                q = f"Where is class {cname} defined?"
                a = f"Class {cname} is defined in {rel} (line {c['line']}) with methods: {m_str}."
                self.qa_pairs.append((q, a, f"# repo: {rel} | class {cname}"))

            for f in info["functions"]:
                fname = f["name"]
                if fname != "main":
                    syms.append(fname)
                    self.symbols[fname] = rel
                self.symbol_details[fname] = {
                    "type": "function",
                    "file": rel,
                    "line": f["line"],
                    "signature": f"def {f['sig']}",
                    "doc": f["doc"],
                }
                if fname != "main":
                    q = f"Where is {fname} defined?"
                    a = f"Function {f['sig']} is defined in {rel} at line {f['line']}."
                    self.qa_pairs.append((q, a, f"# repo: {rel} | def {f['sig']}"))

            info["symbols"] = syms
            self.files.append(info)

            if syms:
                hdr = f"# repo: {', '.join(syms[:4])}\n"
                self.file_headers[rel] = hdr
                primary = syms[0]
                self.training_blocks.append(f"{hdr}def caller(x): return {primary}(x)\n")
                for s in syms:
                    self.training_blocks.append(f"# repo: {s}\ndef caller(x): return {s}(x)\n")
                    self.training_blocks.append(f"# repo: {s}\nres = {s}(x)\n")

            sym_summary = ", ".join(syms[:5]) if syms else "script entry point"
            q_file = f"What is in {rel}?"
            a_file = f"{rel} defines {sym_summary}."
            self.qa_pairs.append((q_file, a_file, f"# repo: {rel} -> {sym_summary}"))

        top_syms = list(self.symbols.keys())[:5]
        self.global_header = f"# repo: {', '.join(top_syms)}\n" if top_syms else ""

        total_lines = sum(f["lines"] for f in self.files)
        q_repo = "What does this codebase do?"
        a_repo = (
            f"{self.repo_dir.name} is a Python codebase with {len(self.files)} files "
            f"({total_lines} lines) defining {', '.join(top_syms)}."
        )
        self.qa_pairs.append((q_repo, a_repo, self.global_header.strip()))

        for q, a, ctx in self.qa_pairs:
            self.training_blocks.append(f"{ctx}\n# Q: {q}\n# A: {a}\n")

    def adapt(self, steps: int = 45) -> float:
        if not self.training_blocks:
            return 0.0
        pad_id = self.tokenizer.token_to_id("<pad>") or 0
        encoded = [self.tokenizer.encode(b).ids[: self.config.max_seq_len] for b in self.training_blocks]
        max_l = max(len(seq) for seq in encoded)

        x_batch, y_batch = [], []
        for seq in encoded:
            padded = seq + [pad_id] * (max_l - len(seq))
            x_batch.append(padded[:-1])
            y_subs = padded[1:]
            y_batch.append([tok if i < len(seq) - 1 else -100 for i, tok in enumerate(y_subs)])

        x_t = torch.tensor(x_batch, dtype=torch.long, device=DEVICE)
        y_t = torch.tensor(y_batch, dtype=torch.long, device=DEVICE)

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1.2e-3, weight_decay=0.01)
        t0 = time.time()

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits = self.model(x_t)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_t.view(-1),
                    ignore_index=-100,
                )
            loss.backward()
            opt.step()

        self.model.eval()
        return round(time.time() - t0, 2)

    @torch.no_grad()
    def generate_neural_answer(self, context_header: str, question: str, max_new_tokens: int = 64) -> tuple[str, float]:
        """Run CodebaseLlama on the retrieved AST context + question to generate a complete neural response."""
        self.model.eval()
        prompt = f"{context_header}\n# Q: {question}\n# A:"
        ids = self.tokenizer.encode(prompt).ids
        start_len = len(ids)
        confs = []

        for _ in range(max_new_tokens):
            win = ids[-self.config.max_seq_len :]
            x = torch.tensor([win], dtype=torch.long, device=DEVICE)
            logits = self.model(x)[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.argmax(probs).item())
            confs.append(float(probs[next_id].item()) * 100.0)
            ids.append(next_id)
            if "\n" in self.tokenizer.decode([next_id]):
                break

        ans = self.tokenizer.decode(ids[start_len:]).splitlines()[0].strip()
        avg_conf = sum(confs) / max(len(confs), 1)
        return ans, avg_conf

    def answer_question(self, question: str):
        """
        Full-Codebase AST-RAG + CodebaseLlama Neural QA Engine.
        Analyzes all files, symbols, call graphs, and docstrings to answer any question.
        """
        q_lower = question.lower().strip()
        tokens = re.findall(r"[a-zA-Z0-9_./]+", q_lower)

        matched_files = [
            f for f in self.files
            if f["file"].lower() in q_lower or Path(f["file"]).name.lower() in q_lower
            or Path(f["file"]).stem.lower() in tokens
        ]
        matched_syms = [
            s for s in self.symbols
            if s.lower() in tokens or s.lower() in q_lower
        ]

        print(f"\n  {BOLD}{MAGENTA}🧠 CODEBASE-LM FULL REPOSITORY ANALYSIS{RESET}")
        print(f"  {DIM}{'─' * 72}{RESET}")

        # Case 1: Specific symbol query
        if matched_syms:
            sym = max(matched_syms, key=len)
            det = self.symbol_details[sym]
            callers = [
                f["file"] for f in self.files if sym in f["calls"] and f["file"] != det["file"]
            ]
            if det["type"] == "class":
                ctx_hdr = f"# repo: {det['file']} | class {sym}"
                q_canonical = f"Where is class {sym} defined?"
            else:
                ctx_hdr = f"# repo: {det['file']} | {det['signature']}"
                q_canonical = f"Where is {sym} defined?"

            neural_ans, conf = self.generate_neural_answer(ctx_hdr, q_canonical)

            print(f"  {DIM}Matched Symbol      :{RESET} {BOLD}{CYAN}{sym}{RESET} ({det['type']})")
            print(f"  {DIM}Defined In          :{RESET} {GREEN}{det['file']}{RESET} (Line {det['line']})")
            print(f"  {DIM}Signature           :{RESET} {YELLOW}{det['signature']}{RESET}")
            if det["doc"]:
                print(f"  {DIM}Docstring Purpose   :{RESET} {det['doc']}")
            print(
                f"  {DIM}Cross-File Callers  :{RESET} {', '.join(callers) if callers else 'Used locally / entry point'}"
            )
            print(
                f"  {DIM}CodebaseLlama Output:{RESET} {GREEN}{neural_ans}{RESET} {DIM}({conf:.1f}% neural conf){RESET}\n"
            )
            return

        # Case 2: Specific file query
        if matched_files:
            f = matched_files[0]
            syms_str = ", ".join(f["symbols"][:5]) if f["symbols"] else "script entry point"
            ctx_hdr = f"# repo: {f['file']} -> {syms_str}"
            neural_ans, conf = self.generate_neural_answer(ctx_hdr, f"What is in {f['file']}?")

            print(f"  {DIM}Target File         :{RESET} {BOLD}{CYAN}{f['file']}{RESET} ({f['lines']} lines)")
            if f["doc"]:
                print(f"  {DIM}Module Summary      :{RESET} {f['doc']}")
            if f["classes"]:
                cls_list = [f"{c['name']}({', '.join(c['methods'][:4])})" for c in f["classes"]]
                print(f"  {DIM}Classes Defined     :{RESET} {YELLOW}{', '.join(cls_list)}{RESET}")
            if f["functions"]:
                fn_list = [fn["sig"] for fn in f["functions"][:6]]
                print(f"  {DIM}Functions Defined   :{RESET} {GREEN}{', '.join(fn_list)}{RESET}")
            if f["imports"]:
                print(f"  {DIM}Key Imports         :{RESET} {', '.join(f['imports'][:5])}")
            print(
                f"  {DIM}CodebaseLlama Output:{RESET} {GREEN}{neural_ans}{RESET} {DIM}({conf:.1f}% neural conf){RESET}\n"
            )
            return

        # Case 3: Whole-repository architecture / overview question
        total_lines = sum(f["lines"] for f in self.files)
        total_classes = sum(len(f["classes"]) for f in self.files)
        total_funcs = sum(len(f["functions"]) for f in self.files)
        ctx_hdr = self.global_header.strip()
        neural_ans, conf = self.generate_neural_answer(ctx_hdr, "What does this codebase do?")

        print(
            f"  {DIM}Repository Scope    :{RESET} {BOLD}{len(self.files)} Python files{RESET} "
            f"({total_lines:,} lines of code | {total_classes} classes | {total_funcs} functions)"
        )
        print(f"  {DIM}Core Architecture   :{RESET}")
        for f in self.files:
            if f["symbols"] or f["doc"]:
                summary = f["doc"][:65] + "..." if f["doc"] else ", ".join(f["symbols"][:4])
                print(f"    • {CYAN}{f['file']:<30}{RESET} -> {summary}")
        print(
            f"  {DIM}CodebaseLlama Output:{RESET} {GREEN}{neural_ans}{RESET} {DIM}({conf:.1f}% neural conf){RESET}\n"
        )

    def resolve_header_and_prompt(self, raw_prompt: str) -> tuple[str, str, str]:
        prompt = raw_prompt.strip()
        if prompt.startswith("@"):
            parts = prompt.split(" ", 1)
            file_query = parts[0][1:].lower()
            clean_prompt = (parts[1] if len(parts) > 1 else "def caller(x): return ").rstrip() + " "
            for rel, hdr in self.file_headers.items():
                if file_query in rel.lower():
                    return hdr, clean_prompt, ""
            return self.global_header, clean_prompt, ""

        last_word = prompt.replace("(", " ").replace("=", " ").split()[-1] if prompt else ""
        if last_word and last_word not in {"return", "def", "res", "x"} and not prompt.endswith(" "):
            matched = [s for s in self.symbols if s.lower().startswith(last_word.lower())]
            if matched and last_word not in self.symbols:
                base_prompt = prompt[: -len(last_word)]
                return f"# repo: {matched[0]}\n", base_prompt, last_word
        if prompt.endswith("return"):
            prompt = prompt + " "
        return self.global_header, prompt, ""

    @torch.no_grad()
    def complete(self, raw_prompt: str, max_new_tokens: int = 12):
        header, model_prompt, partial = self.resolve_header_and_prompt(raw_prompt)
        out = {}
        runs = [
            ("no_ctx", self.base_model, model_prompt + partial),
            ("with_ctx", self.model, header + model_prompt),
        ]
        for mode, net, full_prompt in runs:
            net.eval()
            ids = self.tokenizer.encode(full_prompt).ids
            start_len = len(ids)
            confidences = []

            for _ in range(max_new_tokens):
                win = ids[-self.config.max_seq_len :]
                x = torch.tensor([win], dtype=torch.long, device=DEVICE)
                logits = net(x)[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.argmax(probs).item())
                conf = float(probs[next_id].item()) * 100.0
                ids.append(next_id)
                tok_str = self.tokenizer.decode([next_id])
                confidences.append((tok_str, conf))
                if "\n" in tok_str:
                    break

            gen = self.tokenizer.decode(ids[start_len:]).splitlines()[0]
            disp_prompt = (model_prompt + partial) if mode == "no_ctx" else model_prompt
            avg_conf = sum(c for _, c in confidences) / max(len(confidences), 1)
            out[mode] = {
                "prompt": disp_prompt,
                "text": gen,
                "conf": avg_conf,
                "header": header.strip(),
            }
        return out

    def verify(self, code_line: str):
        builtins_set = set(dir(__builtins__)) | {
            "print", "len", "range", "isinstance", "max", "min", "sum", "sorted",
            "set", "list", "dict", "tuple", "int", "float", "str", "bool", "open",
            "enumerate", "zip", "map", "filter", "any", "all", "round", "abs", "super",
        }
        try:
            tree = ast.parse(code_line)
        except SyntaxError as e:
            print(f"  {RED}[SYNTAX ERROR]{RESET} {e}")
            return

        found_calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        if not found_calls:
            print(f"  {DIM}No function/class invocations found in snippet.{RESET}")
            return

        for fn in found_calls:
            if fn in self.symbols:
                print(f"  {GREEN}✔ [VERIFIED]{RESET}     {BOLD}{fn}(){RESET} -> Defined in {CYAN}{self.symbols[fn]}{RESET}")
            elif fn in builtins_set:
                print(f"  {CYAN}● [BUILTIN]{RESET}      {BOLD}{fn}(){RESET} -> Python standard builtin")
            else:
                matches = difflib.get_close_matches(fn, self.symbols.keys(), n=2, cutoff=0.45)
                hint = (
                    f"Did you mean {GREEN}{matches[0]}(){RESET} ({self.symbols[matches[0]]})?"
                    if matches
                    else "No close match in repo."
                )
                print(f"  {RED}✖ [HALLUCINATED]{RESET} {BOLD}{fn}(){RESET} -> NOT in repo! {hint}")

    def audit_repository(self):
        print(f"\n{BOLD}REPOSITORY-WIDE AST CALL INTEGRITY AUDIT ({self.repo_dir.name}){RESET}")
        print(f"{DIM}{'─' * 76}{RESET}")
        total_verified = 0
        for f in self.files:
            repo_calls = [f"{c}()" for c in f["calls"] if c in self.symbols]
            total_verified += len(repo_calls)
            call_str = ", ".join(repo_calls[:4]) if repo_calls else "no cross-symbol calls"
            print(f"  {GREEN}✔{RESET} {CYAN}{f['file']:<32}{RESET} {DIM}│{RESET} {call_str}")
        print(f"{DIM}{'─' * 76}{RESET}")
        print(f"{GREEN}✔ Audited {len(self.files)} files: {total_verified} verified repository symbol invocations.{RESET}\n")


def print_ast_map(engine: RepoEngine):
    print(f"\n{BOLD}AST ARCHITECTURE MAP ({engine.repo_dir.name}){RESET}")
    print(f"{DIM}{'─' * 76}{RESET}")
    for f in engine.files:
        syms = ", ".join(f["symbols"][:5]) if f["symbols"] else "script"
        print(f"  {CYAN}{f['file']:<32}{RESET} {DIM}│{RESET} {syms}")
    print(f"{DIM}{'─' * 76}{RESET}")


def is_natural_language_question(text: str) -> bool:
    t = text.strip()
    if t.startswith("@") or t.startswith("def ") or t.startswith("class ") or "=" in t:
        return False
    q_words = {"what", "where", "how", "which", "who", "why", "explain", "tell", "list", "show", "describe", "does", "is"}
    first_word = t.lower().split()[0].rstrip("?:") if t.split() else ""
    return t.endswith("?") or first_word in q_words


def interactive_shell(engine: RepoEngine):
    total_lines = sum(f["lines"] for f in engine.files)
    print(f"\n{BOLD}{GREEN}⚡ CodebaseLM Full-Repo Analyzer & Neural QA Shell (CodebaseLlama 5.87M on {DEVICE.type.upper()}){RESET}")
    print(
        f"Read & Indexed {BOLD}{len(engine.files)}{RESET} files ({BOLD}{total_lines:,}{RESET} lines) "
        f"& {BOLD}{len(engine.symbols)}{RESET} symbols in {CYAN}{engine.repo_dir}{RESET}"
    )
    t_adapt = engine.adapt(steps=45)
    print(f"{GREEN}✔ Repo-Brain & QA Neural Adaptation finished in {t_adapt}s ({len(engine.qa_pairs)} Repo-QA pairs learned){RESET}")
    print(
        f"{DIM}Ask any English question about the codebase | type code to autocomplete | `:verify <code>` | `:audit` | `:map` | `:quit`{RESET}\n"
    )

    while True:
        try:
            user_in = input(f"{BOLD}{CYAN}cblm>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting CodebaseLM shell.")
            break

        if not user_in:
            continue
        if user_in in {":q", ":quit", "exit", "quit"}:
            break
        if user_in == ":map":
            print_ast_map(engine)
            continue
        if user_in == ":audit":
            engine.audit_repository()
            continue
        if user_in.startswith(":verify "):
            engine.verify(user_in[len(":verify ") :].strip())
            print()
            continue
        if user_in.startswith(":ask "):
            engine.answer_question(user_in[len(":ask ") :].strip())
            continue

        if is_natural_language_question(user_in):
            engine.answer_question(user_in)
            continue

        res = engine.complete(user_in)
        print(f"  {DIM}AST Header Injected   :{RESET} {YELLOW}{res['with_ctx']['header']}{RESET}")
        print(
            f"  {DIM}Unadapted (No AST)    :{RESET} {res['no_ctx']['prompt']}{RED}{res['no_ctx']['text']}{RESET} "
            f"{DIM}({res['no_ctx']['conf']:.1f}% conf){RESET}"
        )
        print(
            f"  {GREEN}Repo-Brain (With AST) :{RESET} {res['with_ctx']['prompt']}{BOLD}{GREEN}{res['with_ctx']['text']}{RESET} "
            f"{GREEN}({res['with_ctx']['conf']:.1f}% conf){RESET}\n"
        )


def main():
    args = sys.argv[1:]
    repo_dir = Path.cwd()

    if not args:
        engine = RepoEngine(repo_dir)
        interactive_shell(engine)
        return

    cmd = args[0]
    if cmd == "scan":
        target = Path(args[1]) if len(args) > 1 else repo_dir
        engine = RepoEngine(target)
        print_ast_map(engine)
        elapsed = engine.adapt(steps=45)
        print(f"{GREEN}✔ Repo-Brain adapted to {len(engine.symbols)} symbols & {len(engine.qa_pairs)} QA pairs in {elapsed}s{RESET}")
    elif cmd == "ask" and len(args) > 1:
        engine = RepoEngine(repo_dir)
        engine.adapt(steps=45)
        engine.answer_question(" ".join(args[1:]))
    elif cmd == "audit":
        target = Path(args[1]) if len(args) > 1 else repo_dir
        engine = RepoEngine(target)
        engine.audit_repository()
    elif cmd == "verify" and len(args) > 1:
        engine = RepoEngine(repo_dir)
        engine.verify(" ".join(args[1:]))
    else:
        engine = RepoEngine(repo_dir)
        interactive_shell(engine)


if __name__ == "__main__":
    main()