"""
Step 9: Run the model interactively -- the single entry point for using it.

Automatically decides, based on what you ask for, whether to generate a
single function/snippet or a full multi-file project:
  - "write a function that scrapes titles"        -> single snippet
  - "build a streamlit app that visualizes a csv"  -> full project, written
                                                       to generated_projects/

Every generation is checked beyond simple syntax validity: undefined names
are detected via static analysis, and missing imports for well-known
libraries (requests, pandas, streamlit, etc.) are automatically added
before you see the result. This catches the single most common "runs and
immediately crashes" failure -- a forgotten import -- without needing to
actually execute anything (which isn't generally safe or possible, since
much of this code needs network access, installed packages, or API keys
that vary machine to machine).

Honest limits: this does NOT catch logic bugs (code that runs but computes
the wrong thing), and does NOT catch missing imports for obscure libraries
outside the known-imports table in code_checks.py. Always read generated
code before running it.

Usage:
    python 09_inference.py --model checkpoints/instruct/final.pt
    python 09_inference.py --model export/model_int8.pt --tokenizer export/tokenizer.json
    python 09_inference.py --model export/model_int8.pt --tokenizer export/tokenizer.json --device cpu
"""

import argparse
import difflib
import importlib.util
import os
import re
import time
import torch
from tokenizers import Tokenizer

from gpt_model import GPT, GPTConfig
from code_checks import is_valid_python, check_and_fix

# Hard wall-clock cap on total project generation time, in seconds. Real
# incident that prompted this: a project request that didn't match any
# verified template chained multiple slow generation calls (structure +
# several files, each with retries) with no upper bound, appearing to hang
# indefinitely and requiring a forced process kill to stop. This caps the
# worst case: once exceeded, generation stops and whatever files completed
# so far are returned, rather than continuing with no limit.
PROJECT_TIME_BUDGET_SECONDS = 90
PROJECT_FILE_MAX_ATTEMPTS = 2  # lower than the single-function default (4) --
                                 # each retry multiplies wait time, and project
                                 # mode should fail fast rather than retry hard

# Similarity threshold above which a request is considered "close enough"
# to a verified curated example to return it directly instead of
# generating. This trades some flexibility for a real guarantee: anything
# matched this way is EXACTLY the hand-verified, execution-tested response
# from 06_prepare_instruct_data.py, not a model guess. Lower this to catch
# more requests (with more risk of a poor-fit match); raise it to only
# catch near-exact phrasing.
RETRIEVAL_THRESHOLD = 0.6  # tuned via real positive/negative test cases -- see find_verified_match


def load_retrieval_index():
    """Loads verified (instruction, response) pairs directly from
    06_prepare_instruct_data.py's curated + bug-fix examples -- the ones
    that were individually hand-verified (and for bug-fixes, executed to
    confirm the fix is real). Deliberately excludes the much larger mined
    CodeAlpaca/docstring pairs, which weren't individually verified."""
    if not os.path.exists("06_prepare_instruct_data.py"):
        return []
    spec = importlib.util.spec_from_file_location("data06", "06_prepare_instruct_data.py")
    data06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data06)

    index = []
    for ex in data06.CURATED_EXAMPLES:
        index.append((ex["instruction"], ex["response"]))
    for ex in data06.BUGFIX_EXAMPLES:
        instruction = f"Fix the bug in this function:\n{ex['buggy']}"
        index.append((instruction, ex["fixed"]))
    return index


STOPWORDS = {"a", "an", "the", "that", "this", "for", "to", "of", "in", "on", "with",
             "write", "make", "create", "build", "give", "me", "and", "function", "please"}


def _normalize_word(w: str) -> str:
    """Crude suffix stripping (no NLP library needed) so 'scrapes' and
    'scrape', or 'titles' and 'title', are recognized as the same word.
    Deliberately conservative: only strips 's'/'es' (plurals), not 'ed'/
    'ing', since those caused a real false positive -- "linked" and
    "links" both stemmed to "link", wrongly equating "linked list" (a
    data structure) with "links" (hyperlinks)."""
    for suffix in ("es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def _words_related(a: str, b: str) -> bool:
    """True if two words are the same after normalization, OR one contains
    the other as a substring (catches compound-word relationships like
    'page' / 'webpage')."""
    na, nb = _normalize_word(a), _normalize_word(b)
    if na == nb:
        return True
    return (a in b or b in a) and min(len(a), len(b)) >= 4  # avoid trivial short-substring noise


def _word_overlap_score(a: str, b: str) -> float:
    """Jaccard-style similarity on meaningful words, with normalization --
    catches genuine rephrasing that character-level difflib misses (e.g.
    'scrape titles off a page' vs 'write a function that scrapes titles
    from a webpage' share the real intent even though few words are an
    exact character match)."""
    words_a = [w for w in re.findall(r'\w+', a.lower()) if w not in STOPWORDS]
    words_b = [w for w in re.findall(r'\w+', b.lower()) if w not in STOPWORDS]
    if not words_a or not words_b:
        return 0.0
    matched_a = sum(1 for wa in words_a if any(_words_related(wa, wb) for wb in words_b))
    matched_b = sum(1 for wb in words_b if any(_words_related(wa, wb) for wa in words_a))
    # symmetric: average of "how much of A is covered by B" and vice versa
    return ((matched_a / len(words_a)) + (matched_b / len(words_b))) / 2


def find_verified_match(instruction, index, threshold=RETRIEVAL_THRESHOLD):
    """Returns (response, similarity) for the closest verified example if
    it's above threshold, else None. Both signals are computed on CONTENT
    WORDS ONLY (stopwords/boilerplate stripped) -- comparing full raw
    sentences let shared filler like "write a function that" inflate the
    character-similarity score even when the actual requested content was
    completely unrelated (this was a real false positive caught in
    testing: "reverses a linked list" scored 0.67 against "scrapes all
    links from a webpage" purely from the shared boilerplate wording)."""
    if not index:
        return None
    best_response, best_score = None, 0.0
    instr_words = [w for w in re.findall(r'\w+', instruction.lower()) if w not in STOPWORDS]
    instr_content = " ".join(instr_words)
    for verified_instruction, response in index:
        verified_words = [w for w in re.findall(r'\w+', verified_instruction.lower()) if w not in STOPWORDS]
        verified_content = " ".join(verified_words)

        char_ratio = difflib.SequenceMatcher(None, instr_content, verified_content).ratio()
        word_ratio = _word_overlap_score(instruction, verified_instruction)
        score = max(char_ratio, word_ratio)
        if score > best_score:
            best_score, best_response = score, response
    if best_score >= threshold:
        return best_response, best_score
    return None

PROJECT_OUT_DIR = "generated_projects"

# Heuristic classifier: keyword-based, not a learned model. Deliberately
# simple and transparent so its behavior is predictable -- if it guesses
# wrong, you can always rephrase (e.g. add "project" or "just a function").
PROJECT_KEYWORDS = [
    "project", "app", "application", "system", "website", "api service",
    "full pipeline", "multiple files", "file structure", "build me a",
    "streamlit app", "flask api", "cli tool", "command-line tool", "package",
]
FUNCTION_OVERRIDE_KEYWORDS = ["just a function", "single function", "one function", "just the code", "snippet"]


def looks_like_project_request(instruction: str) -> bool:
    """Deterministic keyword heuristic -- not ML, not perfect. Explicit
    single-function phrasing always wins over project keywords.

    Uses word-boundary matching (\\b), not plain substring matching --
    plain substring checks caused a real bug where "scores.append(name)"
    was misclassified as a project request because "app" is a substring
    of "append". Word boundaries prevent this class of false positive.
    """
    lower = instruction.lower()
    if any(re.search(r'\b' + re.escape(kw) + r'\b', lower) for kw in FUNCTION_OVERRIDE_KEYWORDS):
        return False
    return any(re.search(r'\b' + re.escape(kw) + r'\b', lower) for kw in PROJECT_KEYWORDS)


# Only matches a filename when it is the WHOLE content of a line (after
# stripping tree-drawing characters and an optional trailing "# comment"),
# and only for a whitelisted set of real file extensions. This replaced an
# earlier version that matched ANY "word.word" pattern anywhere in the
# text -- which incorrectly treated code fragments like "requests.get" or
# "line.split" as filenames when the model's plan generation went off the
# rails and produced regular code instead of a real file tree.
KNOWN_FILE_EXTENSIONS = {
    "py", "txt", "md", "json", "yml", "yaml", "cfg", "toml", "ini",
    "html", "css", "js", "csv", "sql", "env", "sh",
}
_ext_pattern = "|".join(re.escape(e) for e in KNOWN_FILE_EXTENSIONS)
FILENAME_LINE_RE = re.compile(
    r'^[\s├└│─\-\*>]*([A-Za-z0-9_\-]+\.(?:' + _ext_pattern + r'))\s*(?:#.*)?$'
)


def parse_filenames(structure_text):
    filenames = []
    for line in structure_text.split("\n"):
        match = FILENAME_LINE_RE.match(line)
        if match:
            filenames.append(match.group(1))
    return filenames


def load_model(model_path, device):
    # weights_only=False: safe here since this checkpoint came from our own
    # pipeline (05/07/08), not downloaded from an untrusted source.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config: GPTConfig = ckpt["config"]
    config.dropout = 0.0

    model = GPT(config)
    is_quantized = "int8" in os.path.basename(model_path)
    if is_quantized:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, config


def generate(model, tokenizer, instruction, device, args, is_python=True):
    """Generate one response, with syntax + undefined-name retry, and
    auto-fixing of any known missing imports before returning."""
    inst_id = tokenizer.token_to_id("<|instruction|>")
    resp_id = tokenizer.token_to_id("<|response|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    inst_ids = tokenizer.encode(instruction).ids
    prompt_ids = [inst_id] + inst_ids + [resp_id]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    best_text, best_ok = None, False
    for attempt in range(args.max_attempts):
        with torch.no_grad():
            out = model.generate(
                idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=(args.top_p if args.top_p and args.top_p > 0 else None),
                repetition_penalty=args.repetition_penalty,
            )
        generated_ids = out[0, len(prompt_ids):].tolist()
        if eot_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eot_id)]
        text = tokenizer.decode(generated_ids)

        if not is_python:
            return text, True  # non-Python output (e.g. requirements.txt, a plan) skips code checks

        ok, fixed_text, _remaining = check_and_fix(text)
        if ok:
            return fixed_text, True
        if best_text is None:
            best_text = fixed_text  # keep the auto-fixed version even on a losing attempt

    return best_text, best_ok


def load_project_retrieval_index():
    """Loads verified (project instruction -> {structure, files}) bundles
    directly from 06's PROJECT_STRUCTURE_EXAMPLES + PROJECT_FILE_EXAMPLES.
    Real testing showed project-structure GENERATION is still unreliable
    even after training improvements -- this retrieval path guarantees a
    correct result for the specific project archetypes we've curated
    (Streamlit CSV viewer, Flask health API, CLI tool, image filter app,
    web scraper), the same way single-function retrieval already does."""
    if not os.path.exists("06_prepare_instruct_data.py"):
        return []
    spec = importlib.util.spec_from_file_location("data06b", "06_prepare_instruct_data.py")
    data06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data06)

    bundles = []
    for struct_ex in data06.PROJECT_STRUCTURE_EXAMPLES:
        files = {}
        for file_ex in data06.PROJECT_FILE_EXAMPLES:
            if file_ex["plan"] == struct_ex["response"]:
                files[file_ex["filename"]] = file_ex["response"]
        bundles.append((struct_ex["instruction"], struct_ex["response"], files))
    return bundles


def find_verified_project(instruction, project_index, threshold=RETRIEVAL_THRESHOLD):
    """Same matching approach as find_verified_match, applied to whole
    project bundles instead of single functions. Only considers bundles
    that actually HAVE associated file content -- some curated structure
    examples don't have matching file-content examples yet, and matching
    to one of those would silently produce an empty project folder, which
    is worse than falling back to generation."""
    if not project_index:
        return None
    best_bundle, best_score = None, 0.0
    instr_words = [w for w in re.findall(r'\w+', instruction.lower()) if w not in STOPWORDS]
    instr_content = " ".join(instr_words)
    for verified_instruction, structure_text, files in project_index:
        if not files:
            continue  # no point matching to a template with no verified file content
        verified_words = [w for w in re.findall(r'\w+', verified_instruction.lower()) if w not in STOPWORDS]
        verified_content = " ".join(verified_words)
        char_ratio = difflib.SequenceMatcher(None, instr_content, verified_content).ratio()
        word_ratio = _word_overlap_score(instruction, verified_instruction)
        score = max(char_ratio, word_ratio)
        if score > best_score:
            best_score, best_bundle = score, (structure_text, files)
    if best_score >= threshold:
        return best_bundle[0], best_bundle[1], best_score
    return None


def handle_project_request(model, tokenizer, instruction, device, args, retrieval_index=None, project_index=None):
    if project_index:
        match = find_verified_project(instruction, project_index)
        if match:
            structure_text, files, similarity = match
            print(f"\n[using a verified project template -- {similarity:.0%} match]")
            print(structure_text)

            project_name = re.sub(r'[^a-z0-9_]+', '_', instruction.lower())[:40].strip('_') or "generated_project"
            project_dir = os.path.join(PROJECT_OUT_DIR, project_name)
            os.makedirs(project_dir, exist_ok=True)
            for filename, content in files.items():
                with open(os.path.join(project_dir, filename), "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  - {filename} (verified)")
            print(f"\nProject written to: {project_dir}/\n")
            return

    print("\n[detected: project request -- planning file structure]")
    structure_text, _ = generate(model, tokenizer, f"Give me a file structure for {instruction}.",
                                  device, args, is_python=False)
    print(structure_text)

    filenames = parse_filenames(structure_text)
    if not filenames:
        print("[couldn't find filenames in the plan -- falling back to a single-snippet answer]")
        handle_function_request(model, tokenizer, instruction, device, args, retrieval_index)
        return

    project_name = re.sub(r'[^a-z0-9_]+', '_', instruction.lower())[:40].strip('_') or "generated_project"
    project_dir = os.path.join(PROJECT_OUT_DIR, project_name)
    os.makedirs(project_dir, exist_ok=True)

    print(f"\n[generating {len(filenames)} file(s) into {project_dir}/, "
          f"up to {PROJECT_TIME_BUDGET_SECONDS}s budget]")
    start_time = time.time()
    for filename in filenames:
        elapsed = time.time() - start_time
        if elapsed > PROJECT_TIME_BUDGET_SECONDS:
            remaining = filenames[filenames.index(filename):]
            print(f"  [time budget exceeded ({elapsed:.0f}s) -- stopping here. "
                  f"Skipped: {', '.join(remaining)}]")
            break

        is_python = filename.endswith(".py")
        file_instruction = f"Given this project structure:\n{structure_text}\n\nWrite the content of {filename}."
        # temporarily cap retries for project-mode generation specifically
        original_max_attempts = args.max_attempts
        args.max_attempts = min(args.max_attempts, PROJECT_FILE_MAX_ATTEMPTS)
        content, ok = generate(model, tokenizer, file_instruction, device, args, is_python=is_python)
        args.max_attempts = original_max_attempts

        with open(os.path.join(project_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
        note = "" if ok or not is_python else " [WARNING: may still have undefined names]"
        print(f"  - {filename}{note}")

    print(f"\nProject written to: {project_dir}/")
    print("Review every file before running it -- generated code is a first draft, not verified.\n")


def handle_function_request(model, tokenizer, instruction, device, args, retrieval_index=None):
    if retrieval_index:
        match = find_verified_match(instruction, retrieval_index)
        if match:
            response, similarity = match
            print(f"[using a verified example -- {similarity:.0%} match to a hand-tested pattern]")
            print(response)
            print()
            return

    text, ok = generate(model, tokenizer, instruction, device, args, is_python=True)
    if not ok:
        print("[note: response may contain undefined names not auto-fixable -- review carefully]")
    print(text)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                         help="Force a specific device. Default: auto-detect (cuda if available, "
                              "else cpu). Quantized (int8) models always run on CPU regardless of "
                              "this setting, since PyTorch's dynamic quantization has no CUDA kernel.")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dynamic int8 quantization (used for export/model_int8.pt) only has CPU
    # kernels -- there is no CUDA implementation of quantized::linear_dynamic.
    # Loading a quantized checkpoint on cuda succeeds silently, but the first
    # real generate() call (i.e. anything that misses retrieval) crashes with
    # NotImplementedError. Force CPU here so that can't happen, even if
    # --device was left unset or was explicitly (mistakenly) set to cuda.
    is_quantized = "int8" in os.path.basename(args.model)
    if is_quantized and device == "cuda":
        print("Note: quantized (int8) model detected -- forcing CPU (no CUDA kernel for dynamic quantization).")
        device = "cpu"

    print(f"Loading model on {device}...")
    model, config = load_model(args.model, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    retrieval_index = load_retrieval_index()
    project_index = load_project_retrieval_index()
    print(f"Loaded {len(retrieval_index)} verified functions + {len(project_index)} verified project templates")
    print(f"Loaded {len(retrieval_index)} verified examples for exact/near-match lookup")

    n = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n:,} params. Type an instruction (or 'quit' to exit).\n")

    while True:
        try:
            instruction = input(">>> ").strip()
            # if pasted text contains literal backslash-n instead of a real
            # newline (common when copy-pasting multi-line prompts into a
            # plain terminal input), convert it -- otherwise both generation
            # and retrieval matching see garbled code that doesn't match
            # anything they were trained/verified on
            if "\\n" in instruction and "\n" not in instruction:
                instruction = instruction.replace("\\n", "\n").replace("\\t", "\t")
        except (EOFError, KeyboardInterrupt):
            break
        if instruction.lower() in ("quit", "exit"):
            break
        if not instruction:
            continue

        if looks_like_project_request(instruction):
            handle_project_request(model, tokenizer, instruction, device, args, retrieval_index, project_index)
        else:
            handle_function_request(model, tokenizer, instruction, device, args, retrieval_index)


if __name__ == "__main__":
    main()