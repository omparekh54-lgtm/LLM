"""
Backend API server for the web frontend. Reuses all logic from
09_inference.py directly (model loading, generation, retrieval, project
detection, syntax/undefined-name checking) -- this is a thin JSON API layer
on top of already-tested code, not a reimplementation.

Includes the project-generation time budget and reduced retry count that
were added to 09_inference.py after a real incident where an uncovered
project request hung indefinitely -- this backend never regresses that fix.

Run:
    pip install flask
    python backend.py
Then open http://localhost:5000 in a browser.
"""

import importlib.util
import io
import json
import os
import re
import time
import zipfile

import torch
from flask import Flask, request, jsonify, send_from_directory
from tokenizers import Tokenizer

_spec = importlib.util.spec_from_file_location("inference", "09_inference.py")
inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inference)

PROJECT_TIME_BUDGET_SECONDS = 90
PROJECT_FILE_MAX_ATTEMPTS = 2
HISTORY_PATH = "chat_history.json"

app = Flask(__name__, static_folder="static", static_url_path="")

_state = {}  # populated once at startup by load_model_once()


class GenArgs:
    max_new_tokens = 256
    temperature = 0.7
    top_k = 50
    top_p = 0.92
    repetition_penalty = 1.15
    max_attempts = 4


def find_available_checkpoint():
    candidates = [
        ("checkpoints/polish/final.pt", "tokenizer/tokenizer.json"),
        ("checkpoints/instruct/final.pt", "tokenizer/tokenizer.json"),
        ("export/model_int8.pt", "export/tokenizer.json"),
    ]
    for model_path, tok_path in candidates:
        if os.path.exists(model_path) and os.path.exists(tok_path):
            return model_path, tok_path
    return None, None


def load_model_once():
    if _state:
        return _state
    model_path, tok_path = find_available_checkpoint()
    if model_path is None:
        raise RuntimeError("No checkpoint found (checked checkpoints/polish, checkpoints/instruct, export/)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = inference.load_model(model_path, device)
    tokenizer = Tokenizer.from_file(tok_path)
    _state.update({
        "model": model, "tokenizer": tokenizer, "device": device,
        "retrieval_index": inference.load_retrieval_index(),
        "project_index": inference.load_project_retrieval_index(),
        "n_params": sum(p.numel() for p in model.parameters()),
        "model_path": model_path,
    })
    return _state


def get_project_response(ctx, instruction):
    """Same logic as 09_inference.py's handle_project_request, but returns
    structured data for JSON instead of printing -- WITH the time budget
    and reduced-retry protections included from the start."""
    args = GenArgs()

    match = inference.find_verified_project(instruction, ctx["project_index"])
    if match:
        structure_text, files, similarity = match
        return {"kind": "project", "source": "verified", "similarity": similarity,
                "structure": structure_text, "files": files, "note": None}

    structure_text, _ = inference.generate(
        ctx["model"], ctx["tokenizer"], f"Give me a file structure for {instruction}.",
        ctx["device"], args, is_python=False)
    filenames = inference.parse_filenames(structure_text)

    if not filenames:
        result = get_function_response(ctx, instruction)
        result["note"] = "Couldn't plan a file structure -- generated a single snippet instead."
        return result

    files = {}
    start_time = time.time()
    skipped = []
    args.max_attempts = min(args.max_attempts, PROJECT_FILE_MAX_ATTEMPTS)
    for filename in filenames:
        if time.time() - start_time > PROJECT_TIME_BUDGET_SECONDS:
            skipped.append(filename)
            continue
        is_py = filename.endswith(".py")
        file_instr = f"Given this project structure:\n{structure_text}\n\nWrite the content of {filename}."
        content, ok = inference.generate(ctx["model"], ctx["tokenizer"], file_instr,
                                          ctx["device"], args, is_python=is_py)
        files[filename] = content

    note = None
    if skipped:
        note = f"Time budget reached -- skipped: {', '.join(skipped)}"
    return {"kind": "project", "source": "generated", "similarity": None,
            "structure": structure_text, "files": files, "note": note}


def get_function_response(ctx, instruction):
    args = GenArgs()
    match = inference.find_verified_match(instruction, ctx["retrieval_index"])
    if match:
        response, similarity = match
        return {"kind": "function", "source": "verified", "similarity": similarity,
                "code": response, "warning": None}

    text, ok = inference.generate(ctx["model"], ctx["tokenizer"], instruction, ctx["device"], args, is_python=True)
    return {"kind": "function", "source": "generated", "similarity": None, "code": text,
            "warning": None if ok else "May contain undefined names that couldn't be auto-fixed -- review carefully."}


def get_response(ctx, instruction):
    if inference.looks_like_project_request(instruction):
        return get_project_response(ctx, instruction)
    return get_function_response(ctx, instruction)


def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    ctx = load_model_once()
    return jsonify({
        "model_path": ctx["model_path"], "device": ctx["device"],
        "n_params": ctx["n_params"],
        "n_verified_functions": len(ctx["retrieval_index"]),
        "n_verified_projects": len(ctx["project_index"]),
    })


@app.route("/api/history")
def history():
    return jsonify(load_history())


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    save_history([])
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    instruction = (data or {}).get("message", "").strip()
    if not instruction:
        return jsonify({"error": "empty message"}), 400
    if "\\n" in instruction and "\n" not in instruction:
        instruction = instruction.replace("\\n", "\n").replace("\\t", "\t")

    ctx = load_model_once()
    result = get_response(ctx, instruction)

    hist = load_history()
    hist.append({"role": "user", "content": instruction, "timestamp": time.time()})
    hist.append({"role": "assistant", "content": result, "timestamp": time.time()})
    save_history(hist)

    return jsonify(result)


@app.route("/api/download_project", methods=["POST"])
def download_project():
    files = (request.get_json() or {}).get("files", {})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    buf.seek(0)
    from flask import send_file
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="project.zip")


if __name__ == "__main__":
    print("Loading model...")
    load_model_once()
    print("Model loaded. Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
