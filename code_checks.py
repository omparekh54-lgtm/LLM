"""
Shared static-analysis checks used by 09_inference.py and 10_project_scaffold.py.

This goes one step beyond syntax checking (ast.parse) to catch a real,
common category of "runs and immediately crashes" bugs: forgetting to
import a library the code actually uses. This is likely the exact class of
error you'd hit pasting generated code into a fresh Jupyter cell -- a
NameError like "name 'requests' is not defined" because the import line
got dropped or wasn't included in a short generation.

Honest limits: this is static analysis, not execution. It cannot catch:
  - Logic bugs (code that runs fine but computes the wrong thing)
  - Errors that depend on runtime values (bad API responses, wrong types
    passed in, files that don't exist, network failures)
  - Missing THIRD-PARTY packages that aren't in the known-imports table
    below (only common/well-known libraries are covered)
It's a real, useful filter for one specific common failure mode -- not a
guarantee of correctness.
"""

import ast
import builtins


def is_valid_python(code: str) -> bool:
    """Check whether code at least parses as valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# Maps a name used in code (e.g. "requests", "BeautifulSoup") to the import
# statement that provides it. Covers the libraries actually used across this
# project's curated training examples and common general-purpose libraries.
KNOWN_IMPORTS = {
    "requests": "import requests",
    "BeautifulSoup": "from bs4 import BeautifulSoup",
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "st": "import streamlit as st",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "json": "import json",
    "csv": "import csv",
    "time": "import time",
    "math": "import math",
    "random": "import random",
    "hashlib": "import hashlib",
    "argparse": "import argparse",
    "sqlite3": "import sqlite3",
    "datetime": "import datetime",
    "Flask": "from flask import Flask",
    "jsonify": "from flask import jsonify",
    "Blueprint": "from flask import Blueprint",
    "telebot": "import telebot",
    "discord": "import discord",
    "praw": "import praw",
    "wraps": "from functools import wraps",
}


class NameUsageVisitor(ast.NodeVisitor):
    """Walks the AST tracking which names are defined (via def, class,
    assignment, import, function parameters, comprehension/loop variables,
    with-statement targets, except targets) versus which names are merely
    *used* (ast.Name with Load context). The difference, minus Python
    builtins, is names that are used but never defined anywhere -- a strong
    signal of an immediate NameError."""

    def __init__(self):
        self.defined = set(dir(builtins))
        self.used = set()

    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        for arg in node.args.args + node.args.kwonlyargs:
            self.defined.add(arg.arg)
        if node.args.vararg:
            self.defined.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined.add(node.args.kwarg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.defined.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)

    def visit_Assign(self, node):
        for target in node.targets:
            self._register_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._register_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.defined.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        for arg in node.args.args:
            self.defined.add(arg.arg)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        else:
            self.defined.add(node.id)

    def visit_Global(self, node):
        for name in node.names:
            self.defined.add(name)

    def _register_target(self, target):
        if isinstance(target, ast.Name):
            self.defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_target(elt)
        elif isinstance(target, ast.Starred):
            self._register_target(target.value)


def find_undefined_names(code: str):
    """Returns the set of names used in code that are never defined,
    imported, or a Python builtin -- likely NameErrors if run as-is."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()  # syntax check is a separate concern, handled elsewhere
    visitor = NameUsageVisitor()
    visitor.visit(tree)
    return visitor.used - visitor.defined


def auto_fix_missing_imports(code: str) -> str:
    """If any undefined names match a known library in KNOWN_IMPORTS,
    prepend the missing import line(s). This is a deterministic patch, not
    a retrain -- it only helps for the specific well-known libraries listed
    above."""
    undefined = find_undefined_names(code)
    missing_imports = []
    for name in sorted(undefined):
        if name in KNOWN_IMPORTS:
            import_line = KNOWN_IMPORTS[name]
            if import_line not in code:  # avoid duplicating an import already present elsewhere
                missing_imports.append(import_line)

    if not missing_imports:
        return code
    return "\n".join(missing_imports) + "\n" + code


def check_and_fix(code: str):
    """Full check: validate syntax, auto-fix any known missing imports,
    then report whether any *unfixable* undefined names remain.

    Returns (is_likely_runnable, fixed_code, remaining_undefined_names).
    """
    if not is_valid_python(code):
        return False, code, set()

    fixed_code = auto_fix_missing_imports(code)
    remaining_undefined = find_undefined_names(fixed_code)
    # exclude a small set of names that are commonly legitimately undefined
    # in a short, standalone snippet (e.g. a variable meant to be supplied
    # by the caller, or a placeholder) -- being too strict here would reject
    # otherwise-correct code for reasons unrelated to real bugs
    remaining_undefined -= {"self", "cls", "_"}

    is_likely_runnable = len(remaining_undefined) == 0
    return is_likely_runnable, fixed_code, remaining_undefined