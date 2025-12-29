"""
RAG Recipes CLI interface.

Provides command-line access to RAG pipeline stages:
- rag-recipes index
- rag-recipes retrieve
- rag-recipes generate
"""

import sys
import argparse
import subprocess


def run_module(module_name: str, args):
    """Run a module using python -m to ensure proper resolution."""
    return subprocess.run(
        [sys.executable, "-m", module_name] + list(args)
    )


def main():
    parser = argparse.ArgumentParser(
        prog="rag-recipes",
        description="RAG Recipes CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Run indexing pipeline")
    subparsers.add_parser("retrieve", help="Run retrieval pipeline")
    subparsers.add_parser("generate", help="Run generation pipeline")

    args, remaining = parser.parse_known_args()

    if args.command == "index":
        sys.exit(run_module("src.indexing", remaining).returncode)

    elif args.command == "retrieve":
        sys.exit(run_module("src.retrieval", remaining).returncode)

    elif args.command == "generate":
        sys.exit(run_module("src.generation", remaining).returncode)

    else:
        parser.print_help()
        sys.exit(1)
