"""
RAG Recipes CLI interface.

Provides command-line access to RAG pipeline phases via subcommands:
- rag-recipes index: Run indexing phase
- rag-recipes retrieve: Run retrieval phase
- rag-recipes generate: Run generation phase

Can be imported and used programmatically from other packages.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def get_rag_root():
    """Get rag_recipes root directory."""
    return Path(__file__).parent.parent


def run_indexing(*args, **kwargs):
    """Run indexing phase programmatically."""
    rag_root = get_rag_root()
    script = rag_root / "src" / "indexing.py"
    
    # Convert args and kwargs to command line format
    cmd = [sys.executable, str(script)] + list(args)
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        if value is not None:
            cmd.append(str(value))
    
    return subprocess.run(cmd, cwd=rag_root)


def run_retrieval(*args, **kwargs):
    """Run retrieval phase programmatically."""
    rag_root = get_rag_root()
    script = rag_root / "src" / "retrieval.py"
    
    # Convert args and kwargs to command line format
    cmd = [sys.executable, str(script)] + list(args)
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        if value is not None:
            cmd.append(str(value))
    
    return subprocess.run(cmd, cwd=rag_root)


def run_generation(*args, **kwargs):
    """Run generation phase programmatically."""
    rag_root = get_rag_root()
    script = rag_root / "src" / "generation.py"
    
    # Convert args and kwargs to command line format
    cmd = [sys.executable, str(script)] + list(args)
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        if value is not None:
            cmd.append(str(value))
    
    return subprocess.run(cmd, cwd=rag_root)


def main():
    """Main entry point for rag-recipes CLI with subcommands."""
    parser = argparse.ArgumentParser(
        prog="rag-recipes",
        description="RAG Recipes - Reusable RAG pipeline components"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index subcommand
    subparsers.add_parser("index", help="Run indexing phase")
    
    # Retrieve subcommand
    subparsers.add_parser("retrieve", help="Run retrieval phase")
    
    # Generate subcommand
    subparsers.add_parser("generate", help="Run generation phase")
    
    # Parse only the command, pass rest to subprocess
    args, remaining = parser.parse_known_args()
    
    if args.command == "index":
        rag_root = get_rag_root()
        script = rag_root / "src" / "indexing.py"
        result = subprocess.run([sys.executable, str(script)] + remaining, cwd=rag_root)
        sys.exit(result.returncode)
    elif args.command == "retrieve":
        rag_root = get_rag_root()
        script = rag_root / "src" / "retrieval.py"
        result = subprocess.run([sys.executable, str(script)] + remaining, cwd=rag_root)
        sys.exit(result.returncode)
    elif args.command == "generate":
        rag_root = get_rag_root()
        script = rag_root / "src" / "generation.py"
        result = subprocess.run([sys.executable, str(script)] + remaining, cwd=rag_root)
        sys.exit(result.returncode)
    else:
        parser.print_help()
        sys.exit(1)


def main_index():
    """Entry point for rag-index CLI command."""
    rag_root = get_rag_root()
    script = rag_root / "src" / "indexing.py"
    result = subprocess.run([sys.executable, str(script)] + sys.argv[1:], cwd=rag_root)
    sys.exit(result.returncode)


def main_retrieve():
    """Entry point for rag-retrieve CLI command."""
    rag_root = get_rag_root()
    script = rag_root / "src" / "retrieval.py"
    result = subprocess.run([sys.executable, str(script)] + sys.argv[1:], cwd=rag_root)
    sys.exit(result.returncode)


def main_generate():
    """Entry point for rag-generate CLI command."""
    rag_root = get_rag_root()
    script = rag_root / "src" / "generation.py"
    result = subprocess.run([sys.executable, str(script)] + sys.argv[1:], cwd=rag_root)
    sys.exit(result.returncode)

