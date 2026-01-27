import argparse
import re
import shlex
import subprocess
from pathlib import Path
from typing import List


def _load_commands(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    commands: List[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        commands.append(stripped)
    if not commands:
        raise ValueError(f"No runnable lines found in {path}")
    return commands


def run_commands(commands: List[str], keep_going: bool, use_shell: bool) -> None:
    for idx, cmd in enumerate(commands, start=1):
        rewritten = _ensure_python3(cmd, use_shell)
        print(f"[{idx}/{len(commands)}] Running: {rewritten}")
        proc = subprocess.run(rewritten if use_shell else shlex.split(rewritten))
        if proc.returncode != 0:
            print(f"[{idx}/{len(commands)}] FAILED (exit {proc.returncode})")
            if not keep_going:
                print("Stopping due to failure.")
                return
        else:
            print(f"[{idx}/{len(commands)}] Done.")
    print("All commands completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential experiment runner.")
    parser.add_argument(
        "--commands-file",
        required=True,
        help="Text file with one command per line; lines starting with # are ignored.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue even if a command fails (default: stop on first failure).",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Run commands through the shell (needed if you use pipes/redirection).",
    )
    return parser.parse_args()


def _ensure_python3(cmd: str, use_shell: bool) -> str:
    """
    Replace a leading 'python' command with 'python3' without touching 'python3' or paths.
    Works for both shell and non-shell execution.
    """
    # quick exit if already using python3
    if re.search(r"(^|\s)python3(\s|$)", cmd):
        return cmd
    # replace only leading standalone 'python' (allows leading whitespace)
    return re.sub(r"^(\s*)python(\s|$)", r"\1python3\2", cmd, count=1)


def main():
    args = parse_args()
    commands = _load_commands(Path(args.commands_file))
    run_commands(commands, keep_going=args.keep_going, use_shell=args.shell)


if __name__ == "__main__":
    main()
