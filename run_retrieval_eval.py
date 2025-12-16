import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline.qa_retrieval import run_qa_retrieval


def parse_args():
    parser = argparse.ArgumentParser(description="Run QA retrieval-style evaluation over chunkers.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_qa_retrieval(args.config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
