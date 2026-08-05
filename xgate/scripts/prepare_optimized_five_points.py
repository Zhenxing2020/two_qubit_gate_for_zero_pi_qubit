"""Extract completed X-gate optimizer results and select five spaced rows."""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def extract_params(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    params = []
    for index, line in enumerate(lines[:-1]):
        if line.strip() == "Drive parameters:":
            value = ast.literal_eval(lines[index + 1].strip())
            if len(value) != 5:
                raise ValueError(f"Expected five pulse parameters in {path}")
            params.append(value)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-log", required=True)
    parser.add_argument("--resume-log", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    params = extract_params(args.old_log) + extract_params(args.resume_log)
    if len(params) != args.expected:
        raise RuntimeError(
            f"Found {len(params)} optimized rows; expected {args.expected}"
        )

    indices = np.rint(np.linspace(0, args.expected - 1, 5)).astype(int)
    selected = np.asarray(params)[indices]
    table = pd.DataFrame(
        selected,
        columns=["tg", "drive_amp_1", "drive_amp_2", "detune_1", "detune_2"],
    )
    table.insert(0, "source_index", indices)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"selected source indices: {indices.tolist()}")
    print(table.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
