#!/usr/bin/env python3
"""Inverse of tools/fix_spconv_checkpoint.py.

fix_spconv_checkpoint.py turns the raw spconv 2.x layout (out, kD, kH, kW, in) of every
5-D ``unet*``/``input_conv*`` weight into (kD, kH, kW, in, out) with permute(1, 2, 3, 4, 0).
This script applies permute(4, 0, 1, 2, 3) to get the raw layout back. The old
tools/test.py (main @ 6a75c37) permutes in memory, so it must be fed the raw layout.

Reuses is_spconv_weight/weight_layout/checkpoint_layout from tools/fix_spconv_checkpoint.py
instead of re-deriving the shape rule.

Exit codes: 0 written, 2 input is already in the raw layout (nothing written), 1 error.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fix_spconv_checkpoint import checkpoint_layout, is_spconv_weight  # noqa: E402


def unconvert_state_dict(state_dict):
    out = dict(state_dict)
    for name, tensor in state_dict.items():
        if is_spconv_weight(name, tensor):
            out[name] = tensor.permute(4, 0, 1, 2, 3).contiguous()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.in_path, map_location='cpu')
    key = 'state_dict'
    if key not in checkpoint:
        print(f'{args.in_path}: no "{key}" entry', file=sys.stderr)
        return 1
    try:
        layout = checkpoint_layout(checkpoint[key])
    except ValueError as exc:
        print(f'{args.in_path}: {exc}', file=sys.stderr)
        return 1
    if layout == 'raw':
        print(f'{args.in_path}: already raw; nothing written')
        return 2
    checkpoint[key] = unconvert_state_dict(checkpoint[key])
    torch.save(checkpoint, args.out_path)
    print(f'un-converted {args.in_path} -> {args.out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
