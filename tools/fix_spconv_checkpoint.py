"""Convert spconv weights of a training checkpoint to the layout tools/test.py loads.

Raw layout (saved by training):  (out_channels, kD, kH, kW, in_channels)
Converted layout:                (kD, kH, kW, in_channels, out_channels)

The script refuses to convert a checkpoint that is already converted
(exit code 2, message "already converted"), because a second permutation
produces weights that no longer load.
"""
import argparse
import sys

import torch

SPCONV_PREFIXES = ('unet', 'input_conv')


def is_spconv_weight(name, tensor):
    return name.startswith(SPCONV_PREFIXES) and name.endswith('weight') and tensor.dim() == 5


def weight_layout(shape):
    """'raw' when the three kernel dims sit at 1..3, 'converted' when at 0..2.

    Kernel sizes are 1, 2 or 3 while UNet channel counts are >= 32, so the
    rule is unambiguous for every `unet.*` weight. `input_conv.0.weight` has
    3 input channels; in the converted layout its shape (3, 3, 3, 3, 32)
    cannot be told apart by shape alone and is reported as 'ambiguous'.
    """
    s = tuple(int(x) for x in shape)
    raw = s[1] == s[2] == s[3] and s[0] != s[1]
    converted = s[0] == s[1] == s[2] and s[3] != s[2]
    if raw and not converted:
        return 'raw'
    if converted and not raw:
        return 'converted'
    return 'ambiguous'


def checkpoint_layout(state_dict):
    layouts = {weight_layout(t.shape) for n, t in state_dict.items() if is_spconv_weight(n, t)}
    layouts.discard('ambiguous')
    if layouts == {'raw'}:
        return 'raw'
    if layouts == {'converted'}:
        return 'converted'
    if not layouts:
        raise ValueError('no spconv weights found under prefixes ' + ', '.join(SPCONV_PREFIXES))
    raise ValueError(f'mixed weight layouts in checkpoint: {sorted(layouts)}')


def convert_state_dict(state_dict):
    out = dict(state_dict)
    for name, tensor in state_dict.items():
        if is_spconv_weight(name, tensor):
            out[name] = tensor.permute(1, 2, 3, 4, 0).contiguous()
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
    if layout == 'converted':
        print(f'{args.in_path}: already converted; nothing written')
        return 2
    checkpoint[key] = convert_state_dict(checkpoint[key])
    torch.save(checkpoint, args.out_path)
    print(f'converted {args.in_path} -> {args.out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
