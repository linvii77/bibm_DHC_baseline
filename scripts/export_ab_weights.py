#!/usr/bin/env python3
"""Export model A/B state_dict from DHC best_model.pth for inference_Synapse_CPS.py."""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='path to best_model.pth')
    parser.add_argument('--out_dir', type=str, default='./tmp_ckpts')
    parser.add_argument('--prefix', type=str, default='dhc_best')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w = torch.load(args.ckpt, map_location='cpu')
    if 'A' not in w or 'B' not in w:
        raise KeyError(f'Expected keys A/B in checkpoint, got {list(w.keys())}')

    path_a = os.path.join(args.out_dir, f'{args.prefix}_A.pth')
    path_b = os.path.join(args.out_dir, f'{args.prefix}_B.pth')
    torch.save(w['A'], path_a)
    torch.save(w['B'], path_b)
    print(f'Saved: {path_a}')
    print(f'Saved: {path_b}')


if __name__ == '__main__':
    main()
