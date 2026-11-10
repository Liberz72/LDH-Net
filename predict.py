from __future__ import annotations
import argparse
import numpy as np
import torch

from ldh_net import LDHConfig, LDHNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--window', required=True, help='.npy with shape [L,N,F]')
    ap.add_argument('--output', default='forecast.npy')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = LDHConfig(**ckpt['config'])
    model = LDHNet(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    x = np.load(args.window).astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f'window must have shape [L,N,F], got {x.shape}')

    mean = ckpt['scaler_mean']
    std = ckpt['scaler_std']
    x_norm = (x - mean) / std

    xt = torch.from_numpy(x_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(xt, return_aux=True)
        pred = out['pred'][0].cpu().numpy()          # [H,N,Fy], normalized
        A = out['geom']['A'][0].cpu().numpy()        # [N,K]
        support = out['geom']['support'][0].cpu().numpy()  # [K,N]

    pred = pred * std[..., :cfg.target_dim] + mean[..., :cfg.target_dim]
    np.save(args.output, pred)
    np.save(args.output.replace('.npy', '_association.npy'), A)
    np.save(args.output.replace('.npy', '_support.npy'), support)

    print('forecast:', pred.shape, '->', args.output)
    print('association A:', A.shape)
    print('hard support:', support.shape)


if __name__ == '__main__':
    main()
