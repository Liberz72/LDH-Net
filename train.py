from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SlidingWindowDataset, StandardScaler, chronological_split
from ldh_net import LDHConfig, LDHNet


def evaluate(model, loader, device):
    model.eval()
    mae = 0.0
    mse = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mae += (pred - y).abs().sum().item()
            mse += (pred - y).square().sum().item()
            count += y.numel()
    return mae / count, mse / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to .npy array with shape [T,N,F]')
    ap.add_argument('--input-len', type=int, default=96)
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--target-dim', type=int, default=1)
    ap.add_argument('--hidden-dim', type=int, default=128)
    ap.add_argument('--events', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--output', default='checkpoints/ldh_net.pt')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw = np.load(args.data).astype(np.float32)
    if raw.ndim != 3:
        raise ValueError(f'Expected [T,N,F], got {raw.shape}')

    train_raw, val_raw, test_raw = chronological_split(raw)
    scaler = StandardScaler().fit(train_raw)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)

    train_ds = SlidingWindowDataset(train, args.input_len, args.horizon, args.target_dim)
    val_ds = SlidingWindowDataset(val, args.input_len, args.horizon, args.target_dim)
    test_ds = SlidingWindowDataset(test, args.input_len, args.horizon, args.target_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    cfg = LDHConfig(
        input_dim=raw.shape[-1],
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
        num_events=args.events,
        horizon=args.horizon,
    )
    model = LDHNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    bad_epochs = 0

    print(f'device={device}, parameters={sum(p.numel() for p in model.parameters()):,}')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        steps = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = model.compute_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            steps += 1

        val_mae, val_mse = evaluate(model, val_loader, device)
        msg = {k: v / max(steps, 1) for k, v in running.items()}
        print(f"epoch={epoch:03d} train={json.dumps(msg)} val_MAE={val_mae:.6f} val_MSE={val_mse:.6f}")

        if val_mse < best:
            best = val_mse
            bad_epochs = 0
            torch.save({
                'model': model.state_dict(),
                'config': model.config_dict(),
                'scaler_mean': scaler.mean,
                'scaler_std': scaler.std,
            }, out)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print('Early stopping.')
                break

    ckpt = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_mae, test_mse = evaluate(model, test_loader, device)
    print(f'test_MAE={test_mae:.6f} test_MSE={test_mse:.6f}')
    print(f'saved={out}')


if __name__ == '__main__':
    main()
