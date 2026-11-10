"""Small synthetic smoke demo; runs on CPU."""
import numpy as np
import torch

from ldh_net import LDHConfig, LDHNet


torch.manual_seed(7)
B, L, N, F, H = 2, 24, 6, 3, 8
x = torch.randn(B, L, N, F)
y = torch.randn(B, H, N, 1)

cfg = LDHConfig(
    input_dim=F,
    target_dim=1,
    hidden_dim=32,
    attn_dim=16,
    compat_dim=16,
    lorentz_dim=8,
    num_events=4,
    horizon=H,
    message_layers=2,
)
model = LDHNet(cfg)

loss, metrics = model.compute_loss(x, y)
loss.backward()

with torch.no_grad():
    out = model(x, return_aux=True)

print('prediction shape:', tuple(out['pred'].shape))   # [B,H,N,Fy]
print('association shape:', tuple(out['geom']['A'].shape))  # [B,N,K]
print('support shape:', tuple(out['geom']['support'].shape)) # [B,K,N]
print('loss metrics:', metrics)
print('demo OK')
