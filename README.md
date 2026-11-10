# LDH-Net minimal PyTorch implementation

This project implements the workflow:

`Historical Window -> Temporal Encoder -> Structured Event Induction -> Lorentzian Directed Hypergraph -> Set-Coupled Propagation -> Forecast Decoder`

The training-only branch performs event intervention and counterfactual alignment with shared prediction parameters.

## 1. Input format

Use a NumPy `.npy` file with shape:

```text
[T, N, F]
```

- `T`: total time steps
- `N`: entities / variables / nodes
- `F`: observed features per entity

The code automatically builds sliding windows:

```text
X: [L, N, F]
Y: [H, N, Fy]
```

By default the first `Fy` features are forecast targets.

## 2. Quick smoke test

```bash
python demo.py
```

Expected key shapes:

```text
prediction: [B, H, N, Fy]
association A: [B, N, K]
hard support: [B, K, N]
```

## 3. Train

```bash
python train.py \
  --data your_data.npy \
  --input-len 96 \
  --horizon 24 \
  --target-dim 1 \
  --events 8 \
  --epochs 30 \
  --batch-size 16 \
  --output checkpoints/ldh_net.pt
```

## 4. Predict one historical window

Prepare `window.npy` with shape `[L,N,F]`, then:

```bash
python predict.py \
  --checkpoint checkpoints/ldh_net.pt \
  --window window.npy \
  --output forecast.npy
```

Outputs:

- `forecast.npy`: `[H,N,Fy]`
- `forecast_association.npy`: soft event-entity association matrix `[N,K]`
- `forecast_support.npy`: thresholded directed hyperedge support `[K,N]`

## 5. Module mapping to the workflow

### Temporal Encoder
`TemporalEncoder` maps `[B,L,N,F] -> R_t [B,N,d]`.

A GRU is used only as a default temporal backbone because the workflow does not prescribe a specific encoder. It can be replaced with PatchTST, Transformer, Mamba, TCN, etc., as long as it returns one `d`-dimensional state per entity.

### Structured Event Induction
`StructuredEventInduction` uses `K` learnable queries and cross-attention over entity states to produce event slots, then decomposes each event into:

- `h`: content
- `a`: intensity
- `s`: direction
- `u`: uncertainty

### Lorentzian Directed Hypergraph
`LorentzianHypergraph` builds event/entity coordinates, evaluates the future-light-cone interval, computes differentiable reachability, and returns:

- `A [B,N,K]`: soft association matrix
- `support [B,K,N]`: explicit directed hyperedge support

### Set-Coupled Propagation
`SetCoupledPropagation` uses a leave-one-out receiver context. For the message `event k -> entity i`, the context contains the other receivers of the same event, so a hyperedge is not reduced to independent pairwise edges.

### Forecast Decoder
The decoder maps all final entity representations to `[B,H,N,Fy]` and uses a residual prediction around the last observed target value.

### Training-only intervention branch
`compute_loss()` creates perturbed events and passes them through the same geometry, propagation and decoder parameters. It combines:

- prediction loss
- event diversity loss
- light-cone consistency loss
- event-response alignment
- hyperedge structural separation
- forecast-response separation

## 6. Important implementation note

This is a faithful executable reconstruction of the supplied workflow, not an assertion that every engineering choice is uniquely determined by the paper. In particular, the temporal backbone and the internal MLP widths are implementation choices. The event induction, Lorentzian reachability, event-to-set association, leave-one-out set coupling, residual forecast output, and training-only counterfactual branch follow the supplied workflow directly.
