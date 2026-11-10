"""LDH-Net: Lorentzian Directed Hypergraph Network for multivariate forecasting.

This implementation follows the user's workflow:
Historical Window -> Temporal Encoder -> Structured Event Induction ->
Lorentzian Directed Hypergraph -> Set-Coupled Propagation -> Forecast Decoder.

The counterfactual / intervention-inspired alignment branch is enabled only in training.

Tensor convention
-----------------
X: [B, L, N, F]
Y: [B, H, N, Fy]
R: [B, N, d]
A: [B, N, K]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LDHConfig:
    input_dim: int
    target_dim: int = 1
    hidden_dim: int = 128
    attn_dim: int = 64
    compat_dim: int = 64
    lorentz_dim: int = 32
    num_events: int = 8
    horizon: int = 96
    message_layers: int = 2

    # Lorentzian geometry
    rx: float = 1.0
    delta_min: float = 0.10
    delta_max: float = 1.00
    kappa: float = 10.0
    lambda_m: float = 1.0
    lambda_a: float = 1.0
    lambda_u: float = 1.0
    association_bias: float = 0.0
    hard_threshold: float = 0.5
    eps: float = 1e-6

    # Counterfactual branch
    cf_omega: float = 0.20
    align_dim: int = 64
    align_margin: float = 0.20
    rho_h: float = 0.10
    rho_x: float = 0.10

    # Loss weights
    lambda_evt: float = 0.01
    lambda_lc: float = 0.01
    lambda_cf: float = 0.10
    lambda_hyp: float = 1.0
    lambda_resp: float = 1.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalEncoder(nn.Module):
    """Entity-wise temporal encoder.

    The workflow leaves the temporal backbone open. A GRU is used here as a clean,
    replaceable default. Each entity is encoded independently over the L-step window,
    yielding one state r_i per entity.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, F] -> [B*N, L, F]
        b, l, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, l, f)
        _, h = self.gru(seq)
        r = h[-1].reshape(b, n, -1)
        return self.norm(r)


class StructuredEventInduction(nn.Module):
    """K learnable event queries -> event slots -> structured event attributes."""

    def __init__(self, cfg: LDHConfig):
        super().__init__()
        d, da, k = cfg.hidden_dim, cfg.attn_dim, cfg.num_events
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.wq = nn.Linear(d, da, bias=False)
        self.wk = nn.Linear(d, da, bias=False)
        self.wv = nn.Linear(d, d, bias=False)

        self.to_h = nn.Linear(d, d)
        self.to_a = nn.Linear(d, 1)
        self.to_s = nn.Linear(d, d)
        self.to_u = nn.Linear(d, 1)

    def forward(self, r: torch.Tensor) -> Dict[str, torch.Tensor]:
        # r: [B, N, d]
        q = self.wq(self.queries)                         # [K, da]
        keys = self.wk(r)                                # [B, N, da]
        values = self.wv(r)                              # [B, N, d]
        scores = torch.einsum("kd,bnd->bkn", q, keys) / math.sqrt(q.size(-1))
        attn = scores.softmax(dim=-1)                    # softmax over entities N
        e = torch.einsum("bkn,bnd->bkd", attn, values) # [B, K, d]

        h = self.to_h(e)
        a = F.softplus(self.to_a(e))
        s = torch.tanh(self.to_s(e))
        u = F.softplus(self.to_u(e))
        return {"slot": e, "h": h, "a": a, "s": s, "u": u, "slot_attn": attn}


class LorentzianHypergraph(nn.Module):
    """Map events/entities to scale-controlled Lorentzian coordinates and build A."""

    def __init__(self, cfg: LDHConfig):
        super().__init__()
        d, dl, dc = cfg.hidden_dim, cfg.lorentz_dim, cfg.compat_dim
        self.cfg = cfg
        self.ph = nn.Linear(d, dl, bias=False)
        self.ps = nn.Linear(d, dl, bias=False)
        self.pv = nn.Linear(d, dl, bias=False)
        self.w_tau = nn.Linear(d, 1)
        self.ue = nn.Linear(d, dc, bias=False)
        self.uv = nn.Linear(d, dc, bias=False)

    def forward(
        self,
        r: torch.Tensor,
        events: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        h, a, s, u = events["h"], events["a"], events["s"], events["u"]

        x_e = cfg.rx * torch.tanh(self.ph(h) + self.ps(s))   # [B,K,dl]
        x_v = cfg.rx * torch.tanh(self.pv(r))                # [B,N,dl]

        # Prediction-anchor time is used as origin; only relative time matters here.
        tau_e = torch.zeros(x_e.shape[:2], device=x_e.device, dtype=x_e.dtype)  # [B,K]
        delta = cfg.delta_min + (cfg.delta_max - cfg.delta_min) * torch.sigmoid(
            self.w_tau(r).squeeze(-1)
        )                                                     # [B,N]
        tau_v = delta                                         # [B,N]

        dt = tau_v[:, None, :] - tau_e[:, :, None]            # [B,K,N] > 0
        dx = x_v[:, None, :, :] - x_e[:, :, None, :]          # [B,K,N,dl]
        m = dt.square() - dx.square().sum(dim=-1)              # [B,K,N]

        sm = cfg.delta_max ** 2 + 4.0 * (cfg.rx ** 2) * cfg.lorentz_dim
        g_lc = torch.sigmoid(cfg.kappa * m / sm)

        he = self.ue(h)                                       # [B,K,dc]
        rv = self.uv(r)                                       # [B,N,dc]
        compat = torch.einsum("bkd,bnd->bkn", he, rv) / math.sqrt(he.size(-1))

        logit = (
            compat
            + cfg.lambda_m * m / (dt.square() + cfg.eps)
            + cfg.lambda_a * a.squeeze(-1)[:, :, None]
            - cfg.lambda_u * u.squeeze(-1)[:, :, None]
            + cfg.association_bias
        )
        alpha_kn = g_lc * torch.sigmoid(logit)                 # [B,K,N]
        association = alpha_kn.transpose(1, 2)                 # [B,N,K]
        support = (m >= 0) & (alpha_kn > cfg.hard_threshold)   # [B,K,N]

        return {
            "x_e": x_e,
            "x_v": x_v,
            "tau_e": tau_e,
            "tau_v": tau_v,
            "dt": dt,
            "dx": dx,
            "m": m,
            "g_lc": g_lc,
            "alpha_kn": alpha_kn,
            "A": association,
            "support": support,
        }


class SetCoupledPropagation(nn.Module):
    """Set-coupled hypergraph propagation with leave-one-out receiver context."""

    def __init__(self, cfg: LDHConfig):
        super().__init__()
        self.cfg = cfg
        d, dl = cfg.hidden_dim, cfg.lorentz_dim
        self.readout = nn.Linear(dl, d)

        # chi=[dt, dx], h, s, receiver r_i, leave-one-out context c_{-i}
        msg_in = 1 + dl + 4 * d
        self.message_mlps = nn.ModuleList(
            [MLP(msg_in, dl, hidden=max(2 * d, dl)) for _ in range(cfg.message_layers)]
        )
        self.update_gates = nn.ModuleList(
            [nn.Linear(d + dl, 1) for _ in range(cfg.message_layers)]
        )

    def forward(
        self,
        geom: Dict[str, torch.Tensor],
        events: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        x = geom["x_v"]                                      # [B,N,dl]
        x_e = geom["x_e"]                                    # [B,K,dl]
        dt = geom["dt"]                                      # [B,K,N]
        alpha = geom["alpha_kn"]                             # [B,K,N]
        h, s = events["h"], events["s"]                     # [B,K,d]

        for msg_mlp, gate_layer in zip(self.message_mlps, self.update_gates):
            r = self.readout(x)                               # [B,N,d]
            dx = x[:, None, :, :] - x_e[:, :, None, :]       # [B,K,N,dl]

            # Leave-one-out context for every event k and receiver i.
            weighted_total = torch.einsum("bkn,bnd->bkd", alpha, r)          # [B,K,d]
            denom_total = alpha.sum(dim=-1)                                  # [B,K]
            numer_loo = weighted_total[:, :, None, :] - alpha[..., None] * r[:, None, :, :]
            denom_loo = denom_total[:, :, None] - alpha + cfg.eps
            c_loo = numer_loo / denom_loo[..., None]                         # [B,K,N,d]

            b, k, n, _ = dx.shape
            h_exp = h[:, :, None, :].expand(b, k, n, -1)
            s_exp = s[:, :, None, :].expand(b, k, n, -1)
            r_exp = r[:, None, :, :].expand(b, k, n, -1)
            chi = torch.cat([dt[..., None], dx], dim=-1)
            msg_input = torch.cat([chi, h_exp, s_exp, r_exp, c_loo], dim=-1)
            msg = msg_mlp(msg_input)                          # [B,K,N,dl]

            # Receiver-side normalized event aggregation.
            denom_entity = alpha.sum(dim=1).clamp_min(cfg.eps)               # [B,N]
            weights = alpha / denom_entity[:, None, :]                        # [B,K,N]
            mbar = (weights[..., None] * msg).sum(dim=1)                      # [B,N,dl]

            gate = torch.sigmoid(gate_layer(torch.cat([r, mbar], dim=-1)))    # [B,N,1]
            eta = gate * mbar

            # Bounded fixed-time spatial update.
            x_scaled = (x / cfg.rx).clamp(-0.999999, 0.999999)
            x = cfg.rx * torch.tanh(torch.atanh(x_scaled) + eta / cfg.rx)

        return self.readout(x), x


class ForecastDecoder(nn.Module):
    def __init__(self, cfg: LDHConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.hidden_dim, cfg.horizon * cfg.target_dim)

    def forward(self, r_final: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        # r_final: [B,N,d] -> [B,H,N,Fy]
        b, n, _ = r_final.shape
        out = self.proj(r_final).view(b, n, self.cfg.horizon, self.cfg.target_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        # Residual decoder: last observed target value + decoded residual.
        baseline = x_hist[:, -1, :, : self.cfg.target_dim][:, None, :, :]
        return baseline + out


class LDHNet(nn.Module):
    def __init__(self, cfg: LDHConfig):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg.input_dim, cfg.hidden_dim)
        self.event_induction = StructuredEventInduction(cfg)
        self.hypergraph = LorentzianHypergraph(cfg)
        self.propagation = SetCoupledPropagation(cfg)
        self.decoder = ForecastDecoder(cfg)

        # Alignment encoders used only by training loss.
        zeta_dim = 2 * cfg.hidden_dim + 2
        self.event_align_encoder = MLP(zeta_dim, cfg.align_dim, hidden=cfg.hidden_dim)
        self.response_encoder = MLP(cfg.horizon * cfg.target_dim, cfg.align_dim, hidden=cfg.hidden_dim)

    @staticmethod
    def _zeta(events: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([events["h"], events["a"], events["s"], events["u"]], dim=-1)

    def _forward_from_events(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        events: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        geom = self.hypergraph(r, events)
        r_final, x_final = self.propagation(geom, events)
        pred = self.decoder(r_final, x)
        return {"pred": pred, "geom": geom, "r_final": r_final, "x_final": x_final}

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        r = self.temporal(x)
        events = self.event_induction(r)
        out = self._forward_from_events(x, r, events)
        if not return_aux:
            return out["pred"]
        out.update({"r": r, "events": events, "zeta": self._zeta(events)})
        return out

    def _intervene(
        self, events: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Randomly perturb one of {h,a,s,u} for each (batch,event)."""
        cfg = self.cfg
        h, a, s, u = [events[k] for k in ("h", "a", "s", "u")]
        b, k, d = h.shape
        choice = torch.randint(0, 4, (b, k), device=h.device)

        h_cf, a_cf, s_cf, u_cf = h.clone(), a.clone(), s.clone(), u.clone()

        # content intervention
        xi_h = torch.randn_like(h)
        xi_h = xi_h / (xi_h.norm(dim=-1, keepdim=True) + cfg.eps)
        h_prop = h + cfg.cf_omega * (h.norm(dim=-1, keepdim=True) + cfg.eps) * xi_h
        mask_h = (choice == 0)[..., None]
        h_cf = torch.where(mask_h, h_prop, h_cf)

        # intensity intervention
        xi_a = torch.randn_like(a)
        a_prop = a * torch.exp(cfg.cf_omega * xi_a)
        mask_a = (choice == 1)[..., None]
        a_cf = torch.where(mask_a, a_prop, a_cf)

        # direction intervention
        mask_s = (choice == 2)[..., None]
        s_cf = torch.where(mask_s, -s, s_cf)

        # uncertainty intervention
        xi_u = torch.randn_like(u)
        u_prop = u + cfg.cf_omega * xi_u.abs()
        mask_u = (choice == 3)[..., None]
        u_cf = torch.where(mask_u, u_prop, u_cf)

        # Dimensionless intervention magnitude nu_{t,k}.
        dh = (h_cf - h) / (h.norm(dim=-1, keepdim=True) + cfg.eps)
        da = (a_cf - a) / (a + cfg.eps)
        ds = (s_cf - s) / (s.norm(dim=-1, keepdim=True) + cfg.eps)
        du = (u_cf - u) / (u + cfg.eps)
        nu = torch.cat([dh, da, ds, du], dim=-1).norm(dim=-1)  # [B,K]

        cf = {**events, "h": h_cf, "a": a_cf, "s": s_cf, "u": u_cf}
        return cf, nu

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full training objective including training-only counterfactual regularization."""
        cfg = self.cfg
        factual = self.forward(x, return_aux=True)
        pred = factual["pred"]
        events = factual["events"]
        geom = factual["geom"]
        r = factual["r"]

        # L_pred
        l_pred = F.l1_loss(pred, y)

        # L_evt: average absolute cosine similarity between distinct event tensors.
        zeta = factual["zeta"]                             # [B,K,Dz]
        z_norm = F.normalize(zeta, dim=-1, eps=cfg.eps)
        cos = torch.einsum("bkd,bjd->bkj", z_norm, z_norm).abs()
        eye = torch.eye(cfg.num_events, device=cos.device, dtype=torch.bool)[None]
        l_evt = cos.masked_select(~eye.expand_as(cos)).mean() if cfg.num_events > 1 else cos.new_zeros(())

        # L_lc
        alpha = geom["alpha_kn"]                           # [B,K,N]
        m = geom["m"]
        l_lc = (alpha * F.relu(-m)).mean()

        # Training-only counterfactual path; same hypergraph/propagation/decoder parameters.
        cf_events, nu = self._intervene(events)
        cf_out = self._forward_from_events(x, r, cf_events)
        pred_cf = cf_out["pred"]
        a_cf = cf_out["geom"]["A"]                        # [B,N,K]
        a_fact = geom["A"]                                 # [B,N,K]

        # Response sequence relative to the last observed target value.
        baseline = x[:, -1, :, : cfg.target_dim][:, None, :, :]
        delta_y = y - baseline                              # [B,H,N,Fy]
        b, hzn, n, fy = delta_y.shape
        resp_i = delta_y.permute(0, 2, 1, 3).reshape(b, n, hzn * fy)
        resp_i = self.response_encoder(resp_i)              # [B,N,Dalign]

        # P[i,k] = A[i,k] / sum_j A[j,k], stopped for factual response construction.
        p = a_fact / (a_fact.sum(dim=1, keepdim=True) + cfg.eps)
        y_event = torch.einsum("bnk,bnd->bkd", p.detach(), resp_i)

        e_fact = self.event_align_encoder(self._zeta(events))
        e_cf = self.event_align_encoder(self._zeta(cf_events))
        sim_fact = F.cosine_similarity(e_fact, y_event, dim=-1)
        sim_cf = F.cosine_similarity(e_cf, y_event, dim=-1)
        l_align = F.relu(cfg.align_margin - sim_fact + sim_cf).mean()

        # Hyperedge-level separation.
        w = a_fact.mean(dim=1).detach()                      # [B,K]
        d_h = (a_fact - a_cf).abs().mean(dim=1)             # [B,K]
        l_hyp = (w * F.relu(cfg.rho_h * nu - d_h)).mean()

        # Prediction-level separation.
        nu_bar = nu.mean(dim=1)                              # [B]
        d_x = (pred - pred_cf).abs().mean(dim=(1, 2, 3))    # [B]
        l_resp = F.relu(cfg.rho_x * nu_bar - d_x).mean()

        l_cf = l_align + cfg.lambda_hyp * l_hyp + cfg.lambda_resp * l_resp
        total = l_pred + cfg.lambda_evt * l_evt + cfg.lambda_lc * l_lc + cfg.lambda_cf * l_cf

        metrics = {
            "loss": float(total.detach()),
            "pred": float(l_pred.detach()),
            "evt": float(l_evt.detach()),
            "lc": float(l_lc.detach()),
            "cf": float(l_cf.detach()),
            "align": float(l_align.detach()),
            "hyp": float(l_hyp.detach()),
            "resp": float(l_resp.detach()),
            "mean_activation": float(a_fact.detach().mean()),
            "support_density": float(geom["support"].float().mean().detach()),
        }
        return total, metrics

    def config_dict(self) -> Dict:
        return asdict(self.cfg)
