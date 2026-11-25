"""The bot's network: attention over entities, card head, tile head, value head.

Input: the raw crforge observation (int32, 224 header ints then up to 200 entity rows of
34 ints) and the mask of the 2307 actions. Output, in the format read by the crforge app
(desktop/model/ModelPolicy.java):
- first_logp (B, 7): do nothing, play slot 0 to 3, ability 0 or 1;
- tile_logp (B, 4, 576): for each slot, the tile on the 18 x 32 grid;
- value (B,).
Choices closed by the mask have a log-probability of about -1e9.

You can pass fewer than 200 entity rows (the last ones are empty): the network reads the
number of rows present in the input.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADER = 224
ENT = 34
SLOTS = 4
TILES = 576
FIRST = 1 + SLOTS + 2  # wait, 4 slots, 2 abilities
FIRST_ABILITY = 1 + SLOTS * TILES  # 2305
CLOSED = -1e9


@dataclass
class PolicyConfig:
    vocab_size: int = 1466
    buff_size: int = 256
    d_model: int = 128
    heads: int = 4
    layers: int = 3
    trunk: int = 256
    card_dim: int = 32


class Layout:
    """Field indices, read from the crforge metadata.json (kept with each run)."""

    def __init__(self, metadata: dict):
        s = metadata["schema"]
        hf = s["header_fields"]
        self.header: dict[str, list[int]] = {}
        for i, name in enumerate(hf):
            self.header.setdefault(name, []).append(i)
        self.entity = {name: i for i, name in enumerate(s["entity_fields"])}
        voc = metadata.get("vocabulary") or {}
        self.vocab_size = len(voc.get("names", [])) or 1466
        self.buff_size = len(voc.get("buff_names", [])) or 256
        assert int(s["header_ints"]) == HEADER and int(s["entity_ints"]) == ENT

    @classmethod
    def load(cls, path: str | Path) -> Layout:
        return cls(json.loads(Path(path).read_text()))

    def h(self, name: str) -> list[int]:
        return self.header[name]


def _slog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class Policy(nn.Module):
    def __init__(self, layout: Layout, cfg: PolicyConfig | None = None):
        super().__init__()
        cfg = cfg or PolicyConfig(vocab_size=layout.vocab_size, buff_size=layout.buff_size)
        self.cfg = cfg
        L = layout
        d = cfg.d_model
        # fixed indices, kept as buffers for the export
        def reg(name, idx):
            self.register_buffer(name, torch.tensor(idx, dtype=torch.long), persistent=False)

        reg("i_hand", L.h("hand_card"))
        reg("i_hand_cost", L.h("hand_cost"))
        reg("i_hand_form", L.h("hand_form"))
        reg("i_hand_pending", L.h("hand_pending"))
        reg("i_cards", L.h("next_card") + L.h("queue") + L.h("deck") + L.h("enemy_seen") + L.h("enemy_play_card"))
        reg("i_count", L.h("entity_count"))
        e = L.entity
        self.e_row, self.e_kind, self.e_side, self.e_state = e["row"], e["kind"], e["side"], e["state"]
        self.e_flags, self.e_hp, self.e_max = e["flags"], e["hp"], e["max_hp"]
        self.e_buffs = [e["buff_0"], e["buff_1"], e["buff_2"]]
        num = ["x", "y", "shield", "level", "dx", "dy", "target_x", "target_y", "deploy_ms",
               "attack_ms", "load_ms", "speed_percent", "hit_speed_percent", "buff_count",
               "buff_ms_0", "buff_ms_1", "buff_ms_2", "aim_x", "aim_y", "time_left_ms", "radius",
               "ability_ticks", "target"]
        reg("e_num", [e[n] for n in num])
        scale = [18000, 32000, 1000, 16, 1000, 1000, 18000, 32000, 1000, 1000, 1000, 100, 100, 4,
                 10000, 10000, 10000, 18000, 32000, 10000, 5000, 100, 1]
        self.register_buffer("e_scale", torch.tensor(scale, dtype=torch.float32), persistent=False)
        self.register_buffer("bits", 2 ** torch.arange(12), persistent=False)
        self.register_buffer("wait_bias", torch.zeros(()))  # prior correction, set after imitation

        V = cfg.vocab_size + 1  # +1: unknown card (-1)
        self.card = nn.Embedding(V, cfg.card_dim)
        self.kind = nn.Embedding(6, 8)
        self.side = nn.Embedding(4, 8)
        self.state = nn.Embedding(20, 8)
        self.buff = nn.Embedding(cfg.buff_size + 1, 8)
        ent_in = cfg.card_dim + 8 * 3 + 8 + len(num) + 12 + 4
        self.ent_proj = nn.Sequential(nn.Linear(ent_in, d), nn.GELU(), nn.Linear(d, d))
        # header: its 224 ints (signed log), the hand and the other cards seen
        self.head_proj = nn.Sequential(nn.Linear(HEADER + cfg.card_dim * 2, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        T = cfg.trunk
        self.trunk = nn.Sequential(nn.Linear(3 * d, T), nn.GELU(), nn.Linear(T, T), nn.GELU())
        self.wait_head = nn.Linear(T, 1)
        self.ability_head = nn.Linear(T, 2)
        slot_in = T + cfg.card_dim + 1 + 3 + 1
        self.slot_head = nn.Sequential(nn.Linear(slot_in, T), nn.GELU(), nn.Linear(T, 1))
        self.tile_query = nn.Sequential(nn.Linear(slot_in, T), nn.GELU())
        self.tile_out = nn.Linear(T, TILES)
        self.value_head = nn.Sequential(nn.Linear(T, T), nn.GELU(), nn.Linear(T, 1))
        # privileged critic (training only): the opponent's header (hand, elixir, queue) is added
        # to the input of the win estimate. The last layer starts at zero, so at first nothing
        # changes. Without this input (in game, in the ONNX export), the term is zero.
        self.priv_proj = nn.Sequential(nn.Linear(HEADER + cfg.card_dim, T), nn.GELU(), nn.Linear(T, T))
        nn.init.zeros_(self.priv_proj[2].weight)
        nn.init.zeros_(self.priv_proj[2].bias)

    # -- encoding --------------------------------------------------------------------------------

    def _cards(self, ids: torch.Tensor) -> torch.Tensor:
        ids = torch.where(ids < 0, torch.full_like(ids, self.cfg.vocab_size), ids)
        return self.card(ids.clamp(0, self.cfg.vocab_size))

    def encode(self, obs: torch.Tensor):
        obs = obs.long()
        B = obs.shape[0]
        head = obs[:, :HEADER]
        ents = obs[:, HEADER:].reshape(B, -1, ENT)
        E = ents.shape[1]
        count = head.index_select(1, self.i_count).squeeze(1).clamp(max=E)
        pad = torch.arange(E, device=obs.device).unsqueeze(0) >= count.unsqueeze(1)  # True = empty

        f = ents.float()
        hp, mx = f[..., self.e_hp], f[..., self.e_max]
        known = (hp >= 0).float()
        hp_feats = torch.stack([known, torch.log1p(hp.clamp(min=0)) / 10, torch.log1p(mx.clamp(min=0)) / 10,
                                known * hp.clamp(min=0) / mx.clamp(min=1)], -1)
        # flag bits, without bitwise operators (for the ONNX export)
        flags = (torch.div(ents[..., self.e_flags].clamp(min=0).unsqueeze(-1), self.bits, rounding_mode="floor") % 2).float()
        num = f.index_select(2, self.e_num) / self.e_scale
        buffs = sum(self.buff((ents[..., b] + 1).clamp(0, self.cfg.buff_size)) for b in self.e_buffs)
        x = torch.cat([
            self._cards(ents[..., self.e_row]),
            self.kind(ents[..., self.e_kind].clamp(0, 5)),
            self.side(ents[..., self.e_side].clamp(0, 3)),
            self.state((ents[..., self.e_state] + 1).clamp(0, 19)),
            buffs, num, flags, hp_feats,
        ], -1)
        tok = self.ent_proj(x)

        hand = head.index_select(1, self.i_hand)
        others = self._cards(head.index_select(1, self.i_cards)).mean(1)
        hand_emb = self._cards(hand)
        h_in = torch.cat([_slog(head.float()) / 5, hand_emb.mean(1), others], -1)
        htok = self.head_proj(h_in).unsqueeze(1)

        seq = torch.cat([htok, tok], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs.device), pad], 1)
        out = self.encoder(seq, src_key_padding_mask=mask)
        keep = (~pad).float().unsqueeze(-1)
        ent_mean = (out[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1)
        z = self.trunk(torch.cat([out[:, 0], ent_mean, htok.squeeze(1)], -1))

        cost = head.index_select(1, self.i_hand_cost).float().unsqueeze(-1) / 10
        form = F.one_hot(head.index_select(1, self.i_hand_form).clamp(0, 2), 3).float()
        pend = head.index_select(1, self.i_hand_pending).float().unsqueeze(-1)
        slot_in = torch.cat([z.unsqueeze(1).expand(-1, SLOTS, -1), hand_emb, cost, form, pend], -1)
        return z, slot_in

    # -- outputs ---------------------------------------------------------------------------------

    def privileged(self, opp_head: torch.Tensor) -> torch.Tensor:
        """What the critic sees on top: the opponent's header (raw ints)."""
        h = opp_head.long()
        hand = self._cards(h.index_select(1, self.i_hand)).mean(1)
        return self.priv_proj(torch.cat([_slog(h.float()) / 5, hand], -1))

    def forward(self, obs: torch.Tensor, masks: torch.Tensor, priv: torch.Tensor | None = None):
        """first_logp (B, 7), tile_logp (B, 4, 576), value (B,). priv: the opponent's header (privileged
        critic, training only)."""
        z, slot_in = self.encode(obs)
        m = masks.bool()
        tile_mask = m[:, 1:FIRST_ABILITY].reshape(-1, SLOTS, TILES)
        slot_open = tile_mask.any(-1)
        first_open = torch.cat([torch.ones_like(slot_open[:, :1]), slot_open, m[:, FIRST_ABILITY:FIRST_ABILITY + 2]], 1)
        logits = torch.cat([
            self.wait_head(z) + self.wait_bias,
            self.slot_head(slot_in).squeeze(-1),
            self.ability_head(z),
        ], 1)
        logits = logits.float().masked_fill(~first_open, CLOSED)
        first_logp = torch.log_softmax(logits, -1)
        tiles = self.tile_out(self.tile_query(slot_in)).float()
        tile_logp = torch.log_softmax(tiles.masked_fill(~tile_mask, CLOSED), -1)
        zv = z if priv is None else z + self.privileged(priv)
        value = self.value_head(zv).squeeze(-1).float()
        return first_logp, tile_logp, value


# -- actions ---------------------------------------------------------------------------------------

def split_action(action: torch.Tensor):
    """Flat crforge action -> (choice 0..6, slot, tile)."""
    a = action.long()
    is_play = (a >= 1) & (a < FIRST_ABILITY)
    slot = torch.where(is_play, (a - 1) // TILES, torch.zeros_like(a))
    tile = torch.where(is_play, (a - 1) % TILES, torch.zeros_like(a))
    first = torch.where(a == 0, torch.zeros_like(a), torch.where(is_play, 1 + slot, 5 + (a - FIRST_ABILITY)))
    return first, slot, tile, is_play


def log_prob(first_logp, tile_logp, action):
    first, slot, tile, is_play = split_action(action)
    lp = first_logp.gather(1, first.unsqueeze(1)).squeeze(1)
    tl = tile_logp[torch.arange(len(action), device=action.device), slot, tile]
    return lp + torch.where(is_play, tl, torch.zeros_like(tl))


def entropy(first_logp, tile_logp):
    p1 = first_logp.exp()
    h1 = -(p1 * first_logp.clamp(min=-50)).sum(-1)
    pt = tile_logp.exp()
    ht = -(pt * tile_logp.clamp(min=-50)).sum(-1)  # (B, 4)
    return h1 + (p1[:, 1:1 + SLOTS] * ht).sum(-1)


def kl(first_logp, tile_logp, ref_first, ref_tile):
    """KL(policy || reference), on the choice then on the tile."""
    p1 = first_logp.exp()
    k1 = (p1 * (first_logp - ref_first).clamp(-50, 50)).sum(-1)
    pt = tile_logp.exp()
    kt = (pt * (tile_logp - ref_tile).clamp(-50, 50)).sum(-1)
    return k1 + (p1[:, 1:1 + SLOTS] * kt).sum(-1)


@torch.no_grad()
def sample(first_logp, tile_logp, greedy: bool = False):
    if greedy:
        first = first_logp.argmax(-1)
    else:
        first = torch.distributions.Categorical(logits=first_logp).sample()
    slot = (first - 1).clamp(0, SLOTS - 1)
    tl = tile_logp[torch.arange(len(first), device=first.device), slot]
    tile = tl.argmax(-1) if greedy else torch.distributions.Categorical(logits=tl).sample()
    play = 1 + slot * TILES + tile
    ability = FIRST_ABILITY + (first - 5)
    return torch.where(first == 0, torch.zeros_like(first), torch.where(first <= SLOTS, play, ability))


# -- saving ---------------------------------------------------------------------------------------

def save(path, model: Policy, metadata: dict, extra: dict | None = None):
    torch.save({"config": asdict(model.cfg), "state": model.state_dict(), "metadata": metadata,
                **(extra or {})}, path)


def load(path, device="cpu") -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Policy(Layout(ck["metadata"]), PolicyConfig(**ck["config"])).to(device)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    bad = [k for k in missing if not k.startswith("priv_proj")] + list(unexpected)
    if bad:
        raise RuntimeError(f"incompatible checkpoint: {bad[:5]}")
    return model, ck


def trim(obs, count_index: int, cap: int = 200):
    """Cuts the empty entity rows at the end of a batch: keeps the largest count present."""
    n = int(obs[..., count_index].max()) if obs.size else 1
    n = max(1, min(cap, n))
    return obs[..., : HEADER + n * ENT]


def wait_bias_for(keep_wait: float) -> float:
    """Correction to apply to the "wait" choice when only keep_wait of the waits were kept."""
    return math.log(1.0 / keep_wait)


def autocast(dev: str):
    """bf16 on the GPUs that support it (A100, A40, L40S, H100), otherwise full precision (RTX 8000, CPU)."""
    if dev.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
