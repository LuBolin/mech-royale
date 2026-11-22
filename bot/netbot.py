"""Single-threaded NumPy inference for the published recurrent policy."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from features import Encoder, SIZES, TEACHERS, decode

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_weights():
    with np.load(HERE / "policy.npz", allow_pickle=False) as file:
        return {k:np.asarray(v,dtype=np.float32,order="C") for k,v in file.items()}


def forward(w, x, hidden, teacher):
    core = np.maximum(0., w["input.weight"] @ x + w["input.bias"] + w["style.weight"][teacher])
    core = np.maximum(0., w["core.weight"] @ core + w["core.bias"])
    ir, iz, inn = np.split(w["gru.weight_ih_l0"] @ core + w["gru.bias_ih_l0"],3)
    hr, hz, hn = np.split(w["gru.weight_hh_l0"] @ hidden + w["gru.bias_hh_l0"],3)
    r = 1 / (1+np.exp(-np.clip(ir+hr,-60,60)))
    z = 1 / (1+np.exp(-np.clip(iz+hz,-60,60)))
    h = z*hidden + (1-z)*np.tanh(inn+r*hn)
    return w["heads.weight"] @ h + w["heads.bias"], h


class Bot:
    name = "Mech Royale GRU"

    def __init__(self, hello):
        self.encoder = Encoder(hello)
        self.w = load_weights()
        cfg = json.loads((HERE / "policy.json").read_text())
        self.style = TEACHERS.index(cfg["teacher"])
        self.hidden = np.zeros(self.w["gru.weight_hh_l0"].shape[1],np.float32)
        self.decisions = 0

    def act(self, obs):
        x, ctx = self.encoder.encode(obs)
        logits,self.hidden = forward(self.w,x,self.hidden,self.style)
        self.decisions += 1
        return decode(ctx,logits[:sum(SIZES)],obs["tick"])
