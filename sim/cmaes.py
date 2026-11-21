"""Minimal CMA-ES (Hansen's standard update), minimising f. numpy only."""
import numpy as np


class CMA:
    def __init__(self, x0, sigma, popsize=None, seed=0):
        n = self.n = len(x0)
        self.mean = np.array(x0, float)
        self.sigma = float(sigma)
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D, self.C = np.eye(n), np.ones(n), np.eye(n)
        self.g = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, X, f):
        X, f = np.asarray(X), np.asarray(f)
        idx = np.argsort(f)[:self.mu]
        Y = (X[idx] - self.mean) / self.sigma
        yw = self.w @ Y
        self.mean = self.mean + self.sigma * yw
        inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt @ yw
        self.g += 1
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(self.sigma, 1.0)  # the search box is [0, 1]^n
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


if __name__ == "__main__":  # self-check: sphere converges
    es = CMA(np.full(10, 3.0), 1.0, seed=1)
    for _ in range(300):
        X = es.ask()
        es.tell(X, (X ** 2).sum(1))
    assert np.abs(es.mean).max() < 1e-3, es.mean
    print("cmaes ok", np.abs(es.mean).max())
