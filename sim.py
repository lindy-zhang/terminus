"""
The question I'm exploring:
-> does Terminus's central assumption that top results come early survive streaming updates and drift?
- Terminus itself is tested/applied to static indicices, not dynamic, so I want to see how its assumptions
hold for dynamic graph ANNS

My simulation:
  - Incremental Vamana-style graph (FreshDiskANN-like insert: greedy search + robust prune, back-edges).
  - CANDOR-style workloads: warm-up build, then a stream of insert batches, optionally with
    distribution drift and sliding-window deletions (tombstones, optionally periodic consolidation).
  - At checkpoints, run beam search for queries drawn from the *current* distribution and record,
    for every node expansion ("step" ~ one random disk read in DiskANN), which results entered the queue
    and at which rank. From this we get:
      * the step at which each true top-k result (rank 0..k-1) is first discovered
      * the per-step Terminus utility U_t = sum_{new ranks r} w(r), w(r)=exp(-(r/tau)^beta)
  - Replay Terminus' stopping rule (stop when last X utilities <= eps) on the traces and report
    Recall@k, Ranked Recall@k and the fraction of steps saved versus the natural stopping rule.
"""
import numpy as np, json, sys, time, argparse
from multiprocessing import Pool

D = 32
K = 10
TAU, BETA, X_WIN = 1.8, 0.5, 2
EPS_LIST = [0.05, 0.1, 0.2]


def w(r):
    return np.exp(-(np.asarray(r, dtype=float) / TAU) ** BETA)


W_K = w(np.arange(K))


class Index:
    def __init__(self, cap, R=24, Lb=48, alpha=1.2):
        self.X = np.zeros((cap, D), np.float32)
        self.n = 0
        self.nbr = [[] for _ in range(cap)]
        self.deleted = np.zeros(cap, bool)
        self.R, self.Lb, self.alpha = R, Lb, alpha
        self.entry = 0

    def d(self, q, ids):
        v = self.X[ids] - q
        return np.einsum("ij,ij->i", v, v)

    def search(self, q, L, trace=False):
        e = self.entry
        d0 = float(self.d(q, [e])[0])
        visited = set()
        scored = {e: d0}
        first = {e: 0}
        cand = [(d0, e)]
        utils = []
        step = 0
        while True:
            nxt = None
            for _, i in cand:
                if i not in visited:
                    nxt = i
                    break
            if nxt is None:
                break
            visited.add(nxt)
            step += 1
            nb = [j for j in self.nbr[nxt] if j not in scored]
            if nb:
                ds = self.d(q, nb).tolist()
                for j, dj in zip(nb, ds):
                    scored[j] = dj
                    first[j] = step
                merged = sorted(cand + list(zip(ds, nb)))
                out, live = [], 0
                for dd, i in merged:  # queue keeps L *live* entries (deleted ones stay navigable)
                    out.append((dd, i))
                    if not self.deleted[i]:
                        live += 1
                        if live >= L:
                            break
                cand = out
            if trace:
                new = set(nb)
                ranks, r = [], 0
                for _, i in cand:
                    if self.deleted[i]:
                        continue
                    if i in new:
                        ranks.append(r)
                    r += 1
                utils.append(float(w(ranks).sum()) if ranks else 0.0)
        return cand, visited, scored, first, utils, step

    def prune(self, p, pool):
        pool = np.array([i for i in set(pool) if i != p and not self.deleted[i]], dtype=np.int64)
        if len(pool) == 0:
            return []
        dp = self.d(self.X[p], pool)
        o = np.argsort(dp)
        pool, dp = pool[o], dp[o]
        alive = np.ones(len(pool), bool)
        out = []
        a2 = self.alpha * self.alpha
        for idx in range(len(pool)):
            if not alive[idx]:
                continue
            c = int(pool[idx])
            out.append(c)
            if len(out) >= self.R:
                break
            rest = np.nonzero(alive[idx + 1:])[0] + idx + 1
            if len(rest) == 0:
                break
            dc = self.d(self.X[c], pool[rest])
            alive[rest[a2 * dc <= dp[rest]]] = False
        return out

    def insert(self, x):
        p = self.n
        self.X[p] = x
        self.n += 1
        if p == 0:
            return
        _, visited, _, _, _, _ = self.search(x, self.Lb)
        self.nbr[p] = self.prune(p, list(visited))
        for j in self.nbr[p]:
            if p not in self.nbr[j]:
                self.nbr[j].append(p)
                if len(self.nbr[j]) > self.R:
                    self.nbr[j] = self.prune(j, self.nbr[j] + [p])

    def delete(self, ids):
        self.deleted[ids] = True

    def consolidate(self):
        """FreshDiskANN-style delete consolidation: route edges around deleted nodes, then drop them."""
        live = np.nonzero(~self.deleted[: self.n])[0]
        for i in live:
            nb = self.nbr[i]
            if any(self.deleted[j] for j in nb):
                pool = [j for j in nb if not self.deleted[j]]
                for j in nb:
                    if self.deleted[j]:
                        pool += self.nbr[j]
                self.nbr[i] = self.prune(i, pool)
        for j in np.nonzero(self.deleted[: self.n])[0]:
            self.nbr[j] = []
        if self.deleted[self.entry]:
            self.set_entry()

    def set_entry(self):
        live = np.nonzero(~self.deleted[: self.n])[0]
        c = self.X[live].mean(0)
        self.entry = int(live[np.argmin(self.d(c, live))])

    def ground_truth(self, q, k):
        live = np.nonzero(~self.deleted[: self.n])[0]
        dd = self.d(q, live)
        o = np.argpartition(dd, k)[:k]
        o = o[np.argsort(dd[o])]
        return live[o]


class Stream:
    """Gaussian-mixture data whose cluster centers move from `base` to `target` as frac goes 0 -> 1."""

    def __init__(self, seed, C=64):
        r = np.random.default_rng(seed)
        self.base = r.normal(0, 1.5, (C, D))
        self.target = self.base + r.normal(0, 2.0, (C, D))
        self.C = C

    def sample(self, rng, m, frac):
        centers = (1 - frac) * self.base + frac * self.target
        ids = rng.integers(self.C, size=m)
        return (centers[ids] + rng.normal(0, 1, (m, D))).astype(np.float32)


def evaluate(index, qs, Lq=48):
    rows = []
    for q in qs:
        gt = index.ground_truth(q, K)
        cand, visited, scored, first, utils, steps = index.search(q, Lq, trace=True)
        disc = [first.get(int(g), None) for g in gt]  # step at which rank-i result first discovered

        def topk_at(T):
            items = [(scored[i], i) for i, s in first.items() if s <= T and not index.deleted[i]]
            items.sort()
            return [i for _, i in items[:K]]

        def rr(res):
            s = set(res)
            hit = np.array([int(g) in s for g in gt], float)
            return hit.mean(), float((W_K * hit).sum() / W_K.sum())

        nat = rr(topk_at(steps))
        term = {}
        for eps in EPS_LIST:
            T = steps
            for t in range(X_WIN, len(utils) + 1):
                if all(u <= eps for u in utils[t - X_WIN:t]):
                    T = t
                    break
            rec, rrec = rr(topk_at(T))
            term[eps] = dict(T=T, recall=rec, rrecall=rrec)
        rows.append(dict(disc=disc, steps=steps, nat_recall=nat[0], nat_rrecall=nat[1], term=term))
    return rows


def summarize(rows):
    disc = np.array([[np.nan if v is None else v for v in r["disc"]] for r in rows], float)
    steps = np.array([r["steps"] for r in rows], float)
    out = dict(
        steps_mean=float(steps.mean()),
        disc_median=np.nanmedian(disc, 0).tolist(),
        disc_p90=np.nanpercentile(disc, 90, 0).tolist(),
        disc_frac_of_search=np.nanmedian(disc / steps[:, None], 0).tolist(),
        found_by_10=float(np.nanmean(np.where(np.isnan(disc), 0, disc <= 10))),
        missed_frac=np.isnan(disc).mean(0).tolist(),
        nat_recall=float(np.mean([r["nat_recall"] for r in rows])),
        nat_rrecall=float(np.mean([r["nat_rrecall"] for r in rows])),
        term={},
    )
    for eps in EPS_LIST:
        T = np.array([r["term"][eps]["T"] for r in rows], float)
        out["term"][str(eps)] = dict(
            steps_mean=float(T.mean()),
            saved=float(1 - T.sum() / steps.sum()),
            recall=float(np.mean([r["term"][eps]["recall"] for r in rows])),
            rrecall=float(np.mean([r["term"][eps]["rrecall"] for r in rows])),
        )
    return out


SCENARIOS = {
    "stationary": dict(drift="none", window=None, consolidate=None),
    "gradual_drift": dict(drift="gradual", window=None, consolidate=None),
    "abrupt_drift": dict(drift="abrupt", window=None, consolidate=None),
    "drift+deletes(tombstone)": dict(drift="gradual", window=6000, consolidate=None),
    "drift+deletes(consolidated)": dict(drift="gradual", window=6000, consolidate=3),
}


def run(name, warm=4000, batches=12, bsz=1000, ck_every=3, nq=200, seed=1):
    cfg = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    qrng = np.random.default_rng(seed + 100)
    st = Stream(seed)
    idx = Index(warm + batches * bsz)
    t0 = time.time()
    for x in st.sample(rng, warm, 0.0):
        idx.insert(x)
    idx.set_entry()

    def frac_at(b):
        if cfg["drift"] == "none":
            return 0.0
        if cfg["drift"] == "gradual":
            return b / batches
        return 0.0 if b <= batches // 2 else 1.0

    results = []

    def checkpoint(b):
        qs = st.sample(qrng, nq, frac_at(b))
        s = summarize(evaluate(idx, qs))
        s.update(batch=b, live=int((~idx.deleted[: idx.n]).sum()))
        results.append(s)
        print(f"{name}: batch {b} done ({time.time() - t0:.0f}s)", flush=True)

    checkpoint(0)
    for b in range(1, batches + 1):
        for x in st.sample(rng, bsz, frac_at(b)):
            idx.insert(x)
        if cfg["window"]:
            live = np.nonzero(~idx.deleted[: idx.n])[0]
            if len(live) > cfg["window"]:
                idx.delete(live[: len(live) - cfg["window"]])  # expire oldest
                if idx.deleted[idx.entry]:
                    idx.set_entry()
        if cfg["consolidate"] and b % cfg["consolidate"] == 0:
            idx.consolidate()
        if b % ck_every == 0:
            checkpoint(b)
    return name, dict(results=results, secs=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    names = list(SCENARIOS)
    kw = dict(warm=1000, batches=6, bsz=300, ck_every=3, nq=50) if a.quick else {}
    with Pool(len(names)) as pool:
        res = dict(pool.starmap(run, [(n,) for n in names], chunksize=1) if not kw else
                   [run(n, **kw) for n in names])
    json.dump(res, open("results.json", "w"), indent=1)
    for n, r in res.items():
        print(n, f"{r['secs']:.0f}s")
        for c in r["results"]:
            t = c["term"]["0.1"]
            print(f"  batch {c['batch']:2d} live {c['live']:5d} | steps {c['steps_mean']:5.1f} | "
                  f"rank0 med {c['disc_median'][0]:4.1f} rank9 med {c['disc_median'][9]:4.1f} | "
                  f"natRR {c['nat_rrecall']:.3f} natR {c['nat_recall']:.3f} | "
                  f"Term(.1) RR {t['rrecall']:.3f} R {t['recall']:.3f} saved {t['saved']:.2f}")