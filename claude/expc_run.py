"""Monte Carlo runner for Experiment C (adaptive online LQR).

Usage:
  python expc_run.py --mode old   --trials 20 --out mc_old.npz
  python expc_run.py --mode tuned --trials 20 --out mc_tuned.npz
  python expc_run.py --mode tuned --set M_fw=100 beta_explore_scale=0.1 ...

Seeds follow the notebook convention: seed = trial + 17, trial = 0..N-1.
Saves per-seed cumulative regret curves (T,) for fw / naive / sampling,
the config, and FW per-episode diagnostics.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def _worker(args):
    seed, cfg_dict, methods = args
    import expc_core as core
    cfg = core.Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    t0 = time.time()
    out = core.run_single_seed(seed, cfg, methods=tuple(methods))
    out["runtime_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "tuned"], default="tuned")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=17)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--methods", type=str, default="fw,naive,sampling")
    ap.add_argument("--set", nargs="*", default=[],
                    help="config overrides key=value (python literals)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import expc_core as core

    cfg = core.notebook_config() if args.mode == "old" else core.Config()
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if not hasattr(cfg, k):
            raise KeyError(k)
        setattr(cfg, k, v)
    cfg_dict = cfg.as_dict()
    methods = args.methods.split(",")
    seeds = [args.seed0 + t for t in range(args.trials)]
    print(f"mode={args.mode} seeds={seeds}")
    print("config:", json.dumps(cfg_dict))

    jobs = [(s, cfg_dict, methods) for s in seeds]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            results = []
            for i, r in enumerate(ex.map(_worker, jobs)):
                results.append(r)
                msg = f"seed {r['seed']} done ({i+1}/{len(jobs)}) {r['runtime_s']:.0f}s"
                for meth in methods:
                    if meth in r:
                        msg += f" | {meth} final={r[meth][-1]:.3g}"
                print(msg, flush=True)
    else:
        results = []
        for j in jobs:
            r = _worker(j)
            results.append(r)
            print(f"seed {r['seed']} done {r['runtime_s']:.0f}s", flush=True)

    save = {"seeds": np.array(seeds), "config_json": json.dumps(cfg_dict),
            "tau": results[0]["tau"], "J_star": results[0]["J_star"]}
    for meth in methods:
        if meth in results[0]:
            save[meth] = np.stack([r[meth] for r in results])
    if "fw_diag" in results[0]:
        save["fw_diag_json"] = json.dumps([r["fw_diag"] for r in results])
    np.savez_compressed(args.out, **save)
    print(f"saved {args.out}  total {time.time()-t0:.0f}s")
    for meth in methods:
        if meth in save:
            fin = save[meth][:, -1]
            print(f"{meth:9s} final regret: mean={np.mean(fin):.6g} "
                  f"median={np.median(fin):.6g} p15={np.percentile(fin,15):.6g} "
                  f"p85={np.percentile(fin,85):.6g} min={fin.min():.6g} "
                  f"max={fin.max():.6g}")


if __name__ == "__main__":
    main()
