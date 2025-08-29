\
    #!/usr/bin/env python3
    import json, glob, csv, argparse, numpy as np
    from collections import defaultdict

    def fd_mode(vals: np.ndarray):
        """Freedman–Diaconis binning + histogram mode (bin center)."""
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return np.nan, np.nan, 0
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = max(q75 - q25, 0.0)
        n = vals.size
        if iqr <= 0.0:
            # fallback: Scott / sqrt rule
            bins = max(10, int(np.sqrt(n)))
        else:
            h = 2.0 * iqr / (n ** (1.0/3.0))
            rng = vals.max() - vals.min()
            if h <= 0.0 or not np.isfinite(h) or rng == 0.0:
                bins = max(10, int(np.sqrt(n)))
            else:
                bins = int(np.clip(np.ceil(rng / h), 10, 5000))
        hist, edges = np.histogram(vals, bins=bins)
        if hist.size == 0:
            return np.nan, np.nan, 0
        idx = int(np.argmax(hist))
        mode = 0.5 * (edges[idx] + edges[idx+1])
        bw = edges[idx+1] - edges[idx]
        return float(mode), float(bw), int(bins)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hits-only", action="store_true", help="仅统计 hit==True 的样本")
    ap.add_argument("--out", type=str, default="outputs/trials_boxplot_stats.csv")
    args = ap.parse_args()

    # 按目标比例分组（优先 fraction_target；无则用 fraction_real 四舍五入）
    groups = defaultdict(list)
    meta = dict(N=None, q=None, qdef=None)
    files = sorted(glob.glob("outputs/trials_N*_p*.jsonl"))
    if not files:
        print("[WARN] no JSONL files found")

    for f in files:
        with open(f) as fp:
            for line in fp:
                r = json.loads(line)
                if meta["N"] is None: meta["N"]=r["N"]
                if meta["q"] is None: meta["q"]=r["q"]
                if meta["qdef"] is None: meta["qdef"]=r["qdef"]
                p = r.get("fraction_target", None)
                if p is None:
                    p = round(float(r["fraction_real"]), 6)
                hit = bool(r.get("hit", True))
                if (not args.hits_only) or hit:
                    groups[p].append(float(r["KMC_time"]))

    rows=[]
    for p in sorted(groups.keys(), key=lambda x: float(x)):
        vals = np.array(groups[p], dtype=float)
        if vals.size == 0: continue
        qs = np.percentile(vals, [5,25,50,75,95])
        mode_hist, bw, bins = fd_mode(vals)
        rows.append([meta["N"], meta["q"], meta["qdef"], float(p), vals.size,
                     qs[0], qs[1], qs[2], qs[3], qs[4], float(vals.mean()),
                     float(mode_hist), float(bw), int(bins)])

    with open(args.out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["N","q","qdef","fraction","n",
                    "q05","q25","q50","q75","q95","mean",
                    "mode_hist","fd_binwidth","fd_bins"])
        w.writerows(rows)

    print(f"[DONE] {args.out} with {len(rows)} p-groups; total samples = {sum(r[4] for r in rows)}")
