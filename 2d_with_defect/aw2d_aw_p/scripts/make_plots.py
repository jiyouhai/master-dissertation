\
    #!/usr/bin/env python3
    import csv, os, numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read_csv(path):
        if not os.path.exists(path): return []
        with open(path) as f:
            return list(csv.DictReader(f))

    exact = read_csv("outputs/exact_summary.csv")
    trials = read_csv("outputs/trials_boxplot_stats.csv")

    for r in exact:
        for k in ["fraction","MFPT_time","Mode_time"]:
            r[k]=float(r[k])
    for r in trials:
        for k in ["fraction","q05","q25","q50","q75","q95","mean","mode_hist"]:
            r[k]=float(r[k])
    exact.sort(key=lambda r:r["fraction"])
    trials.sort(key=lambda r:r["fraction"])

    # 图1：解析趋势
    if exact:
        xs=[r["fraction"] for r in exact]
        y1=[r["MFPT_time"] for r in exact]
        y2=[r["Mode_time"] for r in exact]
        plt.figure(figsize=(6,4),dpi=150)
        plt.plot(xs,y1,marker='o',label="MFPT_time(解析)")
        plt.plot(xs,y2,marker='s',label="Mode_time(解析)")
        plt.xlabel("障碍比例 p"); plt.ylabel("时间")
        plt.title("解析：MFPT 与 Mode 随 p")
        plt.grid(True,ls="--",alpha=0.4); plt.legend(); plt.tight_layout()
        plt.savefig("outputs/trend_vs_p.png"); plt.close()

    # 图2：箱线近似 + 解析 + 试验众数
    if trials:
        ps=[r["fraction"] for r in trials]
        uniq=sorted(set(ps)); centers=np.arange(len(uniq))
        med=[]; q25=[]; q75=[]; q05=[]; q95=[]; modeh=[]
        for p in uniq:
            r=[x for x in trials if abs(x["fraction"]-p)<1e-12][0]
            med.append(r["q50"]); q25.append(r["q25"]); q75.append(r["q75"])
            q05.append(r["q05"]); q95.append(r["q95"]); modeh.append(r["mode_hist"])
        plt.figure(figsize=(8,4),dpi=150)
        for i,(c,l,u) in enumerate(zip(centers,q25,q75)):
            plt.plot([c-0.2,c+0.2],[l,l]); plt.plot([c-0.2,c+0.2],[u,u])
            plt.plot([c-0.2,c-0.2],[l,u]); plt.plot([c+0.2,c+0.2],[l,u])
        for i,(c,lo,hi,l,u) in enumerate(zip(centers,q05,q95,q25,q75)):
            plt.plot([c,c],[lo,l],lw=1); plt.plot([c,c],[u,hi],lw=1)
        plt.scatter(centers, med, marker='o', label="trial 中位数", zorder=3)
        plt.scatter(centers, modeh, marker='x', label="trial 众数 (FD)", zorder=3)
        if exact:
            ex_p=[r["fraction"] for r in exact]
            mfpt=[r["MFPT_time"] for r in exact]
            mode=[r["Mode_time"] for r in exact]
            mfpt_interp=np.interp(uniq, ex_p, mfpt)
            mode_interp=np.interp(uniq, ex_p, mode)
            plt.plot(centers, mfpt_interp, marker='^', label="MFPT_time(解析)")
            plt.plot(centers, mode_interp, marker='s', label="Mode_time(解析)")
        plt.xticks(centers,[f"{p:.2f}" for p in uniq]); plt.xlabel("障碍比例 p"); plt.ylabel("时间")
        plt.title("随机撒点箱线 + 试验众数 + 解析对照")
        plt.grid(True,ls="--",alpha=0.4); plt.legend(); plt.tight_layout()
        plt.savefig("outputs/boxplot_vs_p.png"); plt.close()

    print("[DONE] wrote outputs/trend_vs_p.png and outputs/boxplot_vs_p.png")
