\
    #!/usr/bin/env python3
    import json, glob, csv
    rows=[]
    for f in sorted(glob.glob("outputs/aw_exact_big_N*_f*.json")):
        try:
            arr=json.load(open(f))
            if isinstance(arr,dict): arr=[arr]
            rows.extend(arr)
        except Exception as e:
            print("[WARN]", f, e)
    rows.sort(key=lambda r:(r.get("N",0),r.get("q",0),r.get("qdef",0),r.get("fraction",0)))
    with open("outputs/exact_summary.csv","w",newline="") as g:
        w=csv.writer(g)
        w.writerow(["N","q","qdef","fraction","q_eff","MFPT_steps","MFPT_time","Mode_time","Mode_steps"])
        for r in rows:
            w.writerow([r["N"],r["q"],r["qdef"],r["fraction"],r["q_eff"],
                        r["MFPT_steps"],r["MFPT_time"],r["Mode_time"],r["Mode_steps"]])
    print("[DONE] outputs/exact_summary.csv with", len(rows), "rows")
