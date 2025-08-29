\
    #!/usr/bin/env python3
    import csv, math, sys
    inp="outputs/exact_summary.csv"
    mfpt_steps=None
    with open(inp) as f:
        r=csv.DictReader(f)
        rows=list(r)
        # 优先 fraction==0 的行；如无，则取该文件中 MFPT_steps 的第一行
        for row in rows:
            if float(row["fraction"])==0.0:
                mfpt_steps=float(row["MFPT_steps"]); break
        if mfpt_steps is None and rows:
            mfpt_steps=float(rows[0]["MFPT_steps"])
    if mfpt_steps is None:
        print("[ERR] Cannot find MFPT_steps from outputs/exact_summary.csv", file=sys.stderr)
        sys.exit(1)
    max_steps=math.ceil(5.0*mfpt_steps)
    open("outputs/max_steps.txt","w").write(str(max_steps)+"\n")
    print(f"[OK] MFPT_steps={mfpt_steps:.0f}, recommended max_steps={max_steps}")
