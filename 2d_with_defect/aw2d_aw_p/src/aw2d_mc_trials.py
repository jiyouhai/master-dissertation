\
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    import os, json, argparse, numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        from numba import njit
        JIT=True
    except Exception:
        JIT=False

    def parse_pair(s:str):
        a,b=s.split(","); return int(a),int(b)

    def make_mask_fixed(N, p, rng):
        M=int(round(float(p)*N*N))
        idx=rng.choice(N*N,size=M,replace=False)
        mask=np.zeros(N*N,dtype=np.uint8); mask[idx]=1
        return mask.reshape(N,N), M, M/(N*N)

    def make_mask_bernoulli(N,p,rng):
        mask=(rng.random((N,N))<float(p)).astype(np.uint8)
        M=int(mask.sum()); return mask, M, M/(N*N)

    if JIT:
        from numba import njit
        @njit(cache=True)
        def step_reflect(x,y,N,u):
            if u<0.25: return (x, y-1) if y>0 else (x,1)
            elif u<0.5: return (x, y+1) if y<N-1 else (x, N-2)
            elif u<0.75: return (x-1,y) if x>0 else (1,y)
            else: return (x+1,y) if x<N-1 else (N-2,y)
        @njit(cache=True)
        def simulate_once_seeded(N,q,qdef,mask,xs,ys,xt,yt,max_steps,seed):
            np.random.seed(seed)
            x,y=xs,ys; t=0.0; steps=0; hit_def=0
            while True:
                if x==xt and y==yt: break
                rate = qdef if mask[x,y]==1 else q
                u = np.random.random(); 
                if u<=1e-300: u=1e-300
                t += -np.log(u)/rate
                v = np.random.random()
                x,y = step_reflect(x,y,N,v)
                steps += 1
                if mask[x,y]==1: hit_def += 1
                if steps>=max_steps: break
            return float(t), int(steps), int(hit_def)
    else:
        def step_reflect(x,y,N,u):
            if u<0.25: return (x, y-1) if y>0 else (x,1)
            elif u<0.5: return (x, y+1) if y<N-1 else (x, N-2)
            elif u<0.75: return (x-1,y) if x>0 else (1,y)
            else: return (x+1,y) if x<N-1 else (N-2,y)
        def simulate_once_py(N,q,qdef,mask,xs,ys,xt,yt,max_steps,rng):
            x,y=xs,ys; t=0.0; steps=0; hit_def=0
            while True:
                if x==xt and y==yt: break
                rate=qdef if mask[x,y]==1 else q
                u=rng.random(); 
                if u<=1e-300: u=1e-300
                t += -np.log(u)/rate
                v=rng.random()
                x,y = step_reflect(x,y,N,v)
                steps += 1
                if mask[x,y]==1: hit_def += 1
                if steps>=max_steps: break
            return float(t), int(steps), int(hit_def)

    def one_trial(N,q,qdef,start,target,mask,max_steps,seed):
        xs,ys=start; xt,yt=target
        if JIT:
            return simulate_once_seeded(N,q,qdef,mask,xs,ys,xt,yt,max_steps,seed)
        else:
            rng=np.random.default_rng(seed)
            return simulate_once_py(N,q,qdef,mask,xs,ys,xt,yt,max_steps,rng)

    def run_worker(wid, args, out_path, seed_base, trials_this_worker):
        rng=np.random.default_rng(seed_base+10007*wid)
        N=args.N_mc; xs,ys=args.start; xt,yt=args.target
        with open(out_path,"a") as fp:
            for k in range(trials_this_worker):
                if args.mask_mode=="fixed":
                    mask, M_used, frac_real = make_mask_fixed(N,args.fraction,rng)
                else:
                    mask, M_used, frac_real = make_mask_bernoulli(N,args.fraction,rng)
                for r in range(args.reps):
                    seed = seed_base + wid*1_000_003 + k*9176 + r*101
                    t,steps,hit_def = one_trial(N,args.q,args.qdef,(xs,ys),(xt,yt),mask,args.max_steps,seed)
                    rec=dict(N=N,q=args.q,qdef=args.qdef,
                             fraction_target=float(args.fraction), fraction_real=float(frac_real),
                             M=int(M_used), start=[xs,ys], target=[xt,yt],
                             trial_id=int(args.trial_offset + wid*trials_this_worker + k),
                             rep=int(r), worker=int(wid), seed=int(seed_base),
                             KMC_time=float(t), KMC_steps=int(steps), steps_on_defect=int(hit_def),
                             hit=bool(steps<args.max_steps))
                    fp.write(json.dumps(rec,ensure_ascii=False)+"\n"); fp.flush()

    def main():
        ap=argparse.ArgumentParser(description="Random-mask KMC trials -> JSONL")
        ap.add_argument("--N_mc",type=int,default=4001)
        ap.add_argument("--q",type=float,default=0.8)
        ap.add_argument("--qdef",type=float,default=0.5)
        ap.add_argument("--fraction",type=float,required=True)
        ap.add_argument("--mask_mode",type=str,default="fixed",choices=["fixed","bernoulli"])
        ap.add_argument("--start",type=str,default="0,0")
        ap.add_argument("--target",type=str,default="N-1,N-1")
        ap.add_argument("--trials",type=int,default=60)
        ap.add_argument("--reps",type=int,default=1)
        ap.add_argument("--workers",type=int,default=1)
        ap.add_argument("--trial_offset",type=int,default=0)
        ap.add_argument("--seed",type=int,default=2025)
        ap.add_argument("--max_steps",type=int,required=True)
        ap.add_argument("--out",type=str,required=True)
        args=ap.parse_args()

        tgt=args.target.replace("N-1",str(args.N_mc-1))
        args.start=parse_pair(args.start); args.target=parse_pair(tgt)
        os.makedirs(os.path.dirname(args.out),exist_ok=True)

        # 预热 JIT（无文件写入）
        if JIT:
            from numba import njit
            import numpy as np
            Nw=5; dummy=np.zeros((Nw,Nw),dtype=np.uint8)
            simulate_once_seeded(Nw,args.q,args.qdef,dummy,0,0,4,4,100,args.seed)

        W=max(1,int(args.workers))
        base=args.trials//W; rem=args.trials%W
        pieces=[base+(1 if i<rem else 0) for i in range(W)]
        seed_base=int(args.seed); out_path=args.out

        if W==1:
            run_worker(0,args,out_path,seed_base,pieces[0])
        else:
            with ProcessPoolExecutor(max_workers=W) as ex:
                futs=[ex.submit(run_worker,i,args,out_path,seed_base,tw) for i,tw in enumerate(pieces)]
                for fu in as_completed(futs): fu.result()
        print(f"[DONE] Wrote trials to {out_path}")
    if __name__=="__main__": main()
