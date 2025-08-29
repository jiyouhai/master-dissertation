\
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    import os, math, time, argparse, json
    import numpy as np

    def parse_pair(s: str):
        a, b = s.split(","); return int(a), int(b)

    class Homogeneous2D:
        def __init__(self, N:int, start=(0,0), target=None):
            self.N = int(N)
            if target is None: target = (self.N-1, self.N-1)
            k = np.arange(self.N, dtype=float)
            cos_k = np.cos(k * np.pi / self.N)
            self.A = 0.5 * (cos_k[:, None] + cos_k[None, :])  # λ_kl

            def h_vec(n, n0):
                N=self.N; h=np.zeros(N, dtype=float); h[0]=1.0/N
                if N>1:
                    kk=np.arange(1,N,dtype=float)
                    h[1:]=(2.0/N)*np.cos(kk*np.pi*(2*n+1)/(2*N))*np.cos(kk*np.pi*(2*n0+1)/(2*N))
                return h
            xs,ys=start; xt,yt=target
            self.Ws = h_vec(xt, xs)[:, None] * h_vec(yt, ys)[None, :]
            self.Wt = h_vec(xt, xt)[:, None] * h_vec(yt, yt)[None, :]

        def mfpt_steps(self, tile:int=0)->float:
            N=self.N; s=0.0
            if tile and tile<N:
                for i in range(0,N,tile):
                    j=min(i+tile,N)
                    Ablk=self.A[i:j,:]
                    denom=1.0-Ablk
                    diffW=self.Wt[i:j,:]-self.Ws[i:j,:]
                    if i==0:
                        denom=denom.copy(); diffW=diffW.copy()
                        denom[0,0]=1.0; diffW[0,0]=0.0
                    s+=float(np.sum(diffW/denom))
            else:
                denom=1.0-self.A; diffW=self.Wt-self.Ws
                mask=np.ones((N,N),dtype=bool); mask[0,0]=False
                s=float(np.sum((diffW/denom)[mask]))
            return (N*N)*s

        def F(self, z: complex, tile:int=0) -> complex:
            if not (tile and tile<self.N):
                D0=1.0/(1.0 - z*self.A)
                Gs=np.sum(self.Ws*D0); Gt=np.sum(self.Wt*D0)
                return Gs/Gt
            Gs=0.0+0.0j; Gt=0.0+0.0j
            for i in range(0,self.N,tile):
                j=min(i+tile,self.N); D0=1.0/(1.0 - z*self.A[i:j,:])
                Gs+=np.sum(self.Ws[i:j,:]*D0); Gt+=np.sum(self.Wt[i:j,:]*D0)
            return Gs/Gt

    def euler_invert_pdf(phi, t: float, M: int = 22):
        if t<=0: return 0.0
        k=np.arange(0,2*M+1); beta=(M*math.log(10.0)/3.0)+1j*np.pi*k; s=beta/float(t)
        xi=np.zeros(2*M+1); xi[0]=0.5; xi[1:M+1]=1.0; xi[-1]=0.5/M
        eta=(( -1.0)**k)*xi
        vals=np.array([phi(complex(si)) for si in s], dtype=complex)
        out=(10.0**(M/3.0))/float(t)*np.sum(eta*np.real(vals))
        return max(0.0,float(out))

    def find_mode_time(phi, t0: float, M: int = 22):
        t0=max(1e-9,float(t0))
        tL=t0/2; fL=euler_invert_pdf(phi,tL,M)
        tC=t0;   fC=euler_invert_pdf(phi,tC,M)
        tR=t0*2; fR=euler_invert_pdf(phi,tR,M)
        for _ in range(6):
            if fL<=fC>=fR: break
            if fL>fC:
                tR,fR,tC,fC,tL = tC,fC,tL,fL,max(1e-9,tL/2)
            else:
                tL,fL,tC,fC,tR = tC,fC,tR,fR,tR*2
            fL=euler_invert_pdf(phi,tL,M); fR=euler_invert_pdf(phi,tR,M)
        gr=(math.sqrt(5)-1)/2; a,b=tL,tR
        c=b-gr*(b-a); d=a+gr*(b-a)
        fc=euler_invert_pdf(phi,c,M); fd=euler_invert_pdf(phi,d,M)
        for _ in range(18):
            if fc>fd:
                b,d,fd=c,c,fc; c=b-gr*(b-a); fc=euler_invert_pdf(phi,c,M)
            else:
                a,c,fc=d,d,fd; d=a+gr*(b-a); fd=euler_invert_pdf(phi,d,M)
        return 0.5*(a+b)

    def run_one_fraction(N,q,qdef,p,tile,aw_M,start,target,one_based):
        if one_based:
            start=(start[0]-1,start[1]-1); target=(target[0]-1,target[1]-1)
        # 调和平均的等效速率
        q_eff = 1.0 / ((1.0 - p) / q + p / qdef)
        model = Homogeneous2D(N,start=start,target=target)
        t0=time.time()
        mu_steps = model.mfpt_steps(tile=tile)
        mu_time  = mu_steps / q_eff
        def phi(s: complex):
            z = q_eff / (s + q_eff)
            return complex(model.F(z, tile=tile))
        t_mode = find_mode_time(phi, max(1e-9, mu_time), M=aw_M)
        return dict(N=N,q=q,qdef=qdef,fraction=p,q_eff=float(q_eff),
                    start=start,target=target,one_based=bool(one_based),
                    MFPT_steps=float(mu_steps), MFPT_time=float(mu_time),
                    Mode_time=float(t_mode), Mode_steps=float(max(1.0,q_eff*t_mode)),
                    elapsed_sec=time.time()-t0)

    def main():
        ap=argparse.ArgumentParser(description="Exact MFPT(steps)+AW mode with q_eff (harmonic)")
        ap.add_argument("--N",type=int,default=4001)
        ap.add_argument("--q",type=float,default=0.8)
        ap.add_argument("--qdef",type=float,default=0.5)
        ap.add_argument("--fractions",type=str,
            default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30")
        ap.add_argument("--tile",type=int,default=512)
        ap.add_argument("--aw_M",type=int,default=22)
        ap.add_argument("--start",type=str,default="0,0")
        ap.add_argument("--target",type=str,default="N-1,N-1")
        ap.add_argument("--one_based",action="store_true")
        ap.add_argument("--procs",type=int,default=1)
        ap.add_argument("--out",type=str,default="outputs/aw_exact_big_N{N}.json")
        args=ap.parse_args()

        tgt=args.target.replace("N-1",str(args.N-1))
        start=parse_pair(args.start); target=parse_pair(tgt)

        os.environ.setdefault("OMP_NUM_THREADS","1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
        os.environ.setdefault("MKL_NUM_THREADS","1")

        fracs=[float(x) for x in args.fractions.split(",") if x.strip()!=""]
        table=[]; t_all=time.time()
        for p in fracs:
            rec=run_one_fraction(args.N,args.q,args.qdef,p,args.tile,args.aw_M,start,target,args.one_based)
            print(json.dumps(rec,ensure_ascii=False))
            table.append(rec)
        outp=args.out.format(N=args.N)
        os.makedirs("outputs",exist_ok=True)
        with open(outp,"w") as fp: json.dump(table,fp,indent=2,ensure_ascii=False)
        print(f"[DONE] Saved {len(table)} rows to {outp}. Total elapsed {time.time()-t_all:.1f} sec.")
    if __name__=="__main__": main()
