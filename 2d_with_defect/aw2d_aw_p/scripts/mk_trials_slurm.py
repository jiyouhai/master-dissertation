\
    #!/usr/bin/env python3
    import math, sys, os

    N=4001; Q=0.8; QDEF=0.5
    CPUS=8; MEM="32G"; TIME="12:00:00"
    CONC=24  # 并发上限
    TRIALS_PER_JOB=25
    CHUNKS=40
    FRACTIONS=[0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30]

    max_steps_path="outputs/max_steps.txt"
    if not os.path.exists(max_steps_path):
        print("[ERR] outputs/max_steps.txt not found. Run scripts/compute_max_steps.py first.", file=sys.stderr)
        sys.exit(1)
    MAX_STEPS=int(open(max_steps_path).read().strip())

    total=len(FRACTIONS)*CHUNKS
    slurm=f\"\"\"#!/bin/bash
    #SBATCH --job-name=aw_trials_N{N}
    #SBATCH --account=emat004321
    #SBATCH --time={TIME}
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={CPUS}
    #SBATCH --mem={MEM}
    #SBATCH --output=logs/trials{N}.%A_%a.out
    #SBATCH --error=logs/trials{N}.%A_%a.err
    #SBATCH --array=0-{total-1}%{CONC}

    set -euo pipefail
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    source ~/aw2d_aw_p/venv/bin/activate
    cd ~/aw2d_aw_p
    mkdir -p outputs logs

    N={N}; Q={Q}; QDEF={QDEF}
    WORKERS=${{SLURM_CPUS_PER_TASK}}
    TRIALS_PER_JOB={TRIALS_PER_JOB}
    CHUNKS={CHUNKS}
    FRACTIONS=({ ' '.join([f"{p:.2f}" for p in FRACTIONS]) })

    P_COUNT=${{#FRACTIONS[@]}}
    AID=${{SLURM_ARRAY_TASK_ID}}
    PIDX=$(( AID / CHUNKS ))
    CHUNK=$(( AID % CHUNKS ))
    P=${{FRACTIONS[$PIDX]}}

    OUTFILE="outputs/trials_N${{N}}_p${{P}}_chunk${{CHUNK}}_jid${{SLURM_JOB_ID}}_aid${{AID}}.jsonl"

    python src/aw2d_mc_trials.py \\
      --N_mc ${{N}} \\
      --q ${{Q}} --qdef ${{QDEF}} \\
      --fraction ${{P}} \\
      --mask_mode fixed \\
      --trials ${{TRIALS_PER_JOB}} \\
      --reps 1 \\
      --workers ${{WORKERS}} \\
      --trial_offset $(( CHUNK * TRIALS_PER_JOB )) \\
      --start 0,0 --target N-1,N-1 \\
      --max_steps {MAX_STEPS} \\
      --seed $(( 24680 + AID )) \\
      --out ${{OUTFILE}}

    echo "[DONE] p=${{P}} chunk=${{CHUNK}} -> ${{OUTFILE}}"
    \"\"\"
    open("run_trials_N4001_array.slurm","w").write(slurm)
    print("[OK] wrote run_trials_N4001_array.slurm with MAX_STEPS =", MAX_STEPS)
