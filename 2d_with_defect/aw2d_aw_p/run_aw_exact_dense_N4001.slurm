\
    #!/bin/bash
    #SBATCH --job-name=aw2d_exact_N4001
    #SBATCH --account=emat004321
    #SBATCH --time=12:00:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G
    #SBATCH --output=logs/aw_exact_big.%A_%a.out
    #SBATCH --error=logs/aw_exact_big.%A_%a.err
    #SBATCH --array=0-15%6

    set -euo pipefail
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    source ~/aw2d_aw_p/venv/bin/activate
    cd ~/aw2d_aw_p
    mkdir -p outputs logs

    N=4001; TGT=$((N-1))
    FRACTIONS=(0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30)
    F=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

    python src/aw2d_aw_exact_big.py \
      --N ${N} --q 0.8 --qdef 0.5 \
      --fractions ${F} \
      --tile 512 --aw_M 22 \
      --start 0,0 --target ${TGT},${TGT} \
      --procs ${SLURM_CPUS_PER_TASK} \
      --out outputs/aw_exact_big_N{N}_f${F}.json

    echo "[DONE] fraction=${F}"
