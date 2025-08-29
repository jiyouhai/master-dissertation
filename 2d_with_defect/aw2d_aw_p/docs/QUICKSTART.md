\
    # QUICKSTART

    ## 0) 环境
    ```bash
    module purge
    module load python/3.10
    python -m venv ~/aw2d_aw_p/venv
    source ~/aw2d_aw_p/venv/bin/activate
    pip install --upgrade pip -r requirements.txt
    ```

    ## 1) 解析（MFPT + mode）
    ```bash
    sbatch run_aw_exact_dense_N4001.slurm
    # 作业结束后：
    python scripts/summarize_exact.py
    ```

    ## 2) 推荐 max_steps（≈ 5×MFPT_steps）
    ```bash
    python scripts/compute_max_steps.py
    cat outputs/max_steps.txt
    ```

    ## 3) 生成并提交随机撒点数组作业（trial 不减，固定 M，并行）
    ```bash
    python scripts/mk_trials_slurm.py
    sbatch run_trials_N4001_array.slurm
    ```

    ## 4) 汇总与作图
    ```bash
    python scripts/summarize_trials.py
    python scripts/make_plots.py
    ls -lh outputs/*.png
    ```

    ### Sticky 与 Slippery
    - Sticky: `QDEF=0.5`（默认）
    - Slippery: `QDEF=1.2`（在 `run_aw_exact_dense_N4001.slurm` 与 `scripts/mk_trials_slurm.py` 中同时改）
