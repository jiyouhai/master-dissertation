\
    # aw2d_aw_p

    从零到全跑通的方案 A（增大 `max_steps` 直到几乎无删失、N=4001、trial 数不减）：
    - 解析 MFPT(steps) + AW(Euler) 反演求 mode
    - 从解析结果自动推荐 `max_steps ≈ 5×MFPT_steps`
    - 随机撒点（固定 M）、数组作业 + 进程并行、trial 输出 JSONL
    - 汇总（含样本众数 `mode_hist`）与作图

    详见 `docs/QUICKSTART.md`。
