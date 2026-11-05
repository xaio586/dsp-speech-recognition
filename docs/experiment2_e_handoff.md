# 实验二代码交接（供 E 使用）

## 统一流程

仓库脚本 `scripts/experiment2_mfcc.py` 是运行入口。它沿用 B 的 `preprocess_b.preprocess_waveform`，不重新实现预加重、分帧、加窗或 FFT；之后按 D 的 Mel 三角滤波器组计算 26 维 Log-Mel，再用正交 DCT-II 计算 13 维 MFCC。C 的 `week3_c.py` 保留为 FFT/频谱原理与验证代码，不再作为第二条数据预处理路径，避免不同归一化造成结果不一致。

统一参数：16 kHz、预加重系数 0.97、25 ms 帧长、10 ms 帧移、Hamming 窗、512 点 FFT、单边功率谱 `abs(rfft)**2 / 512`。Mel 使用 2595·log10(1+f/700)，其反变换对应 700·(10^(mel/2595)-1)。能量取自然对数，MFCC 为正交归一化 DCT-II 的前 13 维。

## 运行

在仓库根目录执行：

```powershell
python scripts\experiment2_mfcc.py
```

默认遍历 `data/processed` 下的 WAV，输出到 `outputs/experiment2_mfcc`。只处理一条：

```powershell
python scripts\experiment2_mfcc.py --wav data\processed\spk01\digits\spk01_digit_0_take01.wav
```

每条录音生成 `power.npy`、`log_mel.npy`、`mfcc.npy`；运行参数与维度写入 `run_manifest.json`。批量运行需要仓库已有 WAV 数据；交接包不重复附带原始录音或大型数据集。

验证：

```powershell
python -m pytest -q tests\test_preprocess_b.py tests\test_experiment2_mfcc.py reports\experiment2_c_submission\test_week3_c.py
```

还需在周报中按实际小组分工填写贡献，不要把代码整合与验证结果误写成某位成员独立完成。
