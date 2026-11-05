# 实验2预处理（B）

运行：

```powershell
python scripts\preprocess_b.py
```

脚本默认从仓库内的 `data/raw/` 递归读取 M4A，解码为16 kHz单声道后，使用项目统一参数执行预加重、25 ms分帧、10 ms帧移和Hamming加窗，再计算512点单边功率谱。输出默认放在 `outputs/week03_b_preprocess/`，保留原有说话人和数字/姓名目录结构。

每条 NPY 保存的是二维功率谱数组，不是可以播放的音频。C 的 STFT 代码使用相同的采样率、帧长、帧移、Hamming窗和FFT点数；两者的频率维都是257点。`log.json` 记录逐条检查结果，`run_manifest.json` 记录本次输入数量、公共参数和输出类型。短时长或近静音会跳过输出；解码后超过满幅的峰值会标为提醒，但仍保存结果，因为有损编码的解码峰值超过1不一定表示录音时发生削波。NPY 是可重复生成的中间数据，Git 忽略这些数组；代码、日志和清单可以共享。需要调整输入或输出目录时可运行：

```powershell
python scripts\preprocess_b.py --input data\raw --output outputs\week03_b_preprocess
```
