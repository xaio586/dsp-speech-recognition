# 数字语音识别实验

本项目使用 Python 完成数字信号处理实验1至实验3，目标是实现从语音采集、预处理、时域分析、MFCC特征提取到DTW孤立数字识别的完整流程。

## 四周主线

- 第1周：建立仓库与Python环境，采集数据，完成WAV读取和波形显示。
- 第2周：完成分帧、加窗、短时能量、短时过零率和双门限端点检测。
- 第3周：完成FFT、Mel滤波器组和MFCC特征提取。
- 第4周：完成DTW数字识别、评测和课堂演示准备。

最终课堂演示需要实时麦克风输入。实验三实时界面已接入本地 MFCC+DTW 流程，离线数据评测与网页麦克风入口分别见 `scripts/evaluate_experiment3.py` 和 `scripts/run_experiment3.py`。

## 项目结构

```text
dsp-speech-recognition/
  data/
    raw/                 手机原始录音，只保留，不覆盖
    processed/           统一后的16 kHz单声道WAV
    metadata_template.csv
  outputs/
    figures/             波形、频谱、端点检测等图片
    metrics/             准确率、混淆矩阵等结果
  reports/weekly/        每周进度与验收清单
  scripts/               可直接运行的命令行脚本
  src/dsp_speech/        核心Python模块
  tests/                 自动测试
```

## 第一次配置

建议使用 Python 3.11 或 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts\verify_setup.py
```

## 第一周运行示例

将组长本机的采集清单放在 `data/raw/intake_manifest.csv` 后统一转换：

```powershell
python scripts\prepare_dataset.py data\raw\intake_manifest.csv
```

脚本会安全读取文件夹或ZIP，忽略 macOS 附带文件，将音频统一为16 kHz、单声道、PCM16 WAV，并生成 `data/processed/metadata.csv`。当前小组仓库保存了五位成员共115条原始手机录音和对应的处理后WAV，分别放在 `data/raw/` 与 `data/processed/`，可按清单核对。

检查单个WAV并生成波形：

```powershell
python scripts\inspect_wav.py data\raw\示例.wav
python scripts\analyze_sample.py data\processed\spk01\digits\spk01_digit_0_take01.wav
```

两个脚本会显示采样率、声道、时长、帧数和端点，并在 `outputs/figures/` 中保存波形及时域特征图。实验1使用 `src/dsp_speech/config.py` 中的同一组参数：16 kHz、25 ms帧长、10 ms帧移、Hamming窗。输出包括短时能量、短时平均幅度、过零率和双门限端点。

生成 C 同学负责的窗形状和单帧加窗效果对比图：

```powershell
python scripts\compare_windows.py data\processed\spk01\digits\spk01_digit_0_take01.wav
```

脚本会在 `outputs/figures/` 下保存两张图；默认挑能量最大的一帧，也可以传入 `--frame 20` 指定帧号。

运行测试：

```powershell
python -m pytest
```

## 数据采集

每位成员录制：

- 数字0至9，每个数字2遍，共20条。
- 自己的姓名3遍。
- 每人共23条，五人共115条，其中100条数字录音为核心数据。

手机生成WAV或M4A都可以。原始文件先放在 `data/raw/`，不要通过修改扩展名把M4A伪装成WAV。后续由数据负责人统一转换和命名。

第一周实收并通过批量可播放性检查：100条数字核心数据与15条姓名扩展数据，共115条，五位成员均达到每人23条的采集要求。

## 实验三实时识别

使用仓库根目录下的 Python 虚拟环境启动本地网页：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment3.py
```

打开 `http://127.0.0.1:8765` 并允许浏览器使用麦克风。说完一个0—9数字后点击“停止并识别”，程序会在本机执行重采样、端点检测、13维MFCC提取和DTW模板匹配。页面不提供音频文件上传入口。

用已有100条数字录音运行两折轮换评测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_experiment3.py
```

离线结果写入 `outputs/experiment3/`。网页演示和评测详细说明见 [docs/experiment3_live_app.md](docs/experiment3_live_app.md)。

## 实验四：实时说话人识别（ECAPA-TDNN 深度声纹嵌入）

说话人页面已接入麦克风识别，线上算法为**预训练 ECAPA-TDNN 声纹嵌入 + 余弦质心比对**（SpeechBrain `spkrec-ecapa-voxceleb`，ONNX 导出后用 onnxruntime 推理，无需 PyTorch）：录音经重采样、端点检测后提取 192 维声纹嵌入，与五位组员训练录音的均值嵌入（质心）计算余弦相似度，相似度最高者为识别结果；softmax(相似度×10) 作为置信度，最高置信度低于 0.55 时页面提示结果仅供参考。该方案不要求使用者说固定句子，识别结果不依赖所说内容和语言。它是五位已登记成员之间的闭集识别，不能保证把组外人员判为“未知”。

课程自研管线 **MFCC/LPC 统计特征 + RBF SVM**（102 维定长向量 + `StandardScaler + SVC(RBF)`，C=10、γ=scale，训练侧含声道增强）保留在 `speaker_recognition.py` 中，作为对照实现和 ONNX 模型文件缺失时的回退。

训练数据共 243 条录音：40 条句子录音（每人 4 句 × 2 遍）+ 203 条自由朗读切段（每人约 5 分钟自由朗读经能量语音活动检测切成 3–8 秒段，见 `scripts/ingest_free_speech.py`）；测试录音 27 条（7 条个人自由稿 + 20 条共同短句），另有一组课堂外录音（组长本人雅思口语 86 段，英语随机内容，仅作测试、不进训练集）。`metadata.csv` 标明姓名、编号和训练/测试划分。

运行 `python scripts/evaluate_speaker_ecapa.py` 复现部署版评估（纯 onnxruntime，结果写入 `outputs/experiment4_speaker_ecapa/`）；`python scripts/evaluate_speaker_svm.py` 复现 SVM 对照评估。实测对比：

| 测试组 | MFCC/LPC + RBF SVM | ECAPA-TDNN + 余弦质心 |
|---|---|---|
| 27 条留出测试·干净声道 | 26/27 | **27/27** |
| 27 条留出测试·蓝牙声道模拟 | 25/27 | **27/27** |
| 雅思录音 86 段（跨语言随机内容） | 65/82（另 4 段端点检测拒绝） | **86/86** |

SVM 超参只用训练集按原始录音分组交叉验证决定（增强副本同组）。结果仅代表本次录音和划分，现场麦克风表现可能不同。

## Git协作约定

- `main` 分支只保留可以运行的版本。
- 每项任务使用独立分支，例如 `feature/wav-reader`、`feature/endpoint-detection`。
- 提交信息说明做了什么，例如 `feat: add wav reader`。
- 合并前至少由一名非作者成员运行代码并检查结果。
- 原始音频默认不提交到普通Git历史中；先通过共享压缩包汇总，后续再决定是否使用Git LFS。

详细提交流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，模块之间的初始接口见 [docs/module_interfaces.md](docs/module_interfaces.md)。组员提交代码后，由成员A负责组织合并；集成阶段统一调整函数签名、数据形状、文件路径和测试。
