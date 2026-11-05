# 模块接口约定

这是一版初始约定，用于减少五个人独立开发时的接口冲突。实现细节可以不同，但输入输出需要能够转换成这里的形式。

## 通用音频表示

- 采样率使用整数，单位为Hz。
- 核心算法中的语音数据使用一维 `numpy.ndarray`。
- 推荐数据类型为 `float32`，幅值约在 `[-1, 1]`。
- 所有算法默认接收单声道语音。
- 原始音频统一转换到16 kHz后再进入实验主流程。

仓库已经提供：

```python
from dsp_speech.io_audio import AudioData, load_wav
```

## 预处理模块

建议文件：`src/dsp_speech/preprocess.py`

```python
def remove_dc(samples: np.ndarray) -> np.ndarray: ...

def normalize_amplitude(samples: np.ndarray) -> np.ndarray: ...

def pre_emphasis(samples: np.ndarray, coefficient: float = 0.97) -> np.ndarray: ...
```

输入和输出均为一维 `float32` 数组，长度保持不变。

## 统一参数

当前实验共用 `src/dsp_speech/config.py` 中的 `DEFAULT_CONFIG`：16 kHz采样率、25 ms帧长、10 ms帧移、Hamming窗和0.97预加重。实验1时域分析、实验2 MFCC和绘图脚本都读取这份配置，避免各自写一套默认值。

## 分帧与加窗模块

文件：`src/dsp_speech/preprocess.py`

```python
def frame_signal(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray: ...

def apply_window(frames: np.ndarray, window: str = "hamming") -> np.ndarray: ...
```

帧数组统一使用形状 `(帧数, 每帧采样点数)`。末尾不足一帧时补零，使最后一段语音也参与分析；16 kHz、1秒、25/10 ms配置下输出99帧。窗类型使用 `hamming`、`hann` 或 `rectangular`，同时兼容组员代码中的 `hanning` 和 `rect` 名称。

## 时域特征与端点检测

建议文件：`src/dsp_speech/endpoint.py`

```python
def short_time_energy(frames: np.ndarray) -> np.ndarray: ...

def zero_crossing_rate(frames: np.ndarray) -> np.ndarray: ...

def detect_endpoint(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[int, int]: ...
```

端点检测返回的是原始采样点下标 `(start_sample, end_sample)`，使用左闭右开区间。
当前由 `analyze_time_domain` 一次性输出去直流归一化后的波形、分帧、加窗、短时能量、短时幅度、过零率及端点，所有相关参数从 `DEFAULT_CONFIG` 读取。

## FFT和MFCC

建议文件：`src/dsp_speech/features.py`

```python
def magnitude_spectrum(frames: np.ndarray, n_fft: int = 512) -> np.ndarray: ...

def extract_mfcc(
    samples: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
) -> np.ndarray: ...
```

MFCC统一返回形状 `(帧数, MFCC维数)`，不要返回转置后的矩阵。

## DTW与数字识别

建议文件：`src/dsp_speech/dtw.py`

```python
def dtw_distance(query: np.ndarray, template: np.ndarray) -> float: ...
```

输入矩阵形状均为 `(帧数, 特征维数)`，返回归一化后的浮点距离。

建议文件：`src/dsp_speech/recognize.py`

```python
def recognize_digit(
    features: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> tuple[str, float]: ...
```

返回 `(预测数字, 最小距离)`，预测数字暂时使用字符串 `"0"` 至 `"9"`。

## 集成检查

最终主流程保持以下顺序：

```text
读取或录制语音
-> 统一16 kHz单声道float32
-> 预处理
-> 端点检测
-> 截取有效语音
-> MFCC
-> DTW匹配
-> 输出数字和距离
```

组员实现与本约定不同也可以保留，通过适配函数转换；不要为了接口统一随意删除已验证过的算法代码。

仓库保留模块化实现：`preprocess.py`、`features.py` 与 `dtw.py`。成员C的分帧/加窗实现和成员D的能量、幅度、过零率、双门限端点思路已收敛到 `preprocess.py`；调用脚本统一使用 `DEFAULT_CONFIG`。
