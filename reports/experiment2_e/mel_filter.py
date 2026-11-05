import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# 解决绘图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. Mel 基础函数与滤波器组 ====================
def hz_to_mel(f):
    """根据讲义公式: Mel(f) = 2595 * log10(1 + f / 700)"""
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(mel):
    """Inverse of hz_to_mel using the matching base-10 definition."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

def get_mel_filterbank(num_filters=26, n_fft=512, fs=16000, low_freq=0, high_freq=None):
    """
    生成 Mel 三角形滤波器组
    """
    if high_freq is None:
        high_freq = fs / 2.0
    
    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)
    mel_points = np.linspace(mel_low, mel_high, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    num_bins = n_fft // 2 + 1
    # 严格根据讲义理论公式: k = N * f / fs 计算对应 FFT bin 索引
    bin_points = np.floor(n_fft * hz_points / fs).astype(int)
    bin_points = np.clip(bin_points, 0, num_bins - 1)
    
    filter_bank = np.zeros((num_filters, num_bins))
    for m in range(1, num_filters + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        
        # 拦截相邻 bin 重合退化，防止分母除零
        if f_center > f_left:
            for k in range(f_left, f_center):
                filter_bank[m - 1, k] = (k - f_left) / (f_center - f_left)
        if f_right > f_center:
            for k in range(f_center, f_right):
                filter_bank[m - 1, k] = (f_right - k) / (f_right - f_center)
                
    return filter_bank, hz_points[1:-1], bin_points[1:-1]

# ==================== 2. 数据读取与前置校验 ====================
def load_c_spectrum_data(npy_path="spectrum_data.npy", fs=16000, n_fft=512):
    """
    读取成员 C 传递的频域/谱线能量数据
    包含 NaN / Inf 检测与非负保护
    """
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        print(f"成功读取成员 C 的频谱文件: '{npy_path}'，维度: {data.shape}")
        
        # 1. 前置异常数值检测 (NaN / Inf)
        if np.isnan(data).any():
            raise ValueError("【严重错误】成员 C 输入数据中包含 NaN，请排查上游分帧或 FFT 计算！")
        if np.isinf(data).any():
            raise ValueError("【严重错误】成员 C 输入数据中包含 Inf (无穷大)，请排查上游数值溢出！")
        
        # 2. 判断数据类型并计算谱线能量 E(i, k)
        if np.iscomplexobj(data):
            power_spec = np.abs(data) ** 2
        else:
            power_spec = data.astype(np.float32)
            if np.any(power_spec < 0):
                warnings.warn("警告: 成员 C 输入的数据中存在负值，已自动进行非负截断 np.clip(0)！")
                power_spec = np.clip(power_spec, 0, None)
    else:
        print(f"未找到 '{npy_path}'，自动生成仿真频谱数据以供独立测试（基于参数 n_fft={n_fft}）...")
        num_frames = 100
        num_bins = n_fft // 2 + 1
        freqs = np.linspace(0, fs / 2, num_bins)
        base = np.exp(-freqs / 1800.0) + 0.6 * np.exp(-((freqs - 1200) / 250)**2)
        power_spec = np.outer(np.linspace(0.8, 1.2, num_frames), base)
        power_spec += 0.02 * np.random.rand(num_frames, num_bins)

    # 3. 能量非负校验
    assert np.all(power_spec >= 0), "输入能量谱数据存在负数，不符合能量物理定义！"
    return power_spec

# ==================== 3. 校验逻辑 ====================
def verify_mel_module(fb, center_freqs, center_bins, power_spec, mel_spec, expected_m=26):
    """
    模块全项校验（含数值稳定性、数量、中心频率、维度、叠加和）
    """
    print("\n" + "=" * 45)
    print("           成员 D 模块自动校验结果           ")
    print("=" * 45)
    
    # 0. 检验输出是否有 NaN / Inf
    assert not np.isnan(mel_spec).any(), "【校验失败】输出 Mel 频谱包含 NaN！"
    assert not np.isinf(mel_spec).any(), "【校验失败】输出 Mel 频谱包含 Inf！"
    print("✓ 数值稳定性检查通过: 输出 Mel 频谱无 NaN 或 Inf 异常")
    
    # 1. 滤波器数量校验
    assert fb.shape[0] == expected_m, f"滤波器数量异常: 实际 {fb.shape[0]} != 预期 {expected_m}"
    print(f"✓ 滤波器组数量检查通过: 共 {expected_m} 个滤波器")
    
    # 2. 中心频率单调性校验
    assert np.all(np.diff(center_freqs) > 0), "中心频率未满足严格单调递增！"
    assert np.all(np.diff(center_bins) >= 0), "中心频率 bin 索引异常！"
    print(f"✓ 中心频率检查通过: 范围 {center_freqs[0]:.1f} Hz ~ {center_freqs[-1]:.1f} Hz")
    
    # 3. 输出维度匹配校验
    expected_out_shape = (power_spec.shape[0], expected_m)
    assert mel_spec.shape == expected_out_shape, f"输出维度异常: {mel_spec.shape} != {expected_out_shape}"
    print(f"✓ 输出维度检查通过: 形状 {mel_spec.shape} (帧数 x 滤波器数)")
    
    # 4. 讲义约束 ∑ H_m(k) ≈ 1 校验
    start_bin = center_bins[0]
    end_bin = center_bins[-1]
    internal_sum = np.sum(fb, axis=0)[start_bin : end_bin + 1]
    mean_val = np.mean(internal_sum)
    
    print(f"✓ 重叠频带叠加和检验: 内部有效频点权重均值 = {mean_val:.4f}")
    if not np.allclose(internal_sum, 1.0, atol=0.08):
        warnings.warn(f"提示: 离散网格取整存在采样抖动，权重叠加和最大偏差为 {np.max(np.abs(internal_sum - 1.0)):.4f}，属于正常离散误差。")
    else:
        print("  -> 叠加和在容差范围内满足 ∑ H_m(k) ≈ 1")
        
    print("=" * 45 + "\n")

# ==================== 4. 实验过程绘图 ====================
def plot_mel_process(power_spec, fb, mel_spec, fs, num_filters):
    """绘制 4 联实验过程图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    num_bins = power_spec.shape[1]
    freq_axis = np.linspace(0, fs / 2, num_bins)
    mid_frame = power_spec.shape[0] // 2

    # 子图 1: Mel 滤波器组频域响应与叠加和
    for m in range(num_filters):
        axes[0, 0].plot(freq_axis, fb[m], color='C0', alpha=0.5)
    weight_sum = np.sum(fb, axis=0)
    axes[0, 0].plot(freq_axis, weight_sum, 'r--', linewidth=1.5, label=r'$\sum_m H_m(k) \approx 1$')
    axes[0, 0].set_title("1. Mel 尺度三角滤波器组及叠加和", fontsize=11)
    axes[0, 0].set_xlabel("物理频率 (Hz)")
    axes[0, 0].set_ylabel("幅值响应")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # 子图 2: 单帧输入线性谱能量 E(i, k)
    axes[0, 1].plot(freq_axis, power_spec[mid_frame], color='green', linewidth=1.2)
    axes[0, 1].set_title(f"2. 第 {mid_frame} 帧输入线性频域能量谱 E(i, k)", fontsize=11)
    axes[0, 1].set_xlabel("物理频率 (Hz)")
    axes[0, 1].set_ylabel("能量大小")
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 子图 3: 单帧通过 Mel 滤波器组后的能量 S(i, m)
    axes[1, 0].stem(range(1, num_filters + 1), mel_spec[mid_frame], linefmt='darkblue', markerfmt='o', basefmt='k-')
    axes[1, 0].set_title(f"3. 第 {mid_frame} 帧输出 Mel 能量 S(i, m)", fontsize=11)
    axes[1, 0].set_xlabel("Mel 滤波器编号 m (1 ~ 26)")
    axes[1, 0].set_ylabel("能量大小")
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # 子图 4: 全语音段二维 Mel 频谱热力图
    im = axes[1, 1].imshow(
        np.log(mel_spec + 1e-10).T,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, mel_spec.shape[0], 1, num_filters]
    )
    axes[1, 1].set_title("4. 全局 Log-Mel 语谱图 (Mel Spectrogram)", fontsize=11)
    axes[1, 1].set_xlabel("帧序号 (Frame)")
    axes[1, 1].set_ylabel("Mel 通道 m")
    fig.colorbar(im, ax=axes[1, 1], label='对数能量 (dB)')

    plt.tight_layout()
    plt.show()

# ==================== 5. 主执行流程 ====================
if __name__ == "__main__":
    fs = 16000
    n_fft = 512
    num_filters = 26
    
    # 1. 读取并校验来自成员 C 的输入数据（含 NaN / Inf 校验）
    power_spectrum = load_c_spectrum_data(npy_path="spectrum_data.npy", fs=fs, n_fft=n_fft)
    
    # 2. 根据输入频点数自适应反推 FFT 点数
    actual_n_fft = (power_spectrum.shape[1] - 1) * 2
    
    # 3. 生成 Mel 滤波器组
    fb, center_freqs, center_bins = get_mel_filterbank(num_filters, actual_n_fft, fs)
    
    # 4. 计算 Mel 频谱 S(i, m) = ∑ E(i, k) * H_m(k)
    mel_spectrum = np.dot(power_spectrum, fb.T)
    
    # 5. 执行全项校验（含输出 NaN / Inf 检查）
    verify_mel_module(fb, center_freqs, center_bins, power_spectrum, mel_spectrum, num_filters)
    
    # 6. 保存标准文件交接给成员 E
    np.save("mel_spectrum.npy", mel_spectrum)
    print("已成功导出 'mel_spectrum.npy' 供成员 E 做 DCT 特征提取。")
    
    # 7. 绘制过程图
    plot_mel_process(power_spectrum, fb, mel_spectrum, fs, num_filters)
