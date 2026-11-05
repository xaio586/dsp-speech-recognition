"""Generate Chinese figures for the first-week report."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.config import DEFAULT_CONFIG  # noqa: E402
from dsp_speech.io_audio import load_wav  # noqa: E402
from dsp_speech.preprocess import analyze_time_domain  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT.parent / "output" / "week01_images"
BLUE = "#1F4E78"
LIGHT_BLUE = "#5B9BD5"
ORANGE = "#ED7D31"
GREEN = "#70AD47"
GRAY = "#666666"


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def data_overview() -> Path:
    summary = pd.read_csv(PROJECT_ROOT / "data" / "dataset_summary.csv")
    summary = summary[summary["speaker_id"] != "TOTAL"].copy()
    labels = ["成员A", "成员B", "成员C", "成员D", "成员E"]
    x = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(12, 6.75))
    axis.bar(x, summary["digit_count"], width=0.58, color=BLUE, label="数字录音")
    axis.bar(
        x,
        summary["name_count"],
        width=0.58,
        bottom=summary["digit_count"],
        color=ORANGE,
        label="姓名录音",
    )
    for index, row in summary.reset_index(drop=True).iterrows():
        axis.text(
            index,
            row.total_count + 0.35,
            str(int(row.total_count)),
            ha="center",
            fontsize=14,
        )
    axis.axhline(23, color=GREEN, linestyle="--", linewidth=1.5, label="计划每人23条")
    axis.set_title("第一周语音数据采集与验收结果", fontsize=24, fontweight="bold", pad=18)
    axis.text(
        0.5,
        1.01,
        "100条数字完整｜15条姓名完整｜共115条｜总时长487.557秒",
        transform=axis.transAxes,
        ha="center",
        fontsize=14,
        color=GRAY,
    )
    axis.set_xticks(x, labels, fontsize=13)
    axis.set_ylabel("录音数量（条）", fontsize=13)
    axis.set_ylim(0, 26)
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False, ncols=3, fontsize=12)
    figure.tight_layout()
    destination = OUTPUT_DIR / "01_数据采集统计.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def signal_analysis() -> Path:
    wav = PROJECT_ROOT / "data" / "processed" / "spk01" / "digits" / "spk01_digit_0_take01.wav"
    audio = load_wav(wav, target_sample_rate=DEFAULT_CONFIG.sample_rate)
    result = analyze_time_domain(audio.samples, audio.sample_rate)
    samples = result["samples"]
    energy = result["energy"]
    magnitude = result["magnitude"]
    zcr = result["zcr"]
    start, end = result["endpoints"]
    time = np.arange(len(samples)) / audio.sample_rate
    frame_time = np.arange(len(energy)) * DEFAULT_CONFIG.hop_ms / 1000

    figure, axes = plt.subplots(4, 1, figsize=(12, 8.2), sharex=True)
    axes[0].plot(time, samples, color=BLUE, linewidth=0.75)
    axes[0].axvspan(start / audio.sample_rate, end / audio.sample_rate, color=ORANGE, alpha=0.24)
    axes[0].set_ylabel("幅值")
    axes[0].set_title("真实录音样例：数字“0”的波形与短时特征", fontsize=22, fontweight="bold")
    axes[0].text(
        0.99,
        0.90,
        f"16 kHz｜单声道｜端点 {start / audio.sample_rate:.2f}–{end / audio.sample_rate:.2f} s",
        transform=axes[0].transAxes,
        ha="right",
        color=GRAY,
        fontsize=12,
    )
    axes[1].plot(frame_time, energy, color=GREEN, linewidth=1.2)
    axes[1].set_ylabel("短时能量")
    axes[2].plot(frame_time, magnitude, color=LIGHT_BLUE, linewidth=1.2)
    axes[2].set_ylabel("短时幅度")
    axes[3].plot(frame_time, zcr, color=ORANGE, linewidth=1.2)
    axes[3].set_ylabel("过零率")
    axes[3].set_xlabel("时间（秒）")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    destination = OUTPUT_DIR / "02_真实录音时域分析.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def workflow() -> Path:
    figure, axis = plt.subplots(figsize=(12, 6.75))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")
    axis.set_title("第一周数据处理流程", fontsize=25, fontweight="bold", pad=18)
    axis.text(
        6,
        6.35,
        "手机录音保留原文件，代码自动完成转换、质检和实验输入准备",
        ha="center",
        fontsize=14,
        color=GRAY,
    )

    items = [
        ("手机录音", "5人提交M4A/ZIP", BLUE),
        ("安全汇总", "过滤隐藏文件\n识别数字/姓名/遍次", LIGHT_BLUE),
        ("统一转换", "16 kHz｜单声道\nPCM16 WAV", GREEN),
        ("质量检查", "数量｜时长｜音量\n可播放性｜重复编号", ORANGE),
        ("实验输入", "波形｜分帧｜能量/ZCR\nMFCC｜DTW", "#8064A2"),
    ]
    centers = np.linspace(1.2, 10.8, len(items))
    for index, ((title, detail, color), center) in enumerate(zip(items, centers, strict=True)):
        box = FancyBboxPatch(
            (center - 0.95, 2.45),
            1.9,
            2.25,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white",
        )
        axis.add_patch(box)
        axis.add_patch(
            FancyBboxPatch(
                (center - 0.95, 4.05),
                1.9,
                0.65,
                boxstyle="round,pad=0.03,rounding_size=0.15",
                linewidth=0,
                facecolor=color,
            )
        )
        axis.text(
            center,
            4.37,
            title,
            ha="center",
            va="center",
            fontsize=15,
            color="white",
            fontweight="bold",
        )
        axis.text(center, 3.25, detail, ha="center", va="center", fontsize=12.5, linespacing=1.45)
        if index < len(items) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (center + 0.98, 3.55),
                    (centers[index + 1] - 0.98, 3.55),
                    arrowstyle="-|>",
                    mutation_scale=17,
                    linewidth=1.6,
                    color="#7F7F7F",
                )
            )

    axis.text(
        6,
        1.35,
        "验收结果：100条数字＋15条姓名全部完整，115条音频成功转换并建立清单",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=BLUE,
    )
    axis.text(
        6,
        0.75,
        "隐私约定：原始录音、处理后WAV和真实姓名清单不上传公开仓库",
        ha="center",
        fontsize=13,
        color="#C00000",
    )
    figure.tight_layout()
    destination = OUTPUT_DIR / "03_数据处理流程.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (data_overview(), signal_analysis(), workflow()):
        print(path)


if __name__ == "__main__":
    main()
