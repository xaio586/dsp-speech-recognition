# 固定划分 MFCC 参数评估

运行命令（在仓库目录）：`python scripts/evaluate.py --workers 4`。

训练：fold1_role=train 且 take=1；测试：fold1_role=test 且 take=2。
训练 50 条，测试 50 条；所有配置使用同一划分。
预处理复用 analyze_time_domain，截取检测端点后提取 MFCC；没有随机划分或跳过录音。
每个数字保留全部训练模板，识别复用 recognize_digit。

| 配置 | n_mfcc | n_filters | n_fft | 正确/测试 | 准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 13 | 26 | 512 | 42/50 | 84.00% |
| mfcc_20 | 20 | 26 | 512 | 43/50 | 86.00% |
| mel_40 | 13 | 40 | 512 | 40/50 | 80.00% |
| fft_1024 | 13 | 26 | 1024 | 42/50 | 84.00% |

predictions.csv 保存逐条真实标签、预测标签、正确标记和 DTW 距离；
summary.csv 保存参数及总体准确率（0–1）。
confusion_matrix_*.csv 的行为真实标签、列为预测标签。
split_manifest.csv 保存录音划分、音频校验值及端点。
run_manifest.json 保存环境、共享参数和源码校验值。

这是同一批说话人第 1 遍训练、第 2 遍测试的已见说话人实验。
参数优劣仅描述这份固定测试集；未评估未见说话人，也没有独立验证集或第二折实验。
