# 说话人识别数据

线上服务使用**预训练 ECAPA-TDNN 声纹嵌入 + 余弦质心**：`ecapa_encoder.onnx` 与 `ecapa_encoder.onnx.data` 是 SpeechBrain `spkrec-ecapa-voxceleb` 编码器的 ONNX 导出（fp32，权重为外部数据文件，两者须放在一起）；`ecapa_centroids.npz` 是五位组员的注册声纹——各自全部训练录音的 192 维嵌入均值经 L2 归一化，由 `scripts/build_ecapa_centroids.py` 生成（纯 onnxruntime，不需要 PyTorch）。识别时对输入语音提取嵌入，与五个质心做余弦相似度，取最大者。

课程自研的“定长特征向量 + RBF SVM”保留为回退路径：`svm_features.npz`（243 条训练录音经声道增强后的 922 条 102 维向量，由 `scripts/build_svm_features.py` 从训练 WAV 生成），当 ONNX 模型文件缺失时服务启动自动改用它拟合 `StandardScaler + SVC(RBF)`。更早的 `templates.npz`（40 条训练 MFCC 序列）保留作为无 LPC 特征时的最后回退来源。完整原始 M4A 和统一 WAV 数据包在项目负责人本地单独保存，不进代码仓库。

`metadata.csv` 标出姓名、说话人编号和训练/测试用途；完整评测需要把统一数据包展开到本目录，以提供 `wav/` 文件。当前数据量为 243 条训练录音（40 条句子录音 + 203 条自由朗读切段）、27 条测试录音（7 条个人自由稿 + 20 条共同短句）。WAV 参数为 16 kHz、单声道、PCM16。
