# 实验三手机端实时数字识别部署包

本包只包含实时识别程序、网页界面、数字 WAV 模板和必要依赖，不包含原始录音压缩包、姓名语音、论文或周报。

## Render 部署

1. 将本目录单独推送到一个 GitHub 仓库。
2. 在 Render 选择 New → Blueprint，连接该仓库并使用 `render.yaml`。
3. 部署完成后访问 Render 提供的 HTTPS 地址，手机浏览器允许麦克风权限即可使用。

服务启动时会自动从 `data/processed/metadata.csv` 载入数字模板；网页端通过浏览器采集麦克风音频，再提交到 Python 识别接口。

## 本地启动

```bash
pip install -r requirements-hosted.txt
python scripts/run_experiment3.py
```

