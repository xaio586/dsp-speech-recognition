# 小组代码协作说明

## 基本原则

- 每个人的代码都上传到同一个Git仓库。
- 不直接在 `main` 分支开发，也不要直接覆盖其他成员文件。
- 每项任务建立独立功能分支，完成后发起合并请求。
- 合并前至少运行自己的脚本，并提供一个示例输入和输出。
- 实验数据和程序代码分开保存，代码中不能写死某位成员电脑的绝对路径。

## 每位成员的操作流程

第一次下载项目：

```powershell
git clone <远程仓库地址>
cd dsp-speech-recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

开始一项任务前：

```powershell
git switch main
git pull
git switch -c feature/任务名称
```

分支名称示例：

```text
feature/data-check
feature/framing-window
feature/endpoint-detection
feature/fft
feature/mfcc
feature/dtw
```

完成代码后：

```powershell
git status
git add src tests scripts
git commit -m "feat: 完成具体功能"
git push -u origin feature/任务名称
```

随后在远程仓库发起合并请求，由成员A或指定审查人检查后合并。

## 提交给组长时必须说明

- 负责的模块和入口函数。
- 输入数据类型、形状和单位。
- 返回值类型、形状和含义。
- 依赖的第三方库。
- 如何运行，以及一份真实或合成的示例输出。
- 当前仍存在的问题或参数限制。

## 代码要求

- 核心算法放在 `src/dsp_speech/`，命令行入口放在 `scripts/`。
- 核心算法函数不直接读取固定文件，也不写死 `C:\...` 等绝对路径。
- 核心函数尽量接收 NumPy 数组并返回 NumPy 数组或明确的数据类。
- 绘图与算法分开，算法函数不能以“看图”作为唯一输出。
- 新模块至少提供一个自动测试，放在 `tests/`。
- 所有公开函数写明参数、返回值和数组形状。

## 集成与接口调整

成员可以先在自己的分支完成算法。合并阶段由成员A统一组织集成，主要处理：

- 统一函数名和参数顺序。
- 统一采样率、声道、数据类型和数组形状。
- 去除写死路径和重复代码。
- 增加适配层，避免大幅重写组员已经完成的算法。
- 补充自动测试，检查各模块能否串联运行。
- 解决合并冲突并更新README中的运行方法。

需要自动调整时，把组员分支同步到当前仓库，或将代码文件放入对应目录，再说明各文件由谁编写、原本怎么运行。随后可统一完成接口适配和集成测试。

