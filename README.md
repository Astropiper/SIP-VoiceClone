# 从长录音筛选并转写已登记的说话人

`speech_pipeline.py` 先用 FSMN-VAD 分段，再用 pyannote segmentation 3.0 判断每段有几个人说话。检测到两人轮流说话或同时说话时，整段跳过。剩余单人段用 CAM++ 提取声纹，与 JSON 声纹库比较，匹配成功才交给 SenseVoice 转写。推理在本机 CPU 上执行。

## 环境和模型

`hs` 已有 Python 依赖、FSMN-VAD、CAM++ 和 SenseVoice 权重。其他环境需要 Python 3.11、PATH 中的 FFmpeg，以及 `python -m pip install -r requirements.txt`。首次运行会从 ModelScope 获取本机缺少的 FunASR 权重。

在本目录下载 [sherpa-onnx 的 pyannote segmentation 3.0 ONNX 权重](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)，再运行转写命令。

```bash
mkdir -p models
curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 -o models/segmentation.tar.bz2
tar -xjf models/segmentation.tar.bz2 -C models
```

若权重已存放在其他位置，可用 `--segmentation-model /path/to/model.onnx` 指定。

## 登记声纹

每份参考录音只包含其 ID 对应的说话人。可多次指定同一 ID 来登记不同录音。程序对每段至少 1 秒的参考语音提取一个向量。

```bash
conda run -n hs python speech_pipeline.py enroll \
  --speaker speaker_01 reference_01.wav \
  --speaker speaker_02 reference_02.wav \
  --output voiceprints.json
```

已有声纹库时可跳过登记。JSON 顶层的 `model` 应为 `iic/speech_campplus_sv_zh-cn_16k-common`，`speakers` 对象的键为说话人 ID，值为该人的声纹向量列表。每个向量有 192 个数值，由同一 CAM++ 模型提取。

## 转写

```bash
conda run -n hs --no-capture-output python speech_pipeline.py transcribe 长录音.m4a \
  --voiceprints voiceprints.json \
  --output result.json
```

输出 JSON 的 `segments` 记录每个 VAD 片段的原录音起止时间（毫秒）、检测人数、重叠时长和处理状态。只有 `status` 为 `transcribed` 的条目含 `speaker_id` 和 `text`。其余状态包括 `multiple_speakers`、`no_speaker`、`too_short`、`no_match` 和 `ambiguous_match`。

- FSMN-VAD 每秒读取一块音频，连续语音约每 9 秒切段，使每个片段可以放入 pyannote 的 10 秒窗口。
- pyannote 中累计活动不足 120 毫秒的说话人不计入人数。单人段检测到的有效语音达到 1 秒才进行声纹匹配。
- 每位登记人的得分取其全部参考声纹中的最高余弦相似度。默认需达到 0.50，可用 `--match-threshold` 调整。前两名得分差小于 0.05 时跳过片段。
- 默认自动识别语言。可用 `--language zh` 指定中文。

## 单说话人 TTS 训练

首轮按编码器预训练、Mel 解码器训练与重合成、文本模型训练的顺序运行。两项下游任务都使用同一个冻结的 EMA 编码器检查点。见 [单说话人 TTS 设计记录](单说话人TTS设计记录.md)。

从项目根目录安装训练依赖。PyTorch 和 TorchAudio 使用匹配的版本。

在 NVIDIA 机器上，先在训练环境里安装匹配且支持 CUDA 的 PyTorch 和 TorchAudio，再安装项目依赖。[PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)

```bash
conda run -n hs python -m pip install -r tts/requirements.txt
conda run -n hs python -m tts --help
```

### 准备数据

各阶段读取通用 JSONL 清单，每行对应一个已切好的单说话人 WAV 或 FLAC 片段。`audio` 可以是绝对路径，也可以相对于清单所在目录。`source` 是切分前的原始录音 ID，同一原始录音的所有片段只能进入同一个集合。阶段二要求 `text`，其余阶段只需要音频和来源。

```json
{"audio":"clips/001.wav","source":"recording_01","text":"你好，欢迎收听。"}
```

带乐片段分别列入 `music-train.jsonl` 和 `music-val.jsonl`，干净片段分别列入 `clean-train.jsonl` 和 `clean-val.jsonl`。数据集按需解码、转单声道并重采样。Mel 数据集从同一片段生成 16 kHz 编码器输入和 24 kHz 频谱输入。真实语料接入时，可以让专用加载器直接构造 `tts.data.Clip`，交给现有 Dataset。

### 运行训练

三个训练命令默认加载项目中的 [模型配置](configs/model_config.json) 和 [训练配置](configs/training_config.json)。默认进行 1000 次优化器更新，每个微批次包含 4 条样本。下列示例在命令行传入清单路径和工作目录，也可以在训练配置里填写这些路径。

默认的 `--device auto` 优先选择 CUDA。NVIDIA 训练可用 `--device cuda` 强制使用 CUDA，或用 `--device cuda:0` 选定当前进程可见的第一张卡。音频解码、重采样和 Vocos 的 Mel 目标提取在 CPU 上完成，模型训练在选定的 GPU 上执行。

```bash
# 阶段一，带乐音频预训练
conda run -n hs --no-capture-output python -m tts pretrain \
  --train-manifest music-train.jsonl --valid-manifest music-val.jsonl \
  --output-dir runs/encoder

# 阶段三，干净音频 Mel 解码器
conda run -n hs --no-capture-output python -m tts train-mel \
  --encoder-checkpoint runs/encoder/best.pt \
  --train-manifest clean-train.jsonl --valid-manifest clean-val.jsonl \
  --output-dir runs/mel

# 在留出录音上比较预测 Mel 和真实 Mel 的重合成
conda run -n hs python -m tts reconstruct \
  --encoder-checkpoint runs/encoder/best.pt --mel-checkpoint runs/mel/best.pt \
  --audio heldout.wav --output runs/predicted.wav --reference-output runs/reference.wav

# 阶段二，使用带乐片段的 ASR 文本预测声学特征
conda run -n hs --no-capture-output python -m tts train-text \
  --encoder-checkpoint runs/encoder/best.pt \
  --train-manifest music-train.jsonl --valid-manifest music-val.jsonl \
  --output-dir runs/text
```

### 预训练伴奏增强

`pretrain` 可在训练时只给学生的 16 kHz 波形叠加独立伴奏；EMA 教师读取同一段原始录音，帧遮掩照常进行。验证使用原始录音，`train-text` 和 `train-mel` 沿用各自的原始监督目标。省略 `--background-manifest` 即运行无伴奏增强的基线。

伴奏清单沿用音频 JSONL 格式。选择无人声的音乐片段，其 `source` 必须独立于训练集和验证集。每行类似以下记录。

```json
{"audio":"accompaniment/001.wav","source":"instrumental_01"}
```

```bash
conda run -n hs --no-capture-output python -m tts pretrain \
  --train-manifest music-train.jsonl --valid-manifest music-val.jsonl \
  --background-manifest accompaniment.jsonl --background-probability 0.5 --background-snr-db 10 20 \
  --output-dir runs/encoder-background --updates 1000 --batch-size 4
```

提供清单后默认以 `0.5` 的概率增强每个训练片段，原始录音相对新增伴奏的信噪比默认从 `10–20 dB` 均匀采样，以重叠区的 RMS 计算。伴奏较长时随机裁取等长片段，较短时随机放在录音内部。混音保留输入长度，峰值超过 1 时对整条混合波形作等比衰减。训练指标 `background_mix_fraction` 记录实际混音的样本比例。

`training_config.json` 和检查点记录生效的概率、信噪比范围和伴奏清单的 SHA-256。续训时使用内容相同的伴奏清单。该实验测试对新增伴奏的鲁棒性，教师目标仍含录音原有的背景音乐。

### 指定模型和训练配置

修改项目的默认参数时，直接编辑 `configs/model_config.json` 和 `configs/training_config.json`。模型配置按训练命令分节。训练配置的 `common` 节适用于三个命令，同名阶段节覆盖公共值。用 `--model-config` 和 `--training-config` 可以分别指定实验配置。自定义 JSON 也可以是只包含当前阶段覆盖项的平铺对象，比如 `{"updates": 2000, "batch_size": 2}`。

新训练按默认文件、显式配置文件、命令行参数的顺序应用覆盖。训练配置使用下划线字段名，例如 `batch_size`、`lr`、`accumulate` 和 `device`。配置文件内的相对路径以该文件所在目录为基准，命令行里的相对路径以当前目录为基准。`train_manifest`、`valid_manifest` 和 `output_dir` 需提供；训练下游模型时还需提供 `encoder_checkpoint`。

| 训练命令 | 可调模型参数 |
| --- | --- |
| `pretrain` | `conv_channels`、`dim`、`layers`、`heads`、`ff_dim`、`teacher_layers`、`dropout` |
| `train-mel` | `width`、`layers` |
| `train-text` | `dim`、`encoder_layers`、`decoder_layers`、`heads`、`ff_dim`、`dropout`、`stop_pos_weight` |

可以用仓库里的小型编码器配置开始一次训练。[configs/encoder-small.json](configs/encoder-small.json)

```bash
conda run -n hs --no-capture-output python -m tts pretrain \
  --model-config configs/encoder-small.json \
  --train-manifest music-train.jsonl --valid-manifest music-val.jsonl \
  --output-dir runs/encoder-small --updates 1000 --batch-size 4
```

`dim` 必须能被 `heads` 整除，`teacher_layers` 必须在 1 到 `layers` 之间。下游模型的 `acoustic_dim` 由冻结编码器确定，文本模型的 `vocab_size` 由训练集词表确定。

### 归档与复用

训练初始化后，工作目录的 `input_configs/` 保存本次读取的两份 JSON 原件。工作目录根级的 `model_config.json` 和 `training_config.json` 保存实际生效的值，包括命令行覆盖、解析后的路径和派生字段。训练配置还会记录实际设备、清单 SHA-256 和续训检查点路径。可将根级的两份文件传给同一训练命令并指定新的 `--output-dir` 来复现实验。

使用 `--resume` 时，模型结构和检查点记录的训练设置优先于项目默认配置。显式传入的模型参数必须与检查点一致，影响优化器进度和训练目标的参数也须保持一致。可以通过 `--training-config runs/encoder/training_config.json` 复用原运行的数据路径和运行选项，再用 `--resume runs/encoder/last.pt` 指定断点。续训成功初始化后会更新根级的配置；省略模型配置时会沿用已有的模型配置原件，若工作目录里没有原件则根据检查点写入模型配置。

### 检查点与评估

每次验证都会保存 `last.pt`。预训练要求学生和教师在等长验证音频上的同位置特征距离均达到 `--min-content-distance`（默认 `0.1`），才按遮掩帧验证 MSE 更新 `best.pt`。验证日志的 `content_check_passed` 表示是否通过此检查。日志也记录遮掩比例、平均跨度、教师目标和学生预测的标准差，以及导出表示统计。选择验证片段时应覆盖不同的语音内容。

新预训练默认用 `--variance-weight 0.1` 加权位置编码前的学生特征方差损失，设为 `0` 可做原损失的对照。用原训练参数和清单再次执行命令，并加上 `--resume runs/encoder/last.pt`，可恢复训练进度和学习率调度。旧断点续训沿用原损失。恢复时会按当前阈值复检旧版 `best.pt`，未通过的文件保留为 `rejected-best-*.pt`。

文本模型和 Mel 解码器在验证损失改善时更新 `best.pt`。文本模型以带调拼音作为中文 token，英文按字母编码并保留词间空格。验证会记录自由生成的长度误差和触顶次数。

`train-mel`、`reconstruct` 和 `synthesize` 首次使用时会下载权重。Mel 训练直接使用该检查点自带的特征提取器。[Vocos 权重](https://huggingface.co/charactr/vocos-mel-24khz)

可以将真实带乐留出片段传给 `reconstruct`，听评发音和伴奏残留。`extract` 会导出冻结编码器的帧级特征、帧步长和首帧中心偏移。

```bash
conda run -n hs python -m tts synthesize \
  --text-checkpoint runs/text/best.pt --mel-checkpoint runs/mel/best.pt \
  --text '你好，欢迎收听。' --output runs/tts.wav
```

### 查看训练和推理图表

三个训练命令都会在 `--output-dir` 中实时追加 `train.log`，例如 `runs/encoder/train.log`。每次启动或续训都会写入时间和训练参数，训练期间的终端输出和 Python 异常堆栈也会写入该文件。排查训练问题时，可以把这个文件交给 AI 分析。

`metrics.jsonl` 保存用于绘图的结构化指标。`training.png` 随每次验证及训练结束刷新，验证曲线用菱形标出保存 `best.pt` 的更新步。续训从检查点旁的 `metrics.jsonl` 恢复指标历史，并截去超出断点的记录。加 `--no-plot` 可以跳过绘图并保留两个日志文件。

| 阶段 | `training.png` 的主要指标 |
| --- | --- |
| 编码器预训练 | 遮掩回归 MSE、方差损失、师生同位置距离及选模阈值、有效秩、目标与预测的标准差、遮掩统计和学习率 |
| 文本模型 | 训练与验证损失、声学 L1、停止预测损失、自由生成的长度误差及触及帧数上限的比例 |
| Mel 解码器 | 训练与验证 Mel L1、学习率 |

也可以从保存的 JSONL 或通过 `tee` 保存的控制台日志重新绘图。如果旧的阶段一日志没有记录阈值，可加 `--min-content-distance` 指定当时使用的值。

```bash
conda run -n hs python -m tts plot-training --stage pretrain \
  --metrics runs/encoder/metrics.jsonl --output runs/encoder/training.png
```

推理图默认保存在输出文件旁，和输出文件同名并使用 `.png` 扩展名。`extract` 绘制帧级特征热图。`reconstruct` 绘制共用色标的参考和重合成 log-Mel，以及绝对误差。`synthesize` 绘制生成的 log-Mel 和实际输出波形。三个命令均支持 `--plot-output` 指定图路径，或用 `--no-plot` 跳过绘图。
