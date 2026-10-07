# Disaster Tweets 实验工程

Kaggle 比赛 [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) 的完整实验记录与可复现脚本。

- **任务**：二分类（推文是否描述真实灾难），**指标 F1**
- **数据**：train 7613 / test 3263
- **硬件**：RTX 3050 Laptop (4GB)，本地训练

## 当前成绩

| 实验 | 模型 | CV F1 | public LB |
|---|---|---|---|
| **blend_v1** | BERT-base + TF-IDF 融合 | 0.8089 | **0.83849**（rank 70/442） |
| bert_base_v1 | bert-base-uncased | 0.8085 | — |
| bert_base_v2 | bert-base-uncased（5 epoch） | 0.8071 | — |
| tfidf_lr_v1 | TF-IDF + LogisticRegression | 0.7699 | — |

> 剔除榜单上 7 支疑似使用 test 标签泄漏（≥0.99）的队伍后，名次为 **63/435**，与合法第一（0.85412）差 **0.0156**。
> 完整记录见 `实验记录.md` 与 `disaster-tweets-experiments-summary.ipynb`。

## 目录结构

```
.
├── exp_log.py                     # 实验记录工具：追加 / 查看 / 生成 markdown / 写入 LB
├── experiments.csv                # 实验记录表（唯一事实来源，utf-8-sig 可直接用 Excel 打开）
├── 实验记录.md                    # 实验记录的 markdown 版（由 exp_log.py md 生成）
│
├── tfidf_lr_baseline.py           # 基线：TF-IDF(word1-2 + char_wb3-5) + LogisticRegression
├── finetune_transformer.py        # 微调预训练模型（DeBERTa/BERT/RoBERTa，5 折 CV + 早停 + 折平均）
├── blend.py                       # 模型融合：OOF 上搜权重与阈值，生成提交
│
├── submit_to_kaggle.py            # 提交（带上传目标可达性预检）
├── submit_via_proxy.py            # 经 Windows 侧代理提交（googleapis 被墙时的方案）
│
├── disaster-tweets-experiments-summary.ipynb   # 全过程记录（已推送到 Kaggle）
├── kaggle_push/                   # 推送 Kaggle kernel 用的副本与 kernel-metadata.json
│
├── oof/                           # 各实验的 OOF 与测试概率（融合、阈值搜索都依赖它）
├── submissions/                   # 每次实验产出的提交文件
├── logs/                          # 训练日志
└── models/                        # 预训练权重缓存（4.7G，已被 .gitignore 排除）
```

## 快速开始

```bash
# 环境：conda py310（torch 2.11.0+cu128, transformers, scikit-learn, modelscope）
conda activate py310

# 1) 基线
python tfidf_lr_baseline.py --log --name tfidf_lr_v1

# 2) 微调 BERT（首次会自动从 ModelScope 下载权重到 models/）
python finetune_transformer.py --model google-bert/bert-base-uncased \
    --folds 5 --epochs 3 --bs 16 --name bert_base_v1 --log

# 3) 冲分：DeBERTa-v3-base（4GB 显存用 bs8 + 梯度累积）
python finetune_transformer.py --model microsoft/deberta-v3-base \
    --folds 5 --epochs 4 --bs 8 --accum 2 --name deberta_v1 --log

# 4) 融合并出提交
python blend.py --names bert_base_v2 tfidf_lr_v1 --out submissions/blend_v1.csv \
    --log --exp-name blend_v1

# 5) 查看实验记录
python exp_log.py list
```

## 评估协议（重要）

1. 固定 **5 折 StratifiedKFold**（`random_state=42`），分层依据可选 `target`（默认基准）或 `keyword`（复现赛题真实切分）
2. 指标用 **F1**，输出每折值与均值±std；**不在训练集上评估**
3. 每次实验保存 **OOF**，阈值在 OOF 上搜索
4. 新实验只有 CV F1 ≥ 当前最佳才保留；**不靠 public LB 反复试探**

## 环境相关的两个坑

1. **HuggingFace 被墙** → 脚本内置回退链，改用 **ModelScope** 拉权重：
   `本地路径 → models/ 缓存 → modelscope.snapshot_download → HF id`
2. **Kaggle 提交要把文件 PUT 到 `www.googleapis.com`（被墙）** → 会一直卡住。
   用 `submit_via_proxy.py`：第 1/3 步走本机直连 kaggle.com，第 2 步交给 **Windows 侧 curl.exe + 系统代理**。

## 注意事项

- `models/`（4.7G）**不入库**，需要时脚本会自动重新下载
- 凭据（`kaggle.json` / `access_token`）**绝不提交**，已在 `.gitignore` 中
- 若仓库要设为**公开**，请取消 `.gitignore` 中关于比赛数据与第三方 notebook 的注释
- 赛道红线：**不要使用 test 标签泄漏**（DFE 数据集按 `id` 可还原 test 标签），榜单上 ≥0.99 的队伍几乎都源于此，分数没有意义
