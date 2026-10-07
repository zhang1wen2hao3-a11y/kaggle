# Kaggle 竞赛工程集

按竞赛分目录存放可复现的实验工程（脚本 + 实验记录 + 训练日志 + OOF/提交产物）。
预训练权重等大文件不入库，由脚本按需自动下载。

| 目录 | 竞赛 | 指标 | 当前最佳 | 说明 |
|---|---|---|---|---|
| [`disaster-tweets/`](disaster-tweets/) | [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) | F1 | **public LB 0.83849**（rank 70/442） | TF-IDF 基线 → BERT 微调 → 模型融合 |
| [`the-pokemon-company/`](the-pokemon-company/) | [The Pokémon Company - PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground) | cabt_bo3 (Bo3 胜率) | 进行中 | 宝可梦集换式卡牌对战 agent（数据不入库，跑 `fetch_data.sh`） |

## 约定

- 每个子目录自带 `README.md`、`.gitignore` 与实验记录（`experiments.csv` / `实验记录.md`）
- **不提交**：模型权重（`models/`）、缓存（`__pycache__`）、凭据（`kaggle.json` / `access_token`）
- 实验记录里的 `CV F1` 与 `public LB` 一一对应，便于追溯"代码 → 本地验证 → 榜单成绩"
