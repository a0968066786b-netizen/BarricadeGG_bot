## 快速使用指南

### 🎯 第一步：設置環境

```bash
# 1. 進入項目目錄
cd BarricadeGG_bot

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 驗證安裝
python -c "from quoridor_core import QuoridorEnv; print('✓ 安裝成功!')"
```

### 🏋️ 第二步：訓練模型

```bash
# 快速訓練 (推薦新手)
python scripts/train.py --timesteps 100000

# 詳細訓練 (更好的性能)
python scripts/train.py --timesteps 500000

# 監控訓練進度 (在另一個終端)
tensorboard --logdir=logs
# 打開瀏覽器: http://localhost:6006
```

### 📊 第三步：評估模型

```bash
# 快速評估
python scripts/evaluate.py

# 詳細評估 (50 局)
python scripts/evaluate.py --num-episodes 50

# 查看遊戲過程
python scripts/evaluate.py --num-episodes 10 --render
```

### 🎮 第四步：與 AI 遊玩

```bash
# 進行一局遊戲
python scripts/evaluate.py --mode play --render
```

---

## 常用命令速查

### 訓練腳本參數

```bash
python scripts/train.py \
  --timesteps 100000      # 訓練步數 (預設: 100000)
  --learning-rate 3e-4    # 學習率 (預設: 3e-4)
  --batch-size 64         # 批次大小 (預設: 64)
  --n-steps 2048          # 每次更新步數 (預設: 2048)
  --save-dir models       # 模型保存目錄 (預設: models)
  --log-dir logs          # 日誌保存目錄 (預設: logs)
```

### 評估腳本參數

```bash
python scripts/evaluate.py \
  --mode evaluate              # 執行模式: evaluate 或 play
  --model-path models/...zip   # 模型文件路徑
  --num-episodes 10            # 評估局數
  --render                     # 顯示遊戲過程
```

---

## 目錄結構簡明說明

```
quoridor_core/      ← 遊戲邏輯和環境
  ├── rules.py      ← 棋盤、玩家、規則
  └── env.py        ← Gymnasium 環境

scripts/            ← 訓練和評估
  ├── train.py      ← 訓練模型
  └── evaluate.py   ← 評估模型

models/             ← 存儲訓練好的模型 (自動創建)
logs/               ← TensorBoard 日誌 (自動創建)
```

---

## 常見問題速解

| 問題 | 解決方案 |
|------|--------|
| `ModuleNotFoundError: quoridor_core` | 確保在項目根目錄運行，或執行 `pip install -e .` |
| 訓練很慢 | 正常 (CPU 約 30-60 分鐘/100k 步) |
| 找不到模型文件 | 先訓練模型: `python scripts/train.py` |
| 記憶體不足 | 減少參數: `--batch-size 32 --n-steps 1024` |

---

## 性能指標

| 訓練步數 | 預計時間 | 勝率 | 適用場景 |
|--------|--------|------|--------|
| 10,000 | 2-5 分鐘 | ~ 5% | 快速測試 |
| 100,000 | 30-60 分鐘 | 20-40% | 基礎訓練 |
| 500,000 | 3-6 小時 | 50-70% | 完整訓練 |

---

## 下一步

- 📖 詳細文檔：[訓練說明.md](訓練說明.md)
- 🔧 項目重組總結：[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md)
- 📋 遊戲規則：[遊戲規則.md](遊戲規則.md)

---

## 技術棧

- **遊戲環境**: Gymnasium 0.27+
- **強化學習**: Stable Baselines3 (PPO)
- **監控**: TensorBoard
- **Python**: 3.8+

開始訓練吧！🚀
