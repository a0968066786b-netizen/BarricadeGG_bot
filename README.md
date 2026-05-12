# Quoridor AI Agent

使用 Gymnasium 和 Stable Baselines3 訓練一個能玩 Quoridor 遊戲的 AI Agent。

## 快速開始

### 安裝

```bash
# 安裝依賴
pip install -r requirements.txt

# 或使用 setup.py
pip install -e .
```

### 訓練模型

```bash
# 使用默認配置訓練 (100,000 步)
python scripts/train.py

# 自定義訓練
python scripts/train.py --timesteps 500000 --learning-rate 1e-4
```

### 評估模型

```bash
# 評估模型 (10 局)
python scripts/evaluate.py

# 與 AI 遊玩
python scripts/evaluate.py --mode play
```

## 項目結構

```
├── barricade_core/        # 核心遊戲和環境實現
│   ├── env.py           # Gymnasium 環境
│   └── rules.py         # 遊戲規則
├── scripts/              # 訓練和評估腳本
│   ├── train.py
│   └── evaluate.py
├── models/               # 訓練好的模型存儲位置
├── logs/                 # TensorBoard 日誌
└── setup.py
```

## 特性

- ✅ 完整的 Gymnasium 環境實現
- ✅ PPO 演算法訓練
- ✅ 合法動作約束
- ✅ 動態獎勵函數
- ✅ TensorBoard 監控
- ✅ 可安裝的 Python 包

## 文檔

- [訓練說明.md](訓練說明.md) - 詳細的訓練和使用指南
- [遊戲規則.md](遊戲規則.md) - Quoridor 遊戲規則

## 系統要求

- Python 3.8+
- numpy >= 1.21.0
- gymnasium >= 0.27.0
- stable-baselines3 >= 2.0.0

## 示例

### 基礎訓練

```bash
python scripts/train.py --timesteps 100000
```

### 超參數自定義

```bash
python scripts/train.py \
  --timesteps 500000 \
  --learning-rate 1e-4 \
  --batch-size 32 \
  --n-steps 1024
```

### 模型評估

```bash
# 測試 10 局
python scripts/evaluate.py --num-episodes 10

# 詳細評估 (50 局，顯示遊戲過程)
python scripts/evaluate.py --num-episodes 50 --render
```

## 許可

本項目遵循開源許可。
