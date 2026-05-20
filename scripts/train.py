"""
Quoridor AI Agent 訓練腳本
使用 sb3_contrib 的 MaskablePPO 演算法訓練 AI Agent（具動作遮罩）
"""
import os
import sys
from pathlib import Path

# 將父目錄添加到路徑，以便導入 barricade_core
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback , BaseCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from barricade_core import QuoridorEnv

def linear_schedule(initial_value: float):
    """
    線性遞減函數：從初始值隨著訓練進度趨近於 0
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining 會從 1.0 遞減到 0.0
        return initial_value * progress_remaining
    return func

def train_ppo_agent(
    total_timesteps: int = 100000,
    learning_rate: float = 0.0001,
    n_steps: int = 4096,
    batch_size: int = 64,
    save_dir: str = 'models',
    tensorboard_log_dir: str = 'logs'
):
    """
    訓練 MaskablePPO Agent（具動作遮罩）
    
    :param total_timesteps: 總訓練時間步數
    :param learning_rate: 學習率（預設: 0.0001，降低以求穩定）
    :param n_steps: 每次更新前收集的步數（預設: 4096，增加以提高探索能力）
    :param batch_size: 批次大小
    :param save_dir: 模型保存目錄
    :param tensorboard_log_dir: TensorBoard 日誌目錄
    """
    # 建立保存目錄
    Path(save_dir).mkdir(exist_ok=True)
    Path(tensorboard_log_dir).mkdir(exist_ok=True)
    
    print(f"開始訓練 Quoridor AI Agent（MaskablePPO）...")
    print(f"總時間步數: {total_timesteps}")
    print(f"學習率: {learning_rate}")
    print(f"每次更新前收集的步數: {n_steps}")
    print()
    
    # 創建環境
    # env = QuoridorEnv()
    env = make_vec_env(lambda: QuoridorEnv(), n_envs=1)
    env = VecNormalize(env, norm_reward=True, clip_reward=10.0) # clip_reward 很重要！
    # 配置 MaskablePPO 策略和超參數
    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=linear_schedule(learning_rate),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.05,  # 提高熵係數以增加探索能力（原為 0.0）
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        device='cpu'
    )
    
    # 設定回調函數
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix='quoridor_ppo'
    )
    
    # 開始訓練
    # ✅ 關鍵: MaskablePPO 會自動在訓練過程中調用 env.action_masks()
    # 確保 AI 只從合法動作中採樣，不會「盲打」
    print("\n✅ 動作遮罩已啟用 - AI 將只從合法動作中選擇")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
        tb_log_name='quoridor_ppo_training'
    )
    
    # 保存最終模型
    final_model_path = os.path.join(save_dir, 'quoridor_ppo_final.zip')
    model.save(final_model_path)
    final_env_path= os.path.join(save_dir, 'quoridor_ppo_final_env.pkl')
    env.save(final_env_path)
    print(f"\n最終模型已保存到: {final_model_path}"
          f"環境統計數據已保存到: {final_env_path}")
    
    env.close()
    return model


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quoridor AI 訓練腳本')
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='訓練時間步數 (預設: 100000)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='學習率 (預設: 0.0001，已調整為更穩定的值)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='批次大小 (預設: 64)'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=4096,
        help='每次更新前收集的步數 (預設: 4096，已增加以提高探索能力)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='模型保存目錄 (預設: models)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='TensorBoard 日誌目錄 (預設: logs)'
    )
    
    args = parser.parse_args()
    
    train_ppo_agent(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        tensorboard_log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()
