"""
Quoridor 遊戲訓練腳本
使用 Stable Baselines3 的 PPO 演算法訓練 AI Agent

安裝依賴：
    pip install stable-baselines3 gymnasium sb3-contrib
"""
import numpy as np
import os
from pathlib import Path

# 導入 Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# 導入自定義環境
from quoridor_env import QuoridorEnv


class WinRateCallback(BaseCallback):
    """
    自定義回調函數，用於監控訓練過程中的勝率
    """
    def __init__(self, verbose: int = 0):
        super(WinRateCallback, self).__init__(verbose)
        self.win_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        """
        每一步後執行
        """
        # 檢查是否有完成的遊戲
        if 'done' in self.locals:
            for done in self.locals['dones']:
                if done:
                    self.total_episodes += 1
                    # 這裡可以獲取更多信息，例如回報
        
        return True
    
    def _on_training_end(self) -> None:
        """
        訓練結束後執行
        """
        if self.verbose > 0:
            print(f"訓練完成！總遊戲數: {self.total_episodes}")


def create_train_env(num_envs: int = 1):
    """
    創建訓練環境（支持平行處理）
    
    :param num_envs: 並行環境數
    :return: 向量化環境
    """
    env = make_vec_env(
        'quoridor_env:QuoridorEnv',
        n_envs=num_envs,
        seed=42,
        wrapper_class=None
    )
    return env


def train_ppo_agent(
    total_timesteps: int = 100000,
    num_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    save_dir: str = 'models'
):
    """
    訓練 PPO Agent
    
    :param total_timesteps: 總訓練時間步數
    :param num_envs: 並行環境數
    :param learning_rate: 學習率
    :param n_steps: 每次更新前收集的步數
    :param batch_size: 批次大小
    :param save_dir: 模型保存目錄
    """
    # 建立保存目錄
    Path(save_dir).mkdir(exist_ok=True)
    
    print(f"開始訓練 Quoridor AI Agent...")
    print(f"總時間步數: {total_timesteps}")
    print(f"並行環境數: {num_envs}")
    print(f"學習率: {learning_rate}")
    print()
    
    # 創建單個環境用於測試
    env = QuoridorEnv()
    
    # 配置 PPO 策略和超參數
    model = PPO(
        policy='MlpPolicy',  # 使用全連接網路
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,  # 每次更新的 epoch 數
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # GAE 參數
        clip_range=0.2,  # PPO 裁剪範圍
        clip_range_vf=None,  # 價值函數裁剪範圍
        ent_coef=0.0,  # 熵系數（增加探索）
        vf_coef=0.5,  # 價值函數損失係數
        max_grad_norm=0.5,  # 梯度範數限制
        verbose=1,
        tensorboard_log="./logs",
        device='cpu'  # 可改為 'cuda' 用於 GPU 訓練
    )
    
    # 設定回調函數
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix='quoridor_ppo'
    )
    
    win_rate_callback = WinRateCallback(verbose=1)
    
    # 開始訓練
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, win_rate_callback],
        progress_bar=True,
        tb_log_name='quoridor_ppo_training'
    )
    
    # 保存最終模型
    final_model_path = os.path.join(save_dir, 'quoridor_ppo_final.zip')
    model.save(final_model_path)
    print(f"\n最終模型已保存到: {final_model_path}")
    
    env.close()
    return model


def evaluate_agent(
    model_path: str,
    num_episodes: int = 10,
    render: bool = False
):
    """
    評估訓練好的 Agent
    
    :param model_path: 模型文件路徑
    :param num_episodes: 評估的遊戲數
    :param render: 是否渲染遊戲
    """
    # 加載模型
    model = PPO.load(model_path)
    
    # 創建環境
    env = QuoridorEnv()
    
    # 統計結果
    wins = 0
    total_reward = 0
    
    print(f"\n開始評估 Agent（{num_episodes} 局遊戲）...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # 預測動作
            action, _ = model.predict(obs, deterministic=True)
            
            # 執行動作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            if render:
                env.render()
            
            # 檢查勝利
            if done:
                if 'winner' in info and info['winner'] == 'current_player':
                    wins += 1
                    print(f"局 {episode + 1}: 勝利! (步數: {step_count}, 獎勵: {episode_reward:.2f})")
                else:
                    print(f"局 {episode + 1}: 失敗 (步數: {step_count}, 獎勵: {episode_reward:.2f})")
        
        total_reward += episode_reward
    
    env.close()
    
    # 打印評估結果
    print("-" * 50)
    print(f"評估結果:")
    print(f"  總勝場: {wins}/{num_episodes}")
    print(f"  勝率: {wins/num_episodes*100:.1f}%")
    print(f"  平均獎勵: {total_reward/num_episodes:.2f}")


def play_game_with_agent(model_path: str, render: bool = True):
    """
    與訓練好的 Agent 玩一局遊戲（自由模式）
    
    :param model_path: 模型文件路徑
    :param render: 是否渲染遊戲
    """
    # 加載模型
    model = PPO.load(model_path)
    
    # 創建環境
    env = QuoridorEnv()
    
    print("\n開始新遊戲...")
    obs, _ = env.reset()
    done = False
    step_count = 0
    
    while not done:
        # 預測動作
        action, _ = model.predict(obs, deterministic=True)
        
        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

        if render:
            env.render()
            print(f"Step {step_count} | Action: {action} | Reward: {reward:.2f}")
            print("-" * 30)
        
        if done:
            print(f"遊戲結束!")
            print(f"總步數: {step_count}")
            if 'winner' in info:
                print(f"獲勝者: {info['winner']}")
    
    env.close()


def main():
    """
    主函數：訓練和評估流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Quoridor AI 訓練腳本')
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'play'],
        default='train',
        help='執行模式'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='訓練時間步數'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/quoridor_ppo_final.zip',
        help='模型文件路徑'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='評估遊戲數'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='渲染遊戲'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 訓練模式
        train_ppo_agent(
            total_timesteps=args.timesteps,
            num_envs=4,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            save_dir='models'
        )
    
    elif args.mode == 'evaluate':
        # 評估模式
        if not os.path.exists(args.model_path):
            print(f"錯誤: 找不到模型文件 {args.model_path}")
            print("請先訓練模型: python train.py --mode train")
            return
        
        evaluate_agent(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            render=args.render
        )
    
    elif args.mode == 'play':
        # 遊戲模式
        if not os.path.exists(args.model_path):
            print(f"錯誤: 找不到模型文件 {args.model_path}")
            print("請先訓練模型: python train.py --mode train")
            return
        
        play_game_with_agent(
            model_path=args.model_path,
            render=True
        )


if __name__ == '__main__':
    main()
