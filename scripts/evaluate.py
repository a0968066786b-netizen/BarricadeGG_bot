"""
Quoridor AI Agent 評估腳本
評估訓練好的模型
"""
import os
import sys
from pathlib import Path

# 將父目錄添加到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from barricade_core import QuoridorEnv


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
    :return: 勝率和平均獎勵
    """
    # 檢查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型文件 {model_path}")
        return None, None
    
    # 加載模型
    model = PPO.load(model_path)
    
    # 創建環境
    env = QuoridorEnv(render_mode='human' if render else None)
    
    # 統計結果
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    print(f"\n開始評估 Agent（{num_episodes} 局遊戲）...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        
        while not (terminated or truncated):
            # 預測動作
            action, _ = model.predict(obs, deterministic=True)
            
            # 執行動作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
        
        episode_lengths.append(step_count)
        total_reward += episode_reward
        
        # 判斷勝敗
        if 'winner' in info and info['winner'] == 'current_player':
            wins += 1
            result = "✓ 勝利"
        else:
            result = "✗ 失敗"
        
        print(f"局 {episode + 1:2d}: {result} | 步數: {step_count:3d} | 獎勵: {episode_reward:7.2f}")
    
    env.close()
    
    # 打印評估結果
    print("-" * 60)
    win_rate = wins / num_episodes * 100
    avg_reward = total_reward / num_episodes
    avg_steps = sum(episode_lengths) / len(episode_lengths)
    
    print(f"評估結果:")
    print(f"  總勝場: {wins}/{num_episodes} ({win_rate:.1f}%)")
    print(f"  平均獎勵: {avg_reward:.2f}")
    print(f"  平均步數: {avg_steps:.1f}")
    print()
    
    return win_rate, avg_reward


def play_game(model_path: str):
    """
    與訓練好的 Agent 進行一局遊戲
    
    :param model_path: 模型文件路徑
    """
    # 檢查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型文件 {model_path}")
        return
    
    # 加載模型
    model = PPO.load(model_path)
    
    # 創建環境
    env = QuoridorEnv(render_mode='human')
    
    print("\n開始新遊戲...")
    print("-" * 60)
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    
    while not (terminated or truncated):
        # 預測動作
        action, _ = model.predict(obs, deterministic=True)
        
        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        env.render()
        print(f"Step {step_count} | Action: {action} | Reward: {reward:.2f}")
        print("-" * 60)
    
    print(f"遊戲結束!")
    print(f"總步數: {step_count}")
    if 'winner' in info:
        print(f"結果: {info['winner']}")
    
    env.close()


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quoridor AI 評估腳本')
    parser.add_argument(
        '--mode',
        choices=['evaluate', 'play'],
        default='evaluate',
        help='執行模式 (預設: evaluate)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/quoridor_ppo_final.zip',
        help='模型文件路徑 (預設: models/quoridor_ppo_final.zip)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='評估遊戲數 (預設: 10)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='是否渲染遊戲'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        evaluate_agent(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            render=args.render
        )
    elif args.mode == 'play':
        play_game(model_path=args.model_path)


if __name__ == '__main__':
    main()
