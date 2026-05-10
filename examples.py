"""
快速開始範例
演示如何使用 Quoridor 環境和訓練腳本
"""
import numpy as np
from quoridor_env import QuoridorEnv


def example_1_basic_environment():
    """
    範例 1：基本環境使用
    演示如何創建環境和執行隨機動作
    """
    print("=" * 60)
    print("範例 1：基本環境使用")
    print("=" * 60)
    
    # 創建環境
    env = QuoridorEnv()
    
    # 重置環境並獲取初始觀察
    obs = env.reset()
    print(f"初始觀察向量形狀: {obs.shape}")
    print(f"觀察向量: {obs[:20]}...")  # 只顯示前 20 個元素
    
    # 獲取棋盤快照
    snapshot = env.get_board_snapshot()
    print(f"\n棋盤狀態:")
    print(f"  Player1 位置: {snapshot['player1_pos']}")
    print(f"  Player2 位置: {snapshot['player2_pos']}")
    print(f"  Player1 剩餘牆體: {snapshot['walls_remaining']['p1']}")
    print(f"  Player2 剩餘牆體: {snapshot['walls_remaining']['p2']}")
    
    # 執行幾個隨機動作
    print("\n執行 3 個隨機動作:")
    for step in range(3):
        # 從合法動作中隨機選擇
        legal_mask = env._get_legal_actions_mask()
        legal_actions = np.where(legal_mask)[0]
        action = np.random.choice(legal_actions)
        
        # 執行動作
        obs, reward, done, info = env.step(action)
        print(f"  Step {step+1}: 動作={action}, 獎勵={reward:.2f}, 完成={done}")
        
        if done:
            print(f"    信息: {info}")
            break
    
    env.close()
    print()


def example_2_full_game():
    """
    範例 2：進行一局完整的遊戲
    """
    print("=" * 60)
    print("範例 2：進行一局完整的遊戲")
    print("=" * 60)
    
    env = QuoridorEnv()
    obs = env.reset()
    
    done = False
    step_count = 0
    total_reward = 0
    
    print("開始遊戲...")
    
    while not done:
        # 選擇合法動作
        legal_mask = env._get_legal_actions_mask()
        legal_actions = np.where(legal_mask)[0]
        
        # 隨機選擇（可改為使用訓練好的模型）
        action = np.random.choice(legal_actions)
        
        # 執行動作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # 每 5 步打印一次信息
        if step_count % 5 == 0:
            print(f"Step {step_count}: 累積獎勵={total_reward:.2f}")
        
        if done:
            print(f"\n遊戲結束!")
            print(f"總步數: {step_count}")
            print(f"總獎勵: {total_reward:.2f}")
            print(f"結束原因: {info.get('reason', 'unknown')}")
            if 'winner' in info:
                print(f"獲勝者: {info['winner']}")
    
    env.close()
    print()


def example_3_legal_actions():
    """
    範例 3：探索合法動作機制
    """
    print("=" * 60)
    print("範例 3：合法動作機制")
    print("=" * 60)
    
    env = QuoridorEnv()
    obs = env.reset()
    
    # 獲取合法動作遮罩
    legal_mask = env._get_legal_actions_mask()
    legal_actions = np.where(legal_mask)[0]
    
    print(f"總動作空間大小: {len(legal_mask)}")
    print(f"初始合法動作數: {len(legal_actions)}")
    print(f"初始合法動作: {legal_actions.tolist()}")
    
    # 分類動作
    move_actions = [a for a in legal_actions if a < 81]
    h_wall_actions = [a for a in legal_actions if 81 <= a < 145]
    v_wall_actions = [a for a in legal_actions if 145 <= a]
    
    print(f"\n動作分類:")
    print(f"  移動動作數: {len(move_actions)}")
    print(f"  橫牆動作數: {len(h_wall_actions)}")
    print(f"  直牆動作數: {len(v_wall_actions)}")
    
    # 執行一次移動，看合法動作如何變化
    if move_actions:
        action = move_actions[0]
        obs, reward, done, info = env.step(action)
        
        if not done:
            legal_mask_after = env._get_legal_actions_mask()
            legal_actions_after = np.where(legal_mask_after)[0]
            print(f"\n執行移動後的合法動作數: {len(legal_actions_after)}")
    
    env.close()
    print()


def example_4_reward_structure():
    """
    範例 4：理解獎勵結構
    """
    print("=" * 60)
    print("範例 4：獎勵結構")
    print("=" * 60)
    
    env = QuoridorEnv()
    obs = env.reset()
    
    rewards = []
    actions = []
    
    print("收集 10 個動作的獎勵數據...")
    
    for _ in range(10):
        legal_mask = env._get_legal_actions_mask()
        legal_actions = np.where(legal_mask)[0]
        
        if len(legal_actions) == 0:
            break
        
        action = np.random.choice(legal_actions)
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        actions.append(action)
        
        if done:
            print(f"遊戲結束，原因: {info.get('reason', 'unknown')}")
            break
    
    print(f"\n獎勵統計:")
    print(f"  平均獎勵: {np.mean(rewards):.4f}")
    print(f"  最大獎勵: {np.max(rewards):.4f}")
    print(f"  最小獎勵: {np.min(rewards):.4f}")
    print(f"  獎勵列表: {[f'{r:.2f}' for r in rewards]}")
    
    env.close()
    print()


def main():
    """
    運行所有範例
    """
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        example_num = None
    
    examples = {
        '1': example_1_basic_environment,
        '2': example_2_full_game,
        '3': example_3_legal_actions,
        '4': example_4_reward_structure,
    }
    
    if example_num and example_num in examples:
        # 運行指定的範例
        examples[example_num]()
    else:
        # 運行所有範例
        print("\n" + "=" * 60)
        print("Quoridor AI 快速開始範例")
        print("=" * 60 + "\n")
        
        for key in sorted(examples.keys()):
            examples[key]()
            input("按 Enter 繼續下一個範例... ")


if __name__ == '__main__':
    main()
