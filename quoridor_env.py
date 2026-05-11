"""
Quoridor 遊戲的 OpenAI Gym 環境實現
將 Board 邏輯轉化為標準的 Gym 接口
"""
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Tuple, Dict, Any
from rule import (
    Board, 
    action_id_to_action, 
    pos_to_xy, 
    xy_to_pos,
    BOARD_SIZE,
    MAX_WALLS
)


class QuoridorEnv(gymnasium.Env):
    """
    Quoridor 遊戲的 Gym 環境實現
    
    觀察空間：
    - 棋盤狀態打平為向量或矩陣
    - 包含玩家位置、牆體信息、可走位置等
    
    動作空間：
    - Discrete(209)：0-80 移動動作，81-144 橫牆，145-208 直牆
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(QuoridorEnv, self).__init__()
        
        # 初始化遊戲棋盤
        self.board = Board()
        
        # 定義動作空間：209 個動作 (0-80: 移動, 81-144: 橫牆, 145-208: 直牆)
        self.action_space = spaces.Discrete(209)
        
        # 定義觀察空間（Box 格式）
        # 觀察向量包含：
        # - 玩家1位置: 2 (x, y)
        # - 玩家2位置: 2 (x, y)
        # - 玩家1剩餘牆體: 1
        # - 玩家2剩餘牆體: 1
        # - 棋盤狀態矩陣: 81 (9x9)，用於表示牆體和可走位置
        # 總共：2 + 2 + 1 + 1 + 81 + 81 = 168
        obs_size = 2 + 2 + 1 + 1 + BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(obs_size,), 
            dtype=np.uint8
        )
        
        # 記錄遊戲狀態
        self.current_player_index = 0  # 0 for player1, 1 for player2
        self.step_count = 0
        self.max_steps = 200  # 防止無限遊戲
        
    def reset(self) -> np.ndarray:
        """
        重置環境，開始新的一局遊戲
        
        :return: 初始觀察值
        """
        self.board = Board()
        self.current_player_index = 0
        self.step_count = 0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        將棋盤狀態轉換為觀察向量
        
        :return: 觀察向量 (obs_size,)
        """
        obs = []
        
        # 1. 玩家1位置 (2 values)
        p1_x, p1_y = self.board.player1.pos
        obs.extend([p1_x, p1_y])
        
        # 2. 玩家2位置 (2 values)
        p2_x, p2_y = self.board.player2.pos
        obs.extend([p2_x, p2_y])
        
        # 3. 玩家1剩餘牆體 (1 value)
        obs.append(self.board.player1.walls_left)
        
        # 4. 玩家2剩餘牆體 (1 value)
        obs.append(self.board.player2.walls_left)
        
        # 5. 棋盤狀態矩陣 - 牆體信息 (81 values)
        # 將棋盤轉換為 9x9 的矩陣：
        # 0 = 空格，1 = 玩家1位置，2 = 玩家2位置，3 = 有牆體
        board_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        board_state[p1_y, p1_x] = 1
        board_state[p2_y, p2_x] = 2
        
        # 標記牆體位置（簡化版：只標記有牆的格子）
        for _, col, row in self.board.h_walls:
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                board_state[row, col] = 3
        for _, col, row in self.board.v_walls:
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                board_state[row, col] = 3
        
        obs.extend(board_state.flatten().tolist())
        
        # 6. 可走位置遮罩 (81 values)
        valid_moves_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        for x, y in self.board.current_player.valid_moves:
            valid_moves_mask[y, x] = 1
        
        obs.extend(valid_moves_mask.flatten().tolist())
        
        return np.array(obs, dtype=np.uint8)
    
    def _get_legal_actions_mask(self) -> np.ndarray:
        """
        取得合法動作遮罩，用於約束 AI 的動作選擇
        
        :return: 長度為 209 的布林陣列
        """
        return np.array(self.board.get_legal_actions_mask(), dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        執行一個動作，返回 (觀察值, 獎勵, 完成標誌, 額外信息)
        
        :param action: 動作 ID (0-208)
        :return: (observation, reward, done, info)
        """
        self.step_count += 1
        
        # 預設獎勵
        reward = 0.0
        done = False
        info = {}
        
        # 檢查動作是否合法
        legal_mask = self._get_legal_actions_mask()
        if not legal_mask[action]:
            # 非法動作，給予負獎勵並結束遊戲
            reward = -1.0
            done = True
            info['reason'] = 'illegal_action'
            return self._get_observation(), reward, done, info
        
        # 將動作 ID 轉換為實際動作
        action_type, param = action_id_to_action(action)
        
        # 評估動作的獎勵（基於棋盤邏輯）
        base_reward = self.board.evaluate_action_reward(action_type, param)
        
        # 執行動作
        success = self.board.take_action(action_type, param)
        
        if not success:
            # 動作失敗，給予負獎勵
            reward = -1.0
            done = True
            info['reason'] = 'action_failed'
            return self._get_observation(), reward, done, info
        
        # 若動作成功，使用棋盤評估的獎勵
        if base_reward != float('-inf'):
            reward = base_reward
        else:
            reward = -0.1  # 小負獎勵
        
        # 檢查獲勝條件
        winner = self.board.check_win()
        if winner:
            # 有玩家獲勝
            done = True
            if winner == self.board.current_player.name:
                reward += 100.0  # 巨額獎勵
                info['winner'] = 'current_player'
            else:
                reward -= 50.0  # 失敗懲罰
                info['winner'] = 'other_player'
            info['reason'] = 'game_over'
        
        # 檢查是否超過最大步數
        elif self.step_count >= self.max_steps:
            done = True
            info['reason'] = 'max_steps_exceeded'
            reward -= 0.5  # 輕微懲罰
        
        # 切換到下一個玩家
        if not done:
            self.board.switch_player()
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode: str = 'human'):
        """
        渲染棋盤狀態（文字輸出）
        
        :param mode: 渲染模式
        """
        if mode == 'human':
            self.board.print_board()
    
    def close(self):
        """
        關閉環境
        """
        pass
    
    def get_board_snapshot(self) -> Dict:
        """
        取得棋盤快照（用於調試）
        
        :return: 棋盤狀態字典
        """
        return self.board.get_board_snapshot()
