"""
Quoridor 遊戲的 Gymnasium 環境實現
將 Board 邏輯轉化為標準的 Gym 接口
"""
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Tuple, Dict, Any

from .rules import (
    Board, 
    action_id_to_action, 
    pos_to_xy, 
    xy_to_pos,
    BOARD_SIZE,
    MAX_WALLS
)


class QuoridorEnv(gymnasium.Env):
    """
    Quoridor 遊戲的 Gymnasium 環境實現
    
    觀察空間：
    - 棋盤狀態打平為向量
    - 包含玩家位置、牆體信息、可走位置等
    
    動作空間：
    - Discrete(209)：0-80 移動動作，81-144 橫牆，145-208 直牆
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None):
        super(QuoridorEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # 初始化遊戲棋盤
        self.board = Board()
        
        # 定義動作空間
        self.action_space = spaces.Discrete(209)
        
        # 定義觀察空間
        obs_size = 2 + 2 + 1 + 1 + BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(obs_size,), 
            dtype=np.uint8
        )
        
        # 記錄遊戲狀態
        self.current_player_index = 0
        self.step_count = 0
        self.max_steps = 200
        
    def reset(self, seed=None, options=None):
        """重置環境到初始狀態"""
        super().reset(seed=seed)
        
        self.board = Board()
        self.step_count = 0
        self.current_player_index = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """將棋盤狀態轉換為觀察向量"""
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
        
        # 5. 棋盤狀態矩陣 (81 values)
        board_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        board_state[p1_y, p1_x] = 1
        board_state[p2_y, p2_x] = 2
        
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
        """取得合法動作遮罩"""
        return np.array(self.board.get_legal_actions_mask(), dtype=np.float32)
    
    def action_masks(self) -> np.ndarray:
        """
        標準動作遮罩介面 - 由 MaskablePPO 調用
        
        Returns:
            長度 209 的布林陣列，表示當前可執行的動作
        """
        return np.array(self.board.get_legal_actions_mask(), dtype=np.bool_)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        執行動作，符合 Gymnasium 標準回傳 5 個值
        
        重構的獎勵設計（導向目標型）：
        - 強化距離差獎勵：(delta_self * 5.0) + (delta_opponent * 5.0)
        - 放牆激勵：成功 +1.0，顯著阻挡 +10.0
        - 已有 Masking，移除非法動作懲罰
        - 勝利：+200
        - 超時：-5
        """
        self.step_count += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # 1. 檢查動作是否合法（雖然 Masking 應該防止非法動作，但仍做防守性檢查）
        legal_mask = self._get_legal_actions_mask()
        if not legal_mask[action]:
            # 已有 Masking，此情況理論上不應發生，若發生則忽略並切換玩家
            info['reason'] = 'illegal_action_masked'
            self.board.switch_player()
            return self._get_observation(), reward, terminated, truncated, info
        
        # 2. 記錄動作前的狀態（用於計算距離差）
        action_type, param = action_id_to_action(action)
        
        # 獲取執行動作前的雙方到終點距離
        self_distance_before = self.board.get_distance_to_goal(self.board.current_player.name)
        opponent_distance_before = self.board.get_distance_to_goal(self.board.other_player.name)
        
        # 3. 執行動作
        success = self.board.take_action(action_type, param)
        
        if not success:
            # 動作失敗（不應發生，因為已檢查合法性），切換玩家
            info['reason'] = 'action_failed'
            self.board.switch_player()
            return self._get_observation(), reward, terminated, truncated, info
        
        # 4. 動作成功，計算獎勵
        # 獲取執行動作後的雙方到終點距離
        self_distance_after = self.board.get_distance_to_goal(self.board.current_player.name)
        opponent_distance_after = self.board.get_distance_to_goal(self.board.other_player.name)
        
        # 4.1 計算距離差獎勵
        # delta_self: 我方距離減少（更接近終點）為正值
        # delta_opponent: 對手距離增加（被阻礙）為正值
        if self_distance_after != -1:
            delta_self = self_distance_before - self_distance_after
        else:
            # 自己被封鎖，終止遊戲
            reward = -50.0
            terminated = True
            info['reason'] = 'self_blocked'
            return self._get_observation(), reward, terminated, truncated, info
        
        if opponent_distance_after != -1:
            delta_opponent = opponent_distance_before - opponent_distance_after
        else:
            delta_opponent = 0  # 對手被封鎖不懲罰，但也不額外獎勵
        
        # 強化距離差獎勵
        reward += (delta_self * 5.0) + (delta_opponent * 5.0)
        
        # 4.2 放牆激勵
        if action_type == 'wall':
            # 放置牆體成功，基礎獎勵
            reward += 1.0
            
            # 如果該牆體導致對手的最短路徑顯著增加
            if delta_opponent > 0:
                reward += 10.0
                info['wall_effective'] = True
        
        # 4.3 基本步數稅（略微懲罰每一步，鼓勵高效）
        reward -= 0.05
        
        # 5. 檢查勝利條件
        winner = self.board.check_win()
        if winner:
            terminated = True
            if winner == self.board.current_player.name:
                reward += 200.0
                info['winner'] = 'current_player'
            else:
                reward -= 50.0
                info['winner'] = 'other_player'
        
        # 6. 檢查超時
        elif self.step_count >= self.max_steps:
            truncated = True
            reward -= 50.0
            info['reason'] = 'max_steps_exceeded'
        
        # 7. 切換玩家
        if not (terminated or truncated):
            self.board.switch_player()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        """渲染棋盤狀態（文字輸出）"""
        if self.render_mode == 'human':
            self.board.print_board()
    
    def close(self):
        """關閉環境"""
        pass
    
    def get_board_snapshot(self) -> Dict:
        """取得棋盤快照（用於調試）"""
        return self.board.get_board_snapshot()
