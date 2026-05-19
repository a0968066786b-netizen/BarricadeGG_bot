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
        
        # 【不對稱雙向滾雪球計數器】
        self.violation_stack = 0      # 錯誤欠債層數
        self.valid_streak = 0         # 正確連勝紀錄
        
    def reset(self, seed=None, options=None):
        """重置環境到初始狀態"""
        super().reset(seed=seed)
        
        self.board = Board()
        self.step_count = 0
        self.current_player_index = 0
        
        # 重置計數器
        self.violation_stack = 0
        self.valid_streak = 0
        
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
    
    def _apply_snowball_penalty(self) -> float:
        """
        【不對稱雙向滾雪球計數器 - 犯錯懲罰】
        
        當 AI 犯錯（違規或無效動作）時：
        1. violation_stack += 1（錯誤欠債加重）
        2. valid_streak = 0（好不容易累積的連勝紀錄一秒破功）
        3. 根據 violation_stack 層級進行階梯式或巨額懲罰
        
        Returns:
            額外懲罰金額（負數）
        """
        self.violation_stack += 1
        self.valid_streak = 0  # 致命大棒：連勝紀錄直接歸零
        
        # 階梯式 + 巨額懲罰邏輯
        if self.violation_stack >= 5:
            # 觸發破產門檻：額外巨額懲罰
            extra_penalty = -100.0
        else:
            # 未滿 5 次：階梯式懲罰（每層 -5.0）
            extra_penalty = -5.0 * self.violation_stack
        
        return extra_penalty
    
    def _apply_snowball_reward(self) -> float:
        """
        【不對稱雙向滾雪球計數器 - 正確獎勵】
        
        當 AI 做對（有效動作）時：
        1. violation_stack = max(0, violation_stack - 1)（債務只能靠做好事緩慢減少）
        2. valid_streak += 1（連勝紀錄滾雪球）
        3. 根據 valid_streak 層級進行小額滾雪球或巨額胡蘿蔔
        
        Returns:
            額外獎勵金額（正數）
        """
        self.violation_stack = max(0, self.violation_stack - 1)
        self.valid_streak += 1
        
        # 滾雪球小獎勵 + 高智商巨額胡蘿蔔
        streak_reward = 0.2 * self.valid_streak
        
        if self.valid_streak >= 15:
            # 觸發高智商好棋門檻：一次性巨額胡蘿蔔
            streak_reward += 50.0
        
        return streak_reward

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
        
        獎勵設計（路徑差值獎勵機制 + 不對稱雙向滾雪球計數器）：
        ============================================================
        【移動動作】：
        - 我方距離差獎勵：delta_self * 5.0（更接近終點為正）
        - 對手距離差獎勵：delta_opponent * 5.0（對手被遠離為正）
        - 基本步數稅：-0.5
        
        【放牆動作】：
        A. 無效放牆（路徑未變長）：-0.5
           - Action Mask 判定為非法，或
           - 放牆後對手最短路徑長度沒變長
           
        B. 有效放牆（成功卡位）：+0.5 × (路徑增量)
           - 放牆後對手最短路徑變長
           - 增量 = 放牆後步數 - 放牆前步數
        
        【不對稱雙向滾雪球計數器】：
        - 違規/無效動作 → violation_stack += 1, valid_streak = 0
          - 若 violation_stack >= 5：extra_penalty = -100.0
          - 否則：extra_penalty = -5.0 * violation_stack
        - 有效動作 → violation_stack = max(0, violation_stack - 1), valid_streak += 1
          - streak_reward = 0.2 * valid_streak
          - 若 valid_streak >= 15：streak_reward += 50.0
        
        【特殊事件】：
        - 勝利：+200
        - 失敗（對手勝利）：-50
        - 自己被完全封鎖：-50
        - 非法動作：-2
        - 超時：-50
        ============================================================
        """
        self.step_count += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        ILLEGAL_ACTION_SCORE = -2.0 # 非法動作的懲罰
        INEFFECTIVE_WALKING = -0.5 # 無效移動的懲罰
        INEFFECTIVE_WALL = -1.0 # 無效放牆的懲罰
        info = {}
        
        # 【動作分類標記】用於滾雪球結算
        is_violation = False      # 是否違規
        is_invalid = False        # 是否無效
        is_valid = False          # 是否有效
        
        # 1. 檢查動作是否合法（使用 Action Mask）
        legal_mask = self._get_legal_actions_mask()
        if not legal_mask[action]:
            # 非法動作（不應發生，因為有 Masking），給予懲罰
            info['reason'] = 'illegal_action_masked'
            reward = ILLEGAL_ACTION_SCORE
            is_violation = True
            self.step_count -= 1 # 這裡非法就給懲罰就好，不要增加步數
            return self._get_observation(), reward + self._apply_snowball_penalty(), terminated, truncated, info
        
        # 2. 記錄動作前的狀態
        action_type, param = action_id_to_action(action)
        
        # 記錄動作前對手的最短路徑長度
        opponent_distance_before = self.board.calc_shortest_path_cost(
            self.board.other_player.pos,
            self.board.other_player.goal_row
        )
        
        # 記錄動作前我方的最短路徑長度
        self_distance_before = self.board.calc_shortest_path_cost(
            self.board.current_player.pos,
            self.board.current_player.goal_row
        )
        
        #行動前檢查一下，棋盤面是否已經違反規則了
        #(這裡是檢查牆是否已封鎖玩家的狀態)
        is_historical_corrupted = (self_distance_before == -1 or opponent_distance_before == -1)

        # 3. 執行動作
        success = self.board.take_action(action_type, param)
        # 強制同步更新狀態
        self.board.update_all_valid_moves()
        
        if not success:
            # 動作失敗（不應發生），給予懲罰
            info['reason'] = 'action_failed'
            reward = ILLEGAL_ACTION_SCORE
            is_violation = True
            self.step_count -= 1 
            return self._get_observation(), reward + self._apply_snowball_penalty(), terminated, truncated, info
        
        # 4. 動作成功，計算獎勵
        # 記錄動作後我方的最短路徑長度
        self_distance_after = self.board.calc_shortest_path_cost(
            self.board.current_player.pos,
            self.board.current_player.goal_row
        )
        
        # 記錄動作後對手的最短路徑長度
        opponent_distance_after = self.board.calc_shortest_path_cost(
            self.board.other_player.pos,
            self.board.other_player.goal_row
        )
        
        # 4.1 檢查幹了一個我方與對方是否被完全封鎖的放牆動作(封鎖本身就是一個非法動作)
        if self_distance_after == -1:
            # 自己被封鎖
            reward = ILLEGAL_ACTION_SCORE
            info['reason'] = 'self_blocked'
            is_violation = True
            #復原一次錯誤的狀態(也就是此次的操作錯誤)
            self.board.undo_action()
            self.step_count -= 1
            return self._get_observation(), reward + self._apply_snowball_penalty(), terminated, truncated, info
        
        if opponent_distance_after == -1:
            # 對方被封鎖
            reward = ILLEGAL_ACTION_SCORE
            info['reason'] = 'opponent_completely_blocked_illegal'
            is_violation = True
            #復原一次錯誤的狀態(也就是此次的操作錯誤)
            self.board.undo_action()
            self.step_count -= 1
            return self._get_observation(), reward + self._apply_snowball_penalty(), terminated, truncated, info
        
        if self.board.player1.pos == self.board.player2.pos:
            # 🌟 【防線：行動後攔截當前違規】防止重疊 Bug
            info['reason'] = 'bug_illegal_pawn_overlap'
            reward = ILLEGAL_ACTION_SCORE  # 發現重疊，扣分退回
            is_violation = True
            #復原一次錯誤的狀態(也就是此次的操作錯誤)
            self.board.undo_action()
            self.step_count -= 1
            return self._get_observation(), reward + self._apply_snowball_penalty(), terminated, truncated, info
        
        
        # 4.2 計算距離差獎勵（針對所有動作類型）
        delta_self = self_distance_before - self_distance_after
        delta_opponent = opponent_distance_after - opponent_distance_before
        
        # 4.3 根據動作類型計算獎勵 + 分類為「違規」「無效」「有效」
        # 優先判斷爛旗面（歷史盤面已經違規了）
        if is_historical_corrupted:
            # 🌟 歷史旗面有被封鎖 - 無效動作
            reward = ILLEGAL_ACTION_SCORE
            info['reason'] = 'punished_for_historical_corruption'
            is_invalid = True
        
        elif action_type == 'move':
            if delta_self == 0 and delta_opponent == 0:
                # 【無效動作】原地折返跑
                reward = INEFFECTIVE_WALKING
                info['action_type'] = 'move'
                info['reason'] = 'ineffective_walking'
                is_invalid = True
            else:
                # 【有效動作】推進或卡位
                reward += (delta_self * 5.0) + (delta_opponent * 5.0)
                reward -= 0.05  # 步數稅
                info['action_type'] = 'move'
                info['reason'] = 'effective_move'
                is_valid = True
            
        elif action_type == 'wall':
            # 正常情況：比較路徑變化
            wall_path_delta = opponent_distance_after - opponent_distance_before
                
            if wall_path_delta > 0:
                # 【有效動作】成功卡位：路徑變長
                reward = 0.5 * wall_path_delta
                info['wall_effective'] = True
                info['wall_path_delta'] = wall_path_delta
                info['reason'] = 'wall_successful_blocking'
                is_valid = True
            else:
                # 【無效動作】放垃圾牆：路徑未變長或更短
                reward = INEFFECTIVE_WALL
                info['wall_effective'] = False
                info['wall_path_delta'] = wall_path_delta
                info['reason'] = 'wall_ineffective'
                is_invalid = True

        # 5. 檢查勝利條件
        winner = self.board.check_win()
        if winner:
            terminated = True
            if winner == self.board.current_player.name:
                reward += 200.0
                info['winner'] = self.board.current_player.name
            else:
                reward -= 50.0
                info['winner'] = self.board.other_player.name
        
        # 6. 檢查超時
        elif self.step_count >= self.max_steps:
            truncated = True
            reward -= 50.0
            info['reason'] = 'max_steps_exceeded'
        
        # 7. 【滾雪球結算中心】應用不對稱雙向計數器獎懲
        if is_violation or is_invalid:
            # 犯錯路線：違規或無效動作
            extra_penalty = self._apply_snowball_penalty()
            reward += extra_penalty
            info['violation_stack'] = self.violation_stack
            info['valid_streak'] = self.valid_streak
        elif is_valid:
            # 正確路線：有效動作
            streak_reward = self._apply_snowball_reward()
            reward += streak_reward
            info['violation_stack'] = self.violation_stack
            info['valid_streak'] = self.valid_streak
        
        # 8. 切換玩家
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
