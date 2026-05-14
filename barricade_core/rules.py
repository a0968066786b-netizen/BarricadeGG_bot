"""
Quoridor 遊戲規則實現
依據遊戲規則.md，建立棋盤、玩家、牆體、移動與放置牆體的基本邏輯
"""
from typing import List, Tuple, Set, Dict,Optional
from collections import deque

# 棋盤大小
BOARD_SIZE = 9

# 牆體數量
MAX_WALLS = 10

# 方向定義（上下左右）
DIRECTIONS = {
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0)
}

# 跳躍規則定義
STRAIGHT_JUMP = {
    'up': (0, 2),
    'down': (0, -2),
    'left': (-2, 0),
    'right': (2, 0)
}

# 斜角規則定義 - 相對於對手位置的偏移（當直線被阻擋時可以跳到對手身邊）
DIAGONAL_JUMP = {
    'up': ((-1, 0), (1, 0)),      # 向上被阻時，跳到對手左邊或右邊
    'down': ((-1, 0), (1, 0)),    # 向下被阻時，跳到對手左邊或右邊
    'left': ((0, -1), (0, 1)),    # 向左被阻時，跳到對手上邊或下邊
    'right': ((0, -1), (0, 1))    # 向右被阻時，跳到對手上邊或下邊
}


def pos_to_xy(pos: str) -> Tuple[int, int]:
    """棋盤座標轉換：a1 ~ i9 轉成 (x, y)"""
    col = ord(pos[0]) - ord('a')
    row = int(pos[1]) - 1
    return (col, row)


def xy_to_pos(x: int, y: int) -> str:
    """座標轉換回棋盤代碼"""
    return chr(ord('a') + x) + str(y + 1)


def action_id_to_action(action_id: int) -> Tuple[str, str]:
    """
    將動作 ID（0-208）轉換為行動指令
    - ID 0-80: 移動動作
    - ID 81-144: 放置橫牆
    - ID 145-208: 放置直牆
    """
    if 0 <= action_id <= 80:
        row_index = action_id // 9
        col_index = action_id % 9
        pos = xy_to_pos(col_index, row_index)
        return ('move', pos)
    elif 81 <= action_id <= 144:
        wall_id = action_id - 81
        row_index = wall_id // 8
        col_index = wall_id % 8
        wall_code = 'h' + xy_to_pos(col_index, row_index)
        return ('wall', wall_code)
    elif 145 <= action_id <= 208:
        wall_id = action_id - 145
        row_index = wall_id // 8
        col_index = wall_id % 8
        wall_code = 'v' + xy_to_pos(col_index, row_index)
        return ('wall', wall_code)
    else:
        raise ValueError(f"無效的動作 ID: {action_id}")


def action_to_action_id(action_type: str, param: str) -> int:
    """將行動指令轉換為動作 ID（0-208）"""
    if action_type == 'move':
        col_index, row_index = pos_to_xy(param)
        action_id = row_index * 9 + col_index
        return action_id
    elif action_type == 'wall':
        orientation = param[0]
        col_index = ord(param[1]) - ord('a')
        row_index = int(param[2]) - 1  # 溝槽 1~8 → 動作網格列索引 0~7
        if orientation == 'h':
            action_id = 81 + (row_index * 8 + col_index)
        elif orientation == 'v':
            action_id = 145 + (row_index * 8 + col_index)
        else:
            raise ValueError(f"無效的牆體方向: {orientation}")
        return action_id
    else:
        raise ValueError(f"無效的行動類型: {action_type}")


class Wall:
    """牆體類別，包含橫向(h)與直向(v)牆體。

    代碼後兩位依 遊戲規則.md：為牆溝槽交叉點；第三位為 1~8（牆 k 在第 k 與 k+1 列／欄之間），
    不可誤用棋格 pos_to_xy（例如 hd4 為第 4~5 列間的橫溝，不是棋格 d4）。
    """

    def __init__(self, code: str):
        self.orientation = code[0]  # 'h' 或 'v'
        letter = ord(code[1]) - ord('a')
        groove = int(code[2])
        if not (1 <= groove <= BOARD_SIZE - 1):
            raise ValueError(f"無效的牆體溝槽編號: {code!r}")
        if self.orientation == 'h':
            self.col = letter
            self.row = groove  # 與 _is_blocked_by_wall 向下檢查 ('h', x, y1) 的 y1 一致
        elif self.orientation == 'v':
            # 牆 L 在欄 L 與 L+1 之間 → 與 _is_blocked_by_wall 的 ('v', L+1, *) 一致
            self.col = letter + 1
            self.row = groove - 1  # 溝槽 k 佔棋格列 y=k-1 與 y=k
        else:
            raise ValueError(f"無效的牆體方向: {self.orientation!r}")


class Player:
    """玩家類別，紀錄位置、剩餘牆體數量、終點行"""
    def __init__(self, name: str, start_pos: str, goal_row: int):
        self.name = name
        self.pos = pos_to_xy(start_pos)
        self.walls_left = MAX_WALLS
        self.goal_row = goal_row
        self.valid_moves = set()


class Board:
    """棋盤類別，管理棋子、牆體、移動與放置牆體"""
    def __init__(self):
        self.player1 = Player('player1', 'e1', 8)
        self.player2 = Player('player2', 'e9', 0)
        self.current_player = self.player1
        self.other_player = self.player2
        self.h_walls = set()
        self.v_walls = set()
        self.update_all_valid_moves()

    def get_valid_moves(self, player: Player) -> set:
        """
        回傳指定玩家目前所有可走的格子座標集合
        
        規則：
        1. 可以移動到相鄰的四個方向（上下左右）
        2. 如果對手球在某個方向，可以穿過對手球跳到後面一格（直線跳躍）
        3. 如果直線跳躍被牆體阻擋，改成斜角跳躍（嘗試兩個斜角方向）
        4. 如果斜角位置也被牆體阻擋，該斜角方向不可行
        """
        x, y = player.pos
        ox, oy = self.other_player.pos
        moves = set()
        
        for dir_name, (dx, dy) in DIRECTIONS.items():
            # 相鄰的目標位置
            tx, ty = x + dx, y + dy
            
            # 邊界檢查
            if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE):
                continue
            
            # 檢查是否被牆體阻擋（不管是否有對手）
            if self._is_blocked_by_wall((x, y), (tx, ty)):
                continue
            
            # 目標位置是對手位置
            if (tx, ty) == (ox, oy):
                # 嘗試直線跳躍：穿過對手到對手後方一格
                jx, jy = x + STRAIGHT_JUMP[dir_name][0], y + STRAIGHT_JUMP[dir_name][1]
                
                # 檢查跳躍目標是否在棋盤內
                if 0 <= jx < BOARD_SIZE and 0 <= jy < BOARD_SIZE:
                    # 檢查直線跳躍是否不被牆體阻擋
                    if not self._is_blocked_by_wall((ox, oy), (jx, jy)):
                        moves.add((jx, jy))
                        continue  # 直線跳躍成功，不需考慮斜角
                
                # 直線跳躍被阻擋或超出邊界，嘗試斜角跳躍
                for ddx, ddy in DIAGONAL_JUMP[dir_name]:
                    sx, sy = ox + ddx, oy + ddy
                    
                    # 邊界檢查
                    if not (0 <= sx < BOARD_SIZE and 0 <= sy < BOARD_SIZE):
                        continue
                    
                    # 檢查斜角位置是否被牆體阻擋
                    # 斜角位置與對手位置相鄰，所以檢查這一步是否暢通
                    if not self._is_blocked_by_wall((ox, oy), (sx, sy)):
                        moves.add((sx, sy))
            else:
                # 正常移動到相鄰格子（不是對手位置，且無牆體阻擋）
                moves.add((tx, ty))
        
        return moves

    def update_all_valid_moves(self):
        """更新雙方玩家的可走格子集合"""
        self.player1.valid_moves = self.get_valid_moves(self.player1)
        self.player2.valid_moves = self.get_valid_moves(self.player2)

    def _is_blocked_by_wall(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        判斷相鄰格子之間是否被牆體阻擋（只檢查牆體，不檢查對手或邊界）
        """
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # 上方移動
        if y2 > y1 and (("h", x1, y1+1) in self.h_walls or ("h", x1-1, y1+1) in self.h_walls):
            return True
        # 下方移動
        if y2 < y1 and (("h", x1, y1) in self.h_walls or ("h", x1-1, y1) in self.h_walls):
            return True
        # 右方移動
        if x2 > x1 and (("v", x1+1, y1) in self.v_walls or ("v", x1+1, y1-1) in self.v_walls):
            return True
        # 左方移動
        if x2 < x1 and (("v", x1, y1) in self.v_walls or ("v", x1, y1-1) in self.v_walls):
            return True
        
        return False

    def switch_player(self):
        """切換操控玩家"""
        self.current_player, self.other_player = self.other_player, self.current_player

    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """判斷移動是否合法"""
        x1, y1 = from_pos
        x2, y2 = to_pos
        if not (0 <= x2 < BOARD_SIZE and 0 <= y2 < BOARD_SIZE):
            return False
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        # 只檢查牆體，使用新的方法
        if self._is_blocked_by_wall(from_pos, to_pos):
            return False
        if to_pos == self.other_player.pos:
            return False
        return True

    def walk_to(self, target_code: str) -> bool:
        """走到指定的棋盤代碼位置"""
        target = pos_to_xy(target_code)
        if target in self.current_player.valid_moves:
            self.current_player.pos = target
            self.update_all_valid_moves()
            return True
        else:
            print("移動失敗：不合法的移動")
            return False

    def take_action(self, action_type: str, param: str) -> bool:
        """玩家執行一個行動：'move' 或 'wall'"""
        if action_type == 'move':
            result = self.walk_to(param)
            return result
        elif action_type == 'wall':
            if self.current_player.walls_left == 0:
                print("牆體已用完，無法放置")
                return False
            result = self.place_wall(param)
            if result:
                self.update_all_valid_moves()
            return result
        else:
            print("未知的操作類型")
            return False

    def is_valid_wall(self, wall: Wall) -> bool:
        """判斷牆體放置是否合法"""
        if wall.orientation == 'h':
            if not (0 <= wall.col < BOARD_SIZE - 1 and 1 <= wall.row <= BOARD_SIZE - 1):
                return False
            if ('h', wall.col, wall.row) in self.h_walls or ('h', wall.col+1, wall.row) in self.h_walls or ('h', wall.col-1, wall.row) in self.h_walls:
                return False
            # 橫牆與直牆可直角相接（實體規則允許），不與直牆座標互斥
        elif wall.orientation == 'v':
            if not (1 <= wall.col <= BOARD_SIZE - 1 and 0 <= wall.row < BOARD_SIZE - 1):
                return False
            if ('v', wall.col, wall.row) in self.v_walls or ('v', wall.col, wall.row+1) in self.v_walls or ('v', wall.col, wall.row-1) in self.v_walls:
                return False
            # 同上，不與橫牆互斥
        else:
            return False

        if wall.orientation == 'h':
            temp_h_walls = self.h_walls | {('h', wall.col, wall.row), ('h', wall.col+1, wall.row)}
            temp_v_walls = self.v_walls.copy()
        else:
            temp_v_walls = self.v_walls | {('v', wall.col, wall.row), ('v', wall.col, wall.row+1)}
            temp_h_walls = self.h_walls.copy()

        def bfs(start, goal_row, h_walls, v_walls):
            visited = set()
            queue = deque([start])
            while queue:
                x, y = queue.popleft()
                if y == goal_row:
                    return True
                for dx, dy in DIRECTIONS.values():
                    tx, ty = x + dx, y + dy
                    if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE):
                        continue
                    blocked = False
                    if dy == 1 and (("h", x, y+1) in h_walls or ("h", x-1, y+1) in h_walls):
                        blocked = True
                    if dy == -1 and (("h", x, y) in h_walls or ("h", x-1, y) in h_walls):
                        blocked = True
                    if dx == 1 and (("v", x+1, y) in v_walls or ("v", x+1, y-1) in v_walls):
                        blocked = True
                    if dx == -1 and (("v", x, y) in v_walls or ("v", x, y-1) in v_walls):
                        blocked = True
                    if blocked:
                        continue
                    if (tx, ty) not in visited:
                        visited.add((tx, ty))
                        queue.append((tx, ty))
            return False

        p1_ok = bfs(self.player1.pos, self.player1.goal_row, temp_h_walls, temp_v_walls)
        p2_ok = bfs(self.player2.pos, self.player2.goal_row, temp_h_walls, temp_v_walls)
        return p1_ok and p2_ok

    def place_wall(self, code: str) -> bool:
        """放置牆體，code如'ha3', 'vb5'"""
        wall = Wall(code)
        if self.current_player.walls_left == 0:
            print("牆體已用完，無法放置")
            return False
        if not self.is_valid_wall(wall):
            print("放置失敗：不合法的牆體")
            return False
        if wall.orientation == 'h':
            self.h_walls.add(('h', wall.col, wall.row))
            self.h_walls.add(('h', wall.col+1, wall.row))
        else:
            self.v_walls.add(('v', wall.col, wall.row))
            self.v_walls.add(('v', wall.col, wall.row+1))
        self.current_player.walls_left -= 1
        return True

    def calc_shortest_path_cost(self, start: Tuple[int, int], goal_row: int, h_walls=None, v_walls=None) -> int:
        """計算從start到目標行的最短路徑成本"""
        if h_walls is None:
            h_walls = self.h_walls
        if v_walls is None:
            v_walls = self.v_walls
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            (x, y), cost = queue.popleft()
            if y == goal_row:
                return cost
            for dx, dy in DIRECTIONS.values():
                tx, ty = x + dx, y + dy
                if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE):
                    continue
                blocked = False
                if dy == 1 and (("h", x, y+1) in h_walls or ("h", x-1, y+1) in h_walls):
                    blocked = True
                if dy == -1 and (("h", x, y) in h_walls or ("h", x-1, y) in h_walls):
                    blocked = True
                if dx == 1 and (("v", x+1, y) in v_walls or ("v", x+1, y-1) in v_walls):
                    blocked = True
                if dx == -1 and (("v", x, y) in v_walls or ("v", x, y-1) in v_walls):
                    blocked = True
                if blocked:
                    continue
                if (tx, ty) not in visited:
                    visited.add((tx, ty))
                    queue.append(((tx, ty), cost+1))
        return -1

    def get_distance_to_goal(self, player_name: Optional[str] = None) -> int:
        """
        取得玩家到目標的距離
        
        Args:
            player_name: 玩家名稱 ('player1' 或 'player2')，若為 None 則使用當前玩家
            
        Returns:
            int: 距離值（以最短路徑步數計算），若無法到達則返回 -1
        """
        if player_name is None:
            player = self.current_player
        elif player_name == 'player1':
            player = self.player1
        elif player_name == 'player2':
            player = self.player2
        else:
            return -1
        
        distance = self.calc_shortest_path_cost(player.pos, player.goal_row)
        return distance

    def evaluate_action_reward(self, action_type: str, param: str) -> float:
        """
        評分方法：計算 Total_Reward = ΔSelf_Progress + ΔOpponent_Obstruction
        
        根據AI規則：
        - 靠近終點(離終點更近，最小路徑成本更低)：+ 分數
        - 成功放置牆體阻礙對手(讓對手的最小路徑成本增加)：+ 分數
        - 被封鎖(自己的路徑被封鎖的最小路徑成本增加)：- 分數
        """
        self_cost_before = self.calc_shortest_path_cost(self.current_player.pos, self.current_player.goal_row)
        opp_cost_before = self.calc_shortest_path_cost(self.other_player.pos, self.other_player.goal_row)
        
        if action_type == 'move':
            target = pos_to_xy(param)
            self_pos_after = target
            opp_pos_after = self.other_player.pos
            h_walls = self.h_walls
            v_walls = self.v_walls
        elif action_type == 'wall':
            wall = Wall(param)
            if wall.orientation == 'h':
                h_walls = self.h_walls | {('h', wall.col, wall.row), ('h', wall.col+1, wall.row)}
                v_walls = self.v_walls.copy()
            else:
                v_walls = self.v_walls | {('v', wall.col, wall.row), ('v', wall.col, wall.row+1)}
                h_walls = self.h_walls.copy()
            self_pos_after = self.current_player.pos
            opp_pos_after = self.other_player.pos
        else:
            return float('-inf')
        
        self_cost_after = self.calc_shortest_path_cost(self_pos_after, self.current_player.goal_row, h_walls, v_walls)
        opp_cost_after = self.calc_shortest_path_cost(opp_pos_after, self.other_player.goal_row, h_walls, v_walls)
        
        # 檢查是否有玩家被封鎖
        if self_cost_after == -1 or opp_cost_after == -1:
            return float('-inf')
        
        delta_self = self_cost_before - self_cost_after
        delta_opp = opp_cost_after - opp_cost_before
        total_reward = delta_self + delta_opp
        return total_reward

    def get_reward_for_action(self, action_type: str, param: str) -> float:
        """
        根據AI訓練規則計算完整的獎勵值
        
        規則定義：
        - 非法動作：-1（懲罰）
        - 正常動作：基於 evaluate_action_reward() 計算的路徑成本差異
        - 獲勝動作：+100（勝利獎勵）
        
        Returns:
            float: 獎勵值
                - -1：非法動作或被封鎖
                - 其他值：基於AI進展的獎勵
                - +100：達到終點獲勝
        """
        # 檢查動作是否合法
        if not self.is_valid_action(action_type, param):
            return -1.0
        
        # 執行動作前先計算獎勵（不實際改變狀態）
        base_reward = self.evaluate_action_reward(action_type, param)
        
        # 如果基礎獎勵為負無限（代表被封鎖），返回-1
        if base_reward == float('-inf'):
            return -1.0
        
        # 檢查這個動作是否會導致獲勝
        if action_type == 'move':
            target = pos_to_xy(param)
            if target[1] == self.current_player.goal_row:
                return 100.0  # 獲勝獎勵
        
        return base_reward

    def is_valid_action(self, action_type: str, param: str) -> bool:
        """
        檢查動作是否合法
        
        Args:
            action_type: 'move' 或 'wall'
            param: 位置代碼（e.g., 'e1', 'ha3'）
            
        Returns:
            bool: 動作是否合法
        """
        if action_type == 'move':
            try:
                target = pos_to_xy(param)
                if target not in self.current_player.valid_moves:
                    return False
                return True
            except:
                return False
        elif action_type == 'wall':
            try:
                if self.current_player.walls_left == 0:
                    return False
                wall = Wall(param)
                return self.is_valid_wall(wall)
            except:
                return False
        else:
            return False

    def check_win(self) -> str:
        """檢查是否有玩家獲勝"""
        if self.current_player.pos[1] == self.current_player.goal_row:
            return self.current_player.name
        return ""

    def print_board(self):
        """
        輸出棋盤狀態（包含玩家和牆壁）
        
        格式說明：
        - 'R': Player1（紅方）
        - 'B': Player2（藍方）  
        - '──': 水平牆壁
        - '│': 垂直牆壁
        """
        x1, y1 = self.player1.pos
        x2, y2 = self.player2.pos
        
        print("\n     a   b   c   d   e   f   g   h   i")
        
        for board_y in range(BOARD_SIZE - 1, -1, -1):  # 從上到下 (9 到 1)
            # 第一部分：顯示格子和豎牆
            row_str = f" {board_y + 1}  "
            
            for board_x in range(BOARD_SIZE):
                # 顯示格子內容
                if (board_x, board_y) == (x1, y1):
                    row_str += "R "
                elif (board_x, board_y) == (x2, y2):
                    row_str += "B "
                else:
                    row_str += ". "
                
                # 顯示豎牆（在格子右側；v_walls 的 col 為欄與欄之間的 x）
                if board_x < BOARD_SIZE - 1:
                    if ('v', board_x + 1, board_y) in self.v_walls:
                        row_str += "│"
                    else:
                        row_str += " "
            
            print(row_str)
            
            # 第二部分：顯示水平牆壁（在格子下方）
            # 只在 board_y 不是最後一行時才顯示
            if board_y > 0:
                wall_row = f"    "
                for board_x in range(BOARD_SIZE):
                    # 檢查該位置下方是否有水平牆壁
                    # h_walls 中 ('h', col, row) 表示在 (col, row) 下方的牆壁
                    if ('h', board_x, board_y) in self.h_walls:
                        wall_row += "──"
                    else:
                        wall_row += "  "
                    
                    # 牆壁交叉點檢查
                    if board_x < BOARD_SIZE - 1:
                        has_h = ('h', board_x, board_y) in self.h_walls
                        has_v = (
                            ('v', board_x + 1, board_y) in self.v_walls
                            or ('v', board_x + 1, board_y - 1) in self.v_walls
                        )
                        
                        if has_h and has_v:
                            wall_row += "┼"
                        elif has_h:
                            wall_row += "─"
                        elif has_v:
                            wall_row += "│"
                        else:
                            wall_row += " "
                
                print(wall_row)
        
        # 輸出玩家信息
        print(f"\n📊 遊戲狀態:")
        print(f"  Player1 (R) - 位置: {xy_to_pos(self.player1.pos[0], self.player1.pos[1])}, 剩餘牆體: {self.player1.walls_left}")
        print(f"  Player2 (B) - 位置: {xy_to_pos(self.player2.pos[0], self.player2.pos[1])}, 剩餘牆體: {self.player2.walls_left}")
        print(f"  當前回合: {self.current_player.name}")

    def get_legal_actions_mask(self) -> List[bool]:
        """取得合法動作遮罩（209個布林值的列表）"""
        mask = [False] * 209
        
        for x, y in self.current_player.valid_moves:
            action_id = action_to_action_id('move', xy_to_pos(x, y))
            mask[action_id] = True
        
        if self.current_player.walls_left > 0:
            for row_index in range(BOARD_SIZE - 1):
                for col_index in range(BOARD_SIZE - 1):
                    wall_code = 'h' + xy_to_pos(col_index, row_index)
                    wall = Wall(wall_code)
                    if self.is_valid_wall(wall):
                        action_id = action_to_action_id('wall', wall_code)
                        mask[action_id] = True
            
            for row_index in range(BOARD_SIZE - 1):
                for col_index in range(BOARD_SIZE - 1):
                    wall_code = 'v' + xy_to_pos(col_index, row_index)
                    wall = Wall(wall_code)
                    if self.is_valid_wall(wall):
                        action_id = action_to_action_id('wall', wall_code)
                        mask[action_id] = True
        
        return mask

    def get_board_snapshot(self) -> Dict:
        """取得棋盤狀態快照（字典格式）"""
        h_walls_list = []
        v_walls_list = []
        
        seen_h_codes = set()
        for _, col, row in self.h_walls:
            if ('h', col + 1, row) in self.h_walls:
                wall_code = 'h' + chr(ord('a') + col) + str(row)
                if wall_code not in seen_h_codes:
                    seen_h_codes.add(wall_code)
                    h_walls_list.append(wall_code)

        seen_v_codes = set()
        for _, col, row in self.v_walls:
            if ('v', col, row + 1) in self.v_walls:
                wall_code = 'v' + chr(ord('a') + col - 1) + str(row + 1)
                if wall_code not in seen_v_codes:
                    seen_v_codes.add(wall_code)
                    v_walls_list.append(wall_code)
        
        winner = self.check_win()
        
        return {
            'player1_pos': xy_to_pos(self.player1.pos[0], self.player1.pos[1]),
            'player2_pos': xy_to_pos(self.player2.pos[0], self.player2.pos[1]),
            'walls_remaining': {
                'p1': self.player1.walls_left,
                'p2': self.player2.walls_left
            },
            'placed_walls': {
                'h': h_walls_list,
                'v': v_walls_list
            },
            'current_turn': self.current_player.name,
            'winner': winner
        }
