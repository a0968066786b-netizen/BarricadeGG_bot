"""
Quoridor 遊戲規則實現
依據遊戲規則.md，建立棋盤、玩家、牆體、移動與放置牆體的基本邏輯
"""
from typing import List, Tuple, Set, Dict
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

# 斜角規則定義
DIAGONAL_JUMP = {
    'up': ((-1, 1), (1, 1)),
    'down': ((-1, -1), (1, -1)),
    'left': ((-1, -1), (-1, 1)),
    'right': ((1, 1), (1, -1))
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
        col_index, row_index = pos_to_xy(param[1:])
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
    """牆體類別，包含橫向(h)與直向(v)牆體"""
    def __init__(self, code: str):
        self.orientation = code[0]  # 'h' 或 'v'
        self.col, self.row = pos_to_xy(code[1:])


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
        """回傳指定玩家目前所有可走的格子座標集合"""
        x, y = player.pos
        ox, oy = self.other_player.pos
        moves = set()
        for dir_name, (dx, dy) in DIRECTIONS.items():
            tx, ty = x + dx, y + dy
            if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE):
                continue
            if not self.is_valid_move((x, y), (tx, ty)):
                continue
            if (tx, ty) != (ox, oy):
                moves.add((tx, ty))
            else:
                jx, jy = x + STRAIGHT_JUMP[dir_name][0], y + STRAIGHT_JUMP[dir_name][1]
                if 0 <= jx < BOARD_SIZE and 0 <= jy < BOARD_SIZE:
                    if self.is_valid_move((ox, oy), (jx, jy)):
                        moves.add((jx, jy))
                        continue
                for ddx, ddy in DIAGONAL_JUMP[dir_name]:
                    sx, sy = ox + ddx, oy + ddy
                    if 0 <= sx < BOARD_SIZE and 0 <= sy < BOARD_SIZE:
                        if self.is_valid_move((x, y), (ox, oy)) and self.is_valid_move((ox, oy), (sx, sy)):
                            moves.add((sx, sy))
        return moves

    def update_all_valid_moves(self):
        """更新雙方玩家的可走格子集合"""
        self.player1.valid_moves = self.get_valid_moves(self.player1)
        self.player2.valid_moves = self.get_valid_moves(self.player2)

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
        if y2 > y1 and (("h", x1, y1+1) in self.h_walls or ("h", x1-1, y1+1) in self.h_walls):
            return False
        if y2 < y1 and (("h", x1, y1) in self.h_walls or ("h", x1-1, y1) in self.h_walls):
            return False
        if x2 > x1 and (("v", x1+1, y1) in self.v_walls or ("v", x1+1, y1-1) in self.v_walls):
            return False
        if x2 < x1 and (("v", x1, y1) in self.v_walls or ("v", x1, y1-1) in self.v_walls):
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
            if not (0 <= wall.col < BOARD_SIZE-1 and 0 <= wall.row < BOARD_SIZE-1):
                return False
            if ('h', wall.col, wall.row) in self.h_walls or ('h', wall.col+1, wall.row) in self.h_walls or ('h', wall.col-1, wall.row) in self.h_walls:
                return False
            if ('v', wall.col, wall.row) in self.v_walls or ('v', wall.col+1, wall.row) in self.v_walls:
                return False
        elif wall.orientation == 'v':
            if not (0 <= wall.col < BOARD_SIZE-1 and 0 <= wall.row < BOARD_SIZE-1):
                return False
            if ('v', wall.col, wall.row) in self.v_walls or ('v', wall.col, wall.row+1) in self.v_walls or ('v', wall.col, wall.row-1) in self.v_walls:
                return False
            if ('h', wall.col, wall.row) in self.h_walls or ('h', wall.col, wall.row+1) in self.h_walls:
                return False
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

    def evaluate_action_reward(self, action_type: str, param: str) -> float:
        """評分方法：計算 Total_Reward = ΔSelf_Progress + ΔOpponent_Obstruction"""
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
        
        if self_cost_after == -1 or opp_cost_after == -1:
            return float('-inf')
        
        delta_self = self_cost_before - self_cost_after
        delta_opp = opp_cost_after - opp_cost_before
        total_reward = delta_self + delta_opp
        return total_reward

    def check_win(self) -> str:
        """檢查是否有玩家獲勝"""
        if self.current_player.pos[1] == self.current_player.goal_row:
            return self.current_player.name
        return ""

    def print_board(self):
        """輸出棋盤狀態（簡易文字版）"""
        board = [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        x1, y1 = self.player1.pos
        x2, y2 = self.player2.pos
        board[y1][x1] = 'R'
        board[y2][x2] = 'B'
        for row in reversed(board):
            print(' '.join(row))
        print(f"Player1剩餘牆體: {self.player1.walls_left}, Player2剩餘牆體: {self.player2.walls_left}")

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
        
        seen_h_walls = set()
        for _, col, row in self.h_walls:
            if col not in seen_h_walls or row not in seen_h_walls:
                wall_code = 'h' + xy_to_pos(col, row)
                if wall_code not in h_walls_list:
                    h_walls_list.append(wall_code)
                seen_h_walls.add((col, row))
        
        seen_v_walls = set()
        for _, col, row in self.v_walls:
            if (col, row) not in seen_v_walls:
                wall_code = 'v' + xy_to_pos(col, row)
                if wall_code not in v_walls_list:
                    v_walls_list.append(wall_code)
                seen_v_walls.add((col, row))
        
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
