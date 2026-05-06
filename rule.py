"""
rule.py
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

# 跳躍規則定義（要上下左右的時候，剛好對手也在上下左右的旁邊
# (例如往上走的時候，對手在正上方、或者往下走的時候，對手在正下方)
# 且沒有牆體阻擋時可以跳過對手的方向定義，順序一樣為上下左右的跳躍）
STRAIGHT_JUMP={
    'up': (0, 2),
    'down': (0, -2),
    'left': (-2, 0),
    'right': (2, 0)
}

#斜角規則定義（當對手在上下左右的旁邊，且有牆體阻擋無法跳過對手時，可以往斜角方向走）
DIAGONAL_JUMP = {
    'up': ( (-1, 1),(1, 1) ),
    'down':( (-1, -1),(1, -1) ),
    'left': ( (-1,-1),(-1, 1) ),
    'right': ( (1, 1),(1, -1) )
}

# 棋盤座標轉換：a1 ~ i9 轉成 (x, y)，x跟y都是0~8的索引模式
def pos_to_xy(pos: str) -> Tuple[int, int]:
    col = ord(pos[0]) - ord('a')
    row = int(pos[1]) - 1
    return (col, row)

def xy_to_pos(x: int, y: int) -> str:
    return chr(ord('a') + x) + str(y + 1)

class Wall:
    """
    牆體類別，包含橫向(h)與直向(v)牆體
    牆體代碼如 ha3, vb5

    :param code: 牆體代碼
    :type code: str
    """
    def __init__(self, code: str):
        self.orientation = code[0]  # 'h' 或 'v'
        self.col,self.row=pos_to_xy(code[1:])#轉換成索引模式 

class Player:
    """
    玩家類別，紀錄位置、剩餘牆體數量、終點行

    :param name: 玩家名稱
    :type name: str
    :param start_pos: 起始位置，如 'e1' 或 'e9'
    :type start_pos: str
    :param goal_row: 終點的目標行，player1為8，player2為0
    :type goal_row: int
    """
    def __init__(self, name: str, start_pos: str, goal_row: int):
        self.name = name
        self.pos = pos_to_xy(start_pos)
        self.walls_left = MAX_WALLS
        self.goal_row = goal_row  # player1: 8, player2: 0
        self.valid_moves = set()  # 目前可走的格子 set[(x, y)]

class Board:
    """
    棋盤類別，管理棋子、牆體、移動與放置牆體
    """
    def __init__(self):
        # 初始化玩家
        self.player1 = Player('player1', 'e1', 8)  # 紅色球，目標row 9 (index 8)
        self.player2 = Player('player2', 'e9', 0)  # 藍色球，目標row 1 (index 0)
        #操控玩家與對手玩家設定，current_player為當前操控玩家，other_player為對手玩家
        self.current_player = self.player1
        self.other_player = self.player2
        self.h_walls = set()  # 橫向牆體集合
        self.v_walls = set()  # 直向牆體集合
        # 初始化雙方可走格子
        self.update_all_valid_moves()

    def get_valid_moves(self, player: Player) -> set:
        """
        [公開] 回傳指定玩家目前所有可走的格子座標集合（set[(x, y)]），含跳躍、斜角等規則
        """
        x, y = player.pos
        ox, oy = self.other_player.pos
        moves = set()
        for dir_name, (dx, dy) in DIRECTIONS.items():
            tx, ty = x + dx, y + dy
            # 先檢查是否在棋盤內
            if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE):
                continue
            # 如果該方向有牆，不能走
            if not self.is_valid_move((x, y), (tx, ty)):
                continue
            # 如果該方向不是對手，直接加入
            if (tx, ty) != (ox, oy):
                moves.add((tx, ty))
            else:
                # 遇到對手，判斷能否直跳
                jx, jy = x + STRAIGHT_JUMP[dir_name][0], y + STRAIGHT_JUMP[dir_name][1]
                if 0 <= jx < BOARD_SIZE and 0 <= jy < BOARD_SIZE:
                    # 檢查對手->直跳方向有無牆
                    if self.is_valid_move((ox, oy), (jx, jy)):
                        moves.add((jx, jy))
                        continue
                # 不能直跳，檢查斜角
                for ddx, ddy in DIAGONAL_JUMP[dir_name]:
                    sx, sy = ox + ddx, oy + ddy
                    if 0 <= sx < BOARD_SIZE and 0 <= sy < BOARD_SIZE:
                        # 斜角要檢查：自己到對手、對手到斜角都不能有牆
                        if self.is_valid_move((x, y), (ox, oy)) and self.is_valid_move((ox, oy), (sx, sy)):
                            moves.add((sx, sy))
        return moves

    def update_all_valid_moves(self):
        """
        [私有] 更新雙方玩家的可走格子集合
        """
        self.player1.valid_moves = self.get_valid_moves(self.player1)
        self.player2.valid_moves = self.get_valid_moves(self.player2)

    def switch_player(self):
        """
        [公開] 切換操控玩家
        """
        self.current_player, self.other_player = self.other_player, self.current_player

    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        [私有] 判斷移動是否合法（不穿牆、不出界、考慮對手跳躍）
        這裡僅實作基本上下左右移動與牆體阻擋，進階跳躍可再補充
        :param from_pos: 起始位置 (x, y)
        :type from_pos: Tuple[int, int]
        :param to_pos: 目的地位置 (x, y)
        :type to_pos: Tuple[int, int]
        """
        x1, y1 = from_pos
        x2, y2 = to_pos
        # 檢查是否在棋盤內
        if not (0 <= x2 < BOARD_SIZE and 0 <= y2 < BOARD_SIZE):
            return False
        # 檢查是否移動一格
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        # 修正版：牆體阻擋需考慮一個牆體會阻擋兩個格子
        # 往上
        if y2 > y1 and (("h", x1, y1+1) in self.h_walls or ("h", x1-1, y1+1) in self.h_walls):
            return False
        # 往下
        if y2 < y1 and (("h", x1, y1) in self.h_walls or ("h", x1-1, y1) in self.h_walls):
            return False
        # 往右
        if x2 > x1 and (("v", x1+1, y1) in self.v_walls or ("v", x1+1, y1-1) in self.v_walls):
            return False
        # 往左
        if x2 < x1 and (("v", x1, y1) in self.v_walls or ("v", x1, y1-1) in self.v_walls):
            return False
        # 檢查是否有對手在目的地(這邊需要做跳躍判斷，暫時先禁止移動到對手位置)
        if to_pos == self.other_player.pos:
            return False
        return True


    def walk_to(self, target_code: str) -> bool:
        """
        [公開] 走到指定的棋盤代碼位置（如 'b3'），合法性直接查表（self.current_player.valid_moves）
        """
        target = pos_to_xy(target_code)
        if target in self.current_player.valid_moves:
            self.current_player.pos = target
            self.update_all_valid_moves()
            return True
        else:
            print("移動失敗：不合法的移動")
            return False

    def take_action(self, action_type: str, param: str) -> bool:
        """
        [公開] 玩家執行一個行動：'move' 或 'wall'
        - action_type: 'move' 表示移動，param 為目標格代碼（如 'b3'）
        - action_type: 'wall' 表示放牆，param 為牆體代碼（如 'ha3'）
        每回合只能執行一種操作
        """
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
        """
        [私有] 判斷牆體放置是否合法（不重疊、不交叉、不出界、不封死路徑）
        """
        # 出界檢查
        if wall.orientation == 'h':
            if not (0 <= wall.col < BOARD_SIZE-1 and 0 <= wall.row < BOARD_SIZE-1):
                return False
            # 橫牆重疊檢查：同一橫牆或左右相鄰橫牆
            if ('h', wall.col, wall.row) in self.h_walls or ('h', wall.col+1, wall.row) in self.h_walls or ('h', wall.col-1, wall.row) in self.h_walls:
                return False
            # 橫牆交叉檢查：同一交叉點不能有直牆
            if ('v', wall.col, wall.row) in self.v_walls or ('v', wall.col+1, wall.row) in self.v_walls:
                return False
        elif wall.orientation == 'v':
            if not (0 <= wall.col < BOARD_SIZE-1 and 0 <= wall.row < BOARD_SIZE-1):
                return False
            # 直牆重疊檢查：同一直牆或上下相鄰直牆
            if ('v', wall.col, wall.row) in self.v_walls or ('v', wall.col, wall.row+1) in self.v_walls or ('v', wall.col, wall.row-1) in self.v_walls:
                return False
            # 直牆交叉檢查：同一交叉點不能有橫牆
            if ('h', wall.col, wall.row) in self.h_walls or ('h', wall.col, wall.row+1) in self.h_walls:
                return False
        else:
            return False
        # 檢查是否封死任一玩家路徑（BFS）
        # 1. 臨時加入牆體
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
                    # 檢查牆體阻擋（複製 is_valid_move 牆體部分）
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

        # 檢查雙方玩家都能到終點
        p1_ok = bfs(self.player1.pos, self.player1.goal_row, temp_h_walls, temp_v_walls)
        p2_ok = bfs(self.player2.pos, self.player2.goal_row, temp_h_walls, temp_v_walls)
        return p1_ok and p2_ok

    def place_wall(self, code: str) -> bool:
        """
        [私有] 放置牆體，code如'ha3', 'vb5'
        回傳是否放置成功
        """
        wall = Wall(code)
        if self.current_player.walls_left == 0:
            print("牆體已用完，無法放置")
            return False
        if not self.is_valid_wall(wall):
            print("放置失敗：不合法的牆體")
            return False
        if wall.orientation == 'h':
            # 橫牆阻擋兩個交叉點 (col, row) 與 (col+1, row)
            self.h_walls.add(('h', wall.col, wall.row))
            self.h_walls.add(('h', wall.col+1, wall.row))
        else:
            # 直牆阻擋兩個交叉點 (col, row) 與 (col, row+1)
            self.v_walls.add(('v', wall.col, wall.row))
            self.v_walls.add(('v', wall.col, wall.row+1))
        self.current_player.walls_left -= 1
        return True

    def calc_shortest_path_cost(self, start: Tuple[int, int], goal_row: int, h_walls=None, v_walls=None) -> int:
        """
        計算從start到目標行的最短路徑成本（步數），若無法到達則回傳-1。
        可指定牆體集合（用於模擬放牆後的情境）。
        """
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
                # 牆體阻擋
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
        return -1  # 無法到達

    def evaluate_action_reward(self, action_type: str, param: str) -> float:
        """
        評分方法：計算 Total_Reward = ΔSelf_Progress + ΔOpponent_Obstruction
        - action_type: 'move' 或 'wall'
        - param: 目標格代碼或牆體代碼
        """
        # 取得雙方現有路徑成本
        self_cost_before = self.calc_shortest_path_cost(self.current_player.pos, self.current_player.goal_row)
        opp_cost_before = self.calc_shortest_path_cost(self.other_player.pos, self.other_player.goal_row)
        # 模擬行動後的狀態
        if action_type == 'move':
            # 模擬移動
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
            return float('-inf')  # 非法行動
        # 行動後的路徑成本
        self_cost_after = self.calc_shortest_path_cost(self_pos_after, self.current_player.goal_row, h_walls, v_walls)
        opp_cost_after = self.calc_shortest_path_cost(opp_pos_after, self.other_player.goal_row, h_walls, v_walls)
        # 若有一方被堵死，則此行動無效
        if self_cost_after == -1 or opp_cost_after == -1:
            return float('-inf')
        # Reward 計算
        delta_self = self_cost_before - self_cost_after  # 自己進步越多越好
        delta_opp = opp_cost_after - opp_cost_before    # 對手越難走越好
        total_reward = delta_self + delta_opp
        return total_reward

    def check_win(self) -> str:
        """
        [公開] 檢查是否有玩家獲勝
        """
        if self.current_player.pos[1] == self.current_player.goal_row:
            return self.current_player.name
        return ""

    def print_board(self):
        """
        [公開] 輸出棋盤狀態（簡易文字版）
        """
        board = [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        x1, y1 = self.player1.pos
        x2, y2 = self.player2.pos
        board[y1][x1] = 'R'
        board[y2][x2] = 'B'
        for row in reversed(board):
            print(' '.join(row))
        print(f"Player1剩餘牆體: {self.player1.walls_left}, Player2剩餘牆體: {self.player2.walls_left}")

# 範例遊戲流程
if __name__ == "__main__":
    board = Board()
    board.print_board()
    #流程為單次玩家操控，每次行動後檢查勝利條件並切換玩家
    # (實務上，如果動作失敗了，其實是要回歸該玩家繼續操作，直到操作成功才能換下一個人操作)
    # 玩家1移動到 e2
    board.take_action('move', 'e2')
    board.check_win()
    board.print_board()
    
    # 換玩家
    board.switch_player()
    # 玩家2移動到 e8
    board.take_action('move', 'e8')
    board.check_win()
    board.print_board()

    # 換玩家
    board.switch_player()
    # 玩家1放牆 ha2
    board.take_action('wall', 'ha2')
    board.check_win()
    board.print_board()    