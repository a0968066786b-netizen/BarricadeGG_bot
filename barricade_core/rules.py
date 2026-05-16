"""
Quoridor 完整規則引擎 - 自訂遊戲規則版本

核心邏輯實現：
- Grid.GetNeighboursWithWalls → _legal_cs_neighbors（單步牆體檢查）
- Game.ValidateMove → get_valid_moves（直線跳躍 + 斜角跳躍）
- Game.ValidateWall → is_valid_wall（路徑驗證）
- Algorithm.FindPathToRow → _astar_cost_to_row_cs（A* 開集合選擇）

遊戲規則：
✅ 直線跳躍：對手相鄰時，沿直線跳過（2格）
✅ 斜角跳躍：直線被阻時，側向跳躍到對手旁邊（1格）
✅ 牆體多米諾形式：2格連續（橫或豎）
✅ 牆體放置：允許邊邊碰到的牆體（U 型等）- 只禁止完全重複
✅ 路徑驗證：A* 確保雙方都能到達目標
✅ 邊界檢查：牆體不能超出棋盤範圍

修改說明：
- 禁用了傳統 Quoridor 的「十字交叉」檢查
- 允許水平牆和垂直牆在邊邊碰到的位置共存（如 U 型牆體）
- 只要不是完全重複的牆體就允許放置
- 路徑驗證確保遊戲的可玩性

座標系統：
- Python (x,y)：a1 = (0,0)，x向右，y向上
- C# (row,col)：row 0 在頂端，向下遞增
"""
from __future__ import annotations

from typing import List, Tuple, Set, Dict, Optional

# 棋盤大小（與 C# Grid(gridSize[0], gridSize[1]) 一致）
BOARD_SIZE = 9

# 牆體數量（與 C# Player 初始 WallCount 一致）
MAX_WALLS = 10

# Python 棋盤 (x,y)：x 向右、y 向上（a1 為 x=0,y=0）
# C# (row,col)：row=0 在棋盤頂端、向下遞增（與 Program.cs 玩家起點一致）
DIRECTIONS_PY = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# 舊版 API 相容別名（邏輯已改為 C# 對齊，不再使用斜跳常數）
DIRECTIONS = DIRECTIONS_PY
STRAIGHT_JUMP = {
    "up": (0, 2),
    "down": (0, -2),
    "left": (-2, 0),
    "right": (2, 0),
}
DIAGONAL_JUMP = {
    "up": ((-1, 0), (1, 0)),
    "down": ((-1, 0), (1, 0)),
    "left": ((0, -1), (0, 1)),
    "right": ((0, -1), (0, 1)),
}


def pos_to_xy(pos: str) -> Tuple[int, int]:
    """棋盤座標轉換：a1 ~ i9 轉成 (x, y)"""
    col = ord(pos[0]) - ord("a")
    row = int(pos[1:]) - 1
    return (col, row)


def xy_to_pos(x: int, y: int) -> str:
    """座標轉換回棋盤代碼"""
    return chr(ord("a") + x) + str(y + 1)


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
        return ("move", pos)
    if 81 <= action_id <= 144:
        wall_id = action_id - 81
        row_index = wall_id // 8
        col_index = wall_id % 8
        wall_code = "h" + xy_to_pos(col_index, row_index)
        return ("wall", wall_code)
    if 145 <= action_id <= 208:
        wall_id = action_id - 145
        row_index = wall_id // 8
        col_index = wall_id % 8
        wall_code = "v" + xy_to_pos(col_index, row_index)
        return ("wall", wall_code)
    raise ValueError(f"無效的動作 ID: {action_id}")


def action_to_action_id(action_type: str, param: str) -> int:
    """將行動指令轉換為動作 ID（0-208）"""
    if action_type == "move":
        col_index, row_index = pos_to_xy(param)
        return row_index * 9 + col_index
    if action_type == "wall":
        orientation = param[0]
        col_index = ord(param[1]) - ord("a")
        row_index = int(param[2:]) - 1
        if orientation == "h":
            return 81 + (row_index * 8 + col_index)
        if orientation == "v":
            return 145 + (row_index * 8 + col_index)
        raise ValueError(f"無效的牆體方向: {orientation}")
    raise ValueError(f"無效的行動類型: {action_type}")


class Wall:
    """牆體代碼解析（維持既有字串格式，供 place_wall / mask 使用）"""

    def __init__(self, code: str):
        self.code = code  # 存儲原始代碼
        self.orientation = code[0]
        letter = ord(code[1]) - ord("a")
        groove = int(code[2:])
        if not (1 <= groove <= BOARD_SIZE - 1):
            raise ValueError(f"無效的牆體溝槽編號: {code!r}")
        if self.orientation == "h":
            self.col = letter
            self.row = groove
        elif self.orientation == "v":
            self.col = letter + 1
            self.row = groove - 1
        else:
            raise ValueError(f"無效的牆體方向: {self.orientation!r}")


class Player:
    """玩家：位置為 Python (x,y)，goal_row 為 Python y 目標列"""

    def __init__(self, name: str, start_pos: str, goal_row: int):
        self.name = name
        self.pos = pos_to_xy(start_pos)
        self.walls_left = MAX_WALLS
        self.goal_row = goal_row
        self.valid_moves: Set[Tuple[int, int]] = set()


def _alloc_horizontal() -> List[List[bool]]:
    return [[False] * BOARD_SIZE for _ in range(BOARD_SIZE - 1)]


def _alloc_vertical() -> List[List[bool]]:
    return [[False] * (BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]


def _clone_hv(h: List[List[bool]], v: List[List[bool]]):
    return [row[:] for row in h], [row[:] for row in v]


class Board:
    """
    內部牆體：horizontal[r][c]、vertical[r][c] 索引語意與 C# bool[,] 相同。
    """

    def __init__(self):
        self.player1 = Player("player1", "e1", BOARD_SIZE - 1)
        self.player2 = Player("player2", "e9", 0)
        self.current_player = self.player1
        self.other_player = self.player2
        self._horizontal = _alloc_horizontal()
        self._vertical = _alloc_vertical()
        self.h_walls: Set[Tuple[str, int, int]] = set()
        self.v_walls: Set[Tuple[str, int, int]] = set()
        # 交叉點追蹤：存儲已被水平或垂直牆佔據的交叉點（例如 "d4"）
        self._occupied_junctions: Set[str] = set()
        self._sync_wall_tuple_sets()
        self.update_all_valid_moves()

    # --- 座標：Python (x,y) <-> C# (row,col)；row=0 在頂、對應 Python 最大 y ---

    def _get_junction_from_code(self, wall_code: str) -> str:
        """
        從牆體代碼提取交叉點座標
        例如：hd4 → d4，vd4 → d4
        牆體代碼格式：[h|v][a-i][1-9]
        交叉點：[a-i][1-9]（後兩部分）
        """
        return wall_code[1:]  # 去掉方向字母，保留列和行號

    @staticmethod
    def _py_to_cs(x: int, y: int) -> Tuple[int, int]:
        return (BOARD_SIZE - 1 - y, x)

    @staticmethod
    def _cs_to_py(row: int, col: int) -> Tuple[int, int]:
        return (col, BOARD_SIZE - 1 - row)

    def _goal_row_cs(self, player: Player) -> int:
        """Python goal_row（y）轉成 C# 目標列 row"""
        return BOARD_SIZE - 1 - player.goal_row

    def _sync_wall_tuple_sets(self) -> None:
        """由 C# 對齊陣列產生舊版 h_walls / v_walls 集合（供列印／快照），同時更新交叉點記錄"""
        self.h_walls.clear()
        self.v_walls.clear()
        self._occupied_junctions.clear()
        
        # 水平牆：h[cs_r][cs_c] 對應牆體代碼 h{col}{row}
        # 其中 cs_r = BOARD_SIZE - 1 - row，cs_c = col
        for cs_r in range(BOARD_SIZE - 1):
            for cs_c in range(BOARD_SIZE):
                if self._horizontal[cs_r][cs_c]:
                    row = BOARD_SIZE - 1 - cs_r
                    col = cs_c
                    self.h_walls.add(("h", col, row))
                    # 交叉點座標 = {col}{row}
                    col_char = chr(ord('a') + col)
                    junction = col_char + str(row)
                    self._occupied_junctions.add(junction)
        
        # 垂直牆：v[cs_r][cs_c] 對應牆體代碼 v{col}{row}
        # 其中 cs_r = row，cs_c + 1 = col，所以 col = cs_c + 1，row = cs_r
        # 但牆體代碼中的 groove = row + 1，所以代碼是 v{col}{groove}
        for cs_r in range(BOARD_SIZE):
            for cs_c in range(BOARD_SIZE - 1):
                if self._vertical[cs_r][cs_c]:
                    row = cs_r
                    col = cs_c + 1
                    groove = row + 1  # vXN 中的 N
                    self.v_walls.add(("v", col, row))
                    # 交叉點座標 = {col-1}{groove}
                    col_char = chr(ord('a') + col - 1)
                    junction = col_char + str(groove)
                    self._occupied_junctions.add(junction)

    # --- Grid.GetNeighboursWithWalls：以 C# 索引判定單步是否可通行 ---

    @staticmethod
    def _cs_step_blocked(
        row: int,
        col: int,
        nrow: int,
        ncol: int,
        horizontal: List[List[bool]],
        vertical: List[List[bool]],
    ) -> bool:
        """相鄰 C# 格子 (row,col)->(nrow,ncol) 是否被牆擋（僅一步）。"""
        if nrow == row - 1 and col == ncol:
            return row <= 0 or horizontal[row - 1][col]
        if nrow == row + 1 and col == ncol:
            return row >= BOARD_SIZE - 1 or horizontal[row][col]
        if ncol == col - 1 and row == nrow:
            return col <= 0 or vertical[row][col - 1]
        if ncol == col + 1 and row == nrow:
            return col >= BOARD_SIZE - 1 or vertical[row][col]
        return True

    def _legal_cs_neighbors(
        self,
        row: int,
        col: int,
        horizontal: List[List[bool]],
        vertical: List[List[bool]],
    ) -> List[Tuple[int, int]]:
        """對應 Grid.GetNeighboursWithWalls 回傳的鄰居列表（C# 座標）。"""
        out: List[Tuple[int, int]] = []
        if row > 0 and not horizontal[row - 1][col]:
            out.append((row - 1, col))
        if row < BOARD_SIZE - 1 and not horizontal[row][col]:
            out.append((row + 1, col))
        if col > 0 and not vertical[row][col - 1]:
            out.append((row, col - 1))
        if col < BOARD_SIZE - 1 and not vertical[row][col]:
            out.append((row, col + 1))
        return out

    # --- Game.ValidateMove：對手格直跳；直跳被擋則該方向無著點（無斜跳）---

    def _jump_blocked_after_opponent(
        self,
        move0: int,
        move1: int,
        opp_row: int,
        opp_col: int,
        horizontal: List[List[bool]],
        vertical: List[List[bool]],
    ) -> bool:
        """
        已確認 (newRow,newCol)==對手；對應 C# ValidateMove 中
        「Moving up/down/left/right」四段牆檢查（對手後方直線）。
        """
        new_row, new_col = opp_row, opp_col
        if move0 == -1:
            return new_row - 1 >= 0 and horizontal[new_row - 1][new_col]
        if move0 == 1:
            return new_row < BOARD_SIZE - 1 and horizontal[new_row][new_col]
        if move1 == -1:
            return new_col - 1 >= 0 and vertical[new_row][new_col - 1]
        if move1 == 1:
            return new_col < BOARD_SIZE - 1 and vertical[new_row][new_col]
        return True

    def get_valid_moves(self, player: Player) -> Set[Tuple[int, int]]:
        """
        取得玩家合法移動位置集合
        包括：直線移動、直線跳躍、斜角跳躍
        """
        horizontal, vertical = self._horizontal, self._vertical
        row, col = self._py_to_cs(*player.pos)
        orow, ocol = self._py_to_cs(*self.other_player.pos)
        moves: Set[Tuple[int, int]] = set()

        for move0, move1 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nrow, ncol = row + move0, col + move1
            if not (0 <= nrow < BOARD_SIZE and 0 <= ncol < BOARD_SIZE):
                continue
            if self._cs_step_blocked(row, col, nrow, ncol, horizontal, vertical):
                continue

            if (nrow, ncol) == (orow, ocol):
                # 對手相鄰 - 嘗試直線跳躍
                jr, jc = row + 2 * move0, col + 2 * move1
                if not (0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE):
                    # 直線跳躍超出邊界 - 嘗試斜角跳躍
                    self._add_diagonal_jumps(moves, row, col, move0, move1, orow, ocol, horizontal, vertical)
                    continue
                
                if self._jump_blocked_after_opponent(
                    move0, move1, nrow, ncol, horizontal, vertical
                ):
                    # 直線跳躍被對手後方的牆阻擋 - 嘗試斜角跳躍
                    self._add_diagonal_jumps(moves, row, col, move0, move1, orow, ocol, horizontal, vertical)
                    continue
                
                if self._cs_step_blocked(orow, ocol, jr, jc, horizontal, vertical):
                    # 對手後方有牆 - 嘗試斜角跳躍
                    self._add_diagonal_jumps(moves, row, col, move0, move1, orow, ocol, horizontal, vertical)
                    continue
                
                # 直線跳躍合法
                moves.add(self._cs_to_py(jr, jc))
            else:
                # 空格 - 正常移動
                moves.add(self._cs_to_py(nrow, ncol))

        cur = player.pos
        moves.discard(cur)
        return moves

    def _add_diagonal_jumps(
        self,
        moves: Set[Tuple[int, int]],
        row: int,
        col: int,
        move0: int,
        move1: int,
        orow: int,
        ocol: int,
        horizontal: List[List[bool]],
        vertical: List[List[bool]],
    ) -> None:
        """
        官方Quoridor規則：當直線跳躍被阻時，可側向跳躍到對手旁邊
        - 上下移動 (move0≠0) → 側向是左右 (move1)
        - 左右移動 (move1≠0) → 側向是上下 (move0)
        檢查流程：對手旁邊位置必須在邊界內且從對手位置通往該位置
        """
        if move0 != 0:  # 上下移動，側向是左右
            for side1 in (-1, 1):
                sr, sc = orow, ocol + side1
                if 0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE:
                    # 檢查從對手位置到側向位置是否通暢
                    if not self._cs_step_blocked(orow, ocol, sr, sc, horizontal, vertical):
                        moves.add(self._cs_to_py(sr, sc))
        else:  # 左右移動，側向是上下
            for side0 in (-1, 1):
                sr, sc = orow + side0, ocol
                if 0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE:
                    # 檢查從對手位置到側向位置是否通暢
                    if not self._cs_step_blocked(orow, ocol, sr, sc, horizontal, vertical):
                        moves.add(self._cs_to_py(sr, sc))


    def update_all_valid_moves(self) -> None:
        self.player1.valid_moves = self.get_valid_moves(self.player1)
        self.player2.valid_moves = self.get_valid_moves(self.player2)

    def _is_blocked_by_wall(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """與 Grid 邏輯一致（相鄰一步）。"""
        x1, y1 = from_pos
        x2, y2 = to_pos
        r1, c1 = self._py_to_cs(x1, y1)
        r2, c2 = self._py_to_cs(x2, y2)
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return True
        return self._cs_step_blocked(r1, c1, r2, c2, self._horizontal, self._vertical)

    def switch_player(self) -> None:
        self.current_player, self.other_player = self.other_player, self.current_player

    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        x1, y1 = from_pos
        x2, y2 = to_pos
        if not (0 <= x2 < BOARD_SIZE and 0 <= y2 < BOARD_SIZE):
            return False
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        if self._is_blocked_by_wall(from_pos, to_pos):
            return False
        if to_pos == self.other_player.pos:
            return False
        return True

    def walk_to(self, target_code: str) -> bool:
        target = pos_to_xy(target_code)
        if target in self.current_player.valid_moves:
            self.current_player.pos = target
            self.update_all_valid_moves()
            return True
        print("移動失敗：不合法的移動")
        return False

    def take_action(self, action_type: str, param: str) -> bool:
        if action_type == "move":
            return self.walk_to(param)
        if action_type == "wall":
            if self.current_player.walls_left == 0:
                print("牆體已用完，無法放置")
                return False
            ok = self.place_wall(param)
            if ok:
                self.update_all_valid_moves()
            return ok
        print("未知的操作類型")
        return False

    # --- 牆體驗證：邊界檢查、重複檢查、路徑驗證 ---

    @staticmethod
    def _wall_to_cs_indices(wall: Wall) -> Tuple[str, int, int]:
        """回傳 ('h', hr, hc0) 或 ('v', vr, vc)，與 C# prevRow/prevCol 一致。"""
        if wall.orientation == "h":
            hr = BOARD_SIZE - 1 - wall.row
            hc = wall.col
            return ("h", hr, hc)
        vr = wall.row
        vc = wall.col - 1
        return ("v", vr, vc)

    @staticmethod
    def _junction_plus_exists(h: List[List[bool]], v: List[List[bool]]) -> bool:
        """
        牆體中心點互斥檢查已禁用
        
        原始遊戲設計：允許邊邊碰到的牆體（U 型等）
        只要水平牆和垂直牆不占據完全相同的格子就允許放置
        路徑驗證會確保遊戲的可玩性
        
        返回 False 表示不存在十字交叉，允許放置
        """
        return False

    def _path_exists_cs(
        self,
        start_py: Tuple[int, int],
        goal_py_row: int,
        horizontal: List[List[bool]],
        vertical: List[List[bool]],
    ) -> bool:
        cost = self._astar_cost_to_row_cs(start_py, goal_py_row, horizontal, vertical)
        return cost >= 0

    def _astar_cost_to_row_cs(
        self,
        start_py: Tuple[int, int],
        goal_py_row: int,
        horizontal: List[List[bool]],
        vertical: List[List[bool]],
    ) -> int:
        """
        Algorithm.FindPathToRow：若可達回傳最短步數成本；否則 -1。
        開集合挑選：最小 F，同 F 取較小 H（與 C# 雙層迴圈一致）。
        """
        target_row = BOARD_SIZE - 1 - goal_py_row
        sr, sc = self._py_to_cs(*start_py)

        rows = BOARD_SIZE
        g_cost = [[10**9] * BOARD_SIZE for _ in range(rows)]
        h_cost = [[0] * BOARD_SIZE for _ in range(rows)]
        closed = [[False] * BOARD_SIZE for _ in range(rows)]
        g_cost[sr][sc] = 0
        h_cost[sr][sc] = abs(sr - target_row)
        open_nodes: List[Tuple[int, int]] = [(sr, sc)]

        while open_nodes:
            best_i = 0
            cr, cc = open_nodes[0]
            best_f = g_cost[cr][cc] + h_cost[cr][cc]
            best_h = h_cost[cr][cc]
            for i in range(1, len(open_nodes)):
                r, c = open_nodes[i]
                f = g_cost[r][c] + h_cost[r][c]
                hval = h_cost[r][c]
                if f < best_f or (f == best_f and hval < best_h):
                    best_f = f
                    best_h = hval
                    best_i = i

            row, col = open_nodes.pop(best_i)
            if closed[row][col]:
                continue
            closed[row][col] = True

            if row == target_row:
                return g_cost[row][col]

            for nrow, ncol in self._legal_cs_neighbors(row, col, horizontal, vertical):
                if closed[nrow][ncol]:
                    continue
                tentative = g_cost[row][col] + 1
                if tentative < g_cost[nrow][ncol] or (nrow, ncol) not in open_nodes:
                    g_cost[nrow][ncol] = tentative
                    h_cost[nrow][ncol] = abs(nrow - target_row)
                    if (nrow, ncol) not in open_nodes:
                        open_nodes.append((nrow, ncol))

        return -1

    def is_valid_wall(self, wall: Wall) -> bool:
        horizontal, vertical = self._horizontal, self._vertical
        kind, pr, pc = self._wall_to_cs_indices(wall)

        if kind == "h":
            max_row = BOARD_SIZE - 1
            max_col = BOARD_SIZE
            if pr < 0 or pr >= max_row or pc < 0 or pc >= max_col - 1:
                return False
            if horizontal[pr][pc] or horizontal[pr][pc + 1]:
                return False
        else:
            max_row = BOARD_SIZE
            max_col = BOARD_SIZE - 1
            if pr < 0 or pr >= max_row - 1 or pc < 0 or pc >= max_col:
                return False
            if vertical[pr][pc] or vertical[pr + 1][pc]:
                return False

        # 檢查十字交叉：提取牆體的交叉點，看是否已被佔據
        junction = self._get_junction_from_code(wall.code)
        if junction in self._occupied_junctions:
            # 該交叉點已被水平或垂直牆佔據，禁止放置
            return False

        th, tv = _clone_hv(horizontal, vertical)
        if kind == "h":
            th[pr][pc] = True
            th[pr][pc + 1] = True
        else:
            tv[pr][pc] = True
            tv[pr + 1][pc] = True

        if not (
            self._path_exists_cs(self.player1.pos, self.player1.goal_row, th, tv)
            and self._path_exists_cs(self.player2.pos, self.player2.goal_row, th, tv)
        ):
            return False

        return True

    def place_wall(self, code: str) -> bool:
        wall = Wall(code)
        if self.current_player.walls_left == 0:
            print("牆體已用完，無法放置")
            return False
        if not self.is_valid_wall(wall):
            print("放置失敗：不合法的牆體")
            return False
        kind, pr, pc = self._wall_to_cs_indices(wall)
        if kind == "h":
            self._horizontal[pr][pc] = True
            self._horizontal[pr][pc + 1] = True
        else:
            self._vertical[pr][pc] = True
            self._vertical[pr + 1][pc] = True
        
        # 更新交叉點記錄
        junction = self._get_junction_from_code(code)
        self._occupied_junctions.add(junction)
        
        self.current_player.walls_left -= 1
        self._sync_wall_tuple_sets()
        self.update_all_valid_moves()
        return True

    def calc_shortest_path_cost(
        self,
        start: Tuple[int, int],
        goal_row: int,
        h_walls=None,
        v_walls=None,
    ) -> int:
        """
        最短路徑成本：與 Algorithm.FindPathToRow 相同之 A*（步數）。
        若傳入 h_walls / v_walls 任一集合，則僅由集合重建臨時牆陣列（供 evaluate 使用）。
        """
        if h_walls is None and v_walls is None:
            return self._astar_cost_to_row_cs(start, goal_row, self._horizontal, self._vertical)

        th = _alloc_horizontal()
        tv = _alloc_vertical()
        if h_walls is not None:
            for t, c, r in h_walls:
                if t == "h":
                    th[BOARD_SIZE - 1 - r][c] = True
        if v_walls is not None:
            for t, col, rpy in v_walls:
                if t == "v":
                    tv[rpy][col - 1] = True
        return self._astar_cost_to_row_cs(start, goal_row, th, tv)

    def get_distance_to_goal(self, player_name: Optional[str] = None) -> int:
        if player_name is None:
            player = self.current_player
        elif player_name == "player1":
            player = self.player1
        elif player_name == "player2":
            player = self.player2
        else:
            return -1
        return self.calc_shortest_path_cost(player.pos, player.goal_row)

    def evaluate_action_reward(self, action_type: str, param: str) -> float:
        self_cost_before = self.calc_shortest_path_cost(
            self.current_player.pos, self.current_player.goal_row
        )
        opp_cost_before = self.calc_shortest_path_cost(
            self.other_player.pos, self.other_player.goal_row
        )

        if action_type == "move":
            target = pos_to_xy(param)
            self_pos_after = target
            opp_pos_after = self.other_player.pos
            h_use = None
            v_use = None
        elif action_type == "wall":
            wall = Wall(param)
            if wall.orientation == "h":
                h_use = self.h_walls | {
                    ("h", wall.col, wall.row),
                    ("h", wall.col + 1, wall.row),
                }
                v_use = set(self.v_walls)
            else:
                v_use = self.v_walls | {
                    ("v", wall.col, wall.row),
                    ("v", wall.col, wall.row + 1),
                }
                h_use = set(self.h_walls)
            self_pos_after = self.current_player.pos
            opp_pos_after = self.other_player.pos
        else:
            return float("-inf")

        self_cost_after = self.calc_shortest_path_cost(
            self_pos_after, self.current_player.goal_row, h_use, v_use
        )
        opp_cost_after = self.calc_shortest_path_cost(
            opp_pos_after, self.other_player.goal_row, h_use, v_use
        )

        if self_cost_after == -1 or opp_cost_after == -1:
            return float("-inf")

        delta_self = self_cost_before - self_cost_after
        delta_opp = opp_cost_after - opp_cost_before
        return delta_self + delta_opp

    def get_reward_for_action(self, action_type: str, param: str) -> float:
        if not self.is_valid_action(action_type, param):
            return -1.0
        base_reward = self.evaluate_action_reward(action_type, param)
        if base_reward == float("-inf"):
            return -1.0
        if action_type == "move":
            target = pos_to_xy(param)
            if target[1] == self.current_player.goal_row:
                return 100.0
        return base_reward

    def is_valid_action(self, action_type: str, param: str) -> bool:
        if action_type == "move":
            try:
                target = pos_to_xy(param)
                return target in self.current_player.valid_moves
            except (ValueError, IndexError, TypeError):
                return False
        if action_type == "wall":
            try:
                if self.current_player.walls_left == 0:
                    return False
                return self.is_valid_wall(Wall(param))
            except (ValueError, IndexError, TypeError):
                return False
        return False

    def check_win(self) -> str:
        if self.current_player.pos[1] == self.current_player.goal_row:
            return self.current_player.name
        return ""

    def print_board(self) -> None:
        x1, y1 = self.player1.pos
        x2, y2 = self.player2.pos

        print("\n     a   b   c   d   e   f   g   h   i")

        for board_y in range(BOARD_SIZE - 1, -1, -1):
            row_str = f" {board_y + 1}  "
            for board_x in range(BOARD_SIZE):
                if (board_x, board_y) == (x1, y1):
                    row_str += "R "
                elif (board_x, board_y) == (x2, y2):
                    row_str += "B "
                else:
                    row_str += ". "
                if board_x < BOARD_SIZE - 1:
                    if ("v", board_x + 1, board_y) in self.v_walls:
                        row_str += "│"
                    else:
                        row_str += " "
            print(row_str)

            if board_y > 0:
                wall_row = "    "
                for board_x in range(BOARD_SIZE):
                    if ("h", board_x, board_y) in self.h_walls:
                        wall_row += "──"
                    else:
                        wall_row += "  "
                    if board_x < BOARD_SIZE - 1:
                        has_h = ("h", board_x, board_y) in self.h_walls
                        has_v = ("v", board_x + 1, board_y) in self.v_walls or (
                            ("v", board_x + 1, board_y - 1) in self.v_walls
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

        print(f"\n📊 遊戲狀態:")
        print(
            f"  Player1 (R) - 位置: {xy_to_pos(self.player1.pos[0], self.player1.pos[1])}, 剩餘牆體: {self.player1.walls_left}"
        )
        print(
            f"  Player2 (B) - 位置: {xy_to_pos(self.player2.pos[0], self.player2.pos[1])}, 剩餘牆體: {self.player2.walls_left}"
        )
        print(f"  當前回合: {self.current_player.name}")

    def get_legal_actions_mask(self) -> List[bool]:
        mask = [False] * 209
        cx, cy = self.current_player.pos
        for x, y in self.current_player.valid_moves:
            if (x, y) == (cx, cy):
                continue
            aid = action_to_action_id("move", xy_to_pos(x, y))
            mask[aid] = True
        if self.current_player.walls_left > 0:
            for row_index in range(BOARD_SIZE - 1):
                for col_index in range(BOARD_SIZE - 1):
                    wc = "h" + xy_to_pos(col_index, row_index)
                    w = Wall(wc)
                    if self.is_valid_wall(w):
                        mask[action_to_action_id("wall", wc)] = True
            for row_index in range(BOARD_SIZE - 1):
                for col_index in range(BOARD_SIZE - 1):
                    wc = "v" + xy_to_pos(col_index, row_index)
                    w = Wall(wc)
                    if self.is_valid_wall(w):
                        mask[action_to_action_id("wall", wc)] = True
        return mask

    def get_board_snapshot(self) -> Dict:
        h_walls_list = []
        v_walls_list = []
        seen_h = set()
        for _, col, row in self.h_walls:
            if ("h", col + 1, row) in self.h_walls:
                code = "h" + chr(ord("a") + col) + str(row)
                if code not in seen_h:
                    seen_h.add(code)
                    h_walls_list.append(code)
        seen_v = set()
        for _, col, row in self.v_walls:
            if ("v", col, row + 1) in self.v_walls:
                code = "v" + chr(ord("a") + col - 1) + str(row + 1)
                if code not in seen_v:
                    seen_v.add(code)
                    v_walls_list.append(code)
        return {
            "player1_pos": xy_to_pos(self.player1.pos[0], self.player1.pos[1]),
            "player2_pos": xy_to_pos(self.player2.pos[0], self.player2.pos[1]),
            "walls_remaining": {"p1": self.player1.walls_left, "p2": self.player2.walls_left},
            "placed_walls": {"h": h_walls_list, "v": v_walls_list},
            "current_turn": self.current_player.name,
            "winner": self.check_win(),
        }
