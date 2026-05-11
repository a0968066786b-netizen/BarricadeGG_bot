"""
Quoridor Core Package
包含遊戲規則和 Gymnasium 環境實現
"""

from .env import QuoridorEnv
from .rules import (
    Board,
    Player,
    Wall,
    BOARD_SIZE,
    MAX_WALLS,
    DIRECTIONS,
    STRAIGHT_JUMP,
    DIAGONAL_JUMP,
    pos_to_xy,
    xy_to_pos,
    action_id_to_action,
    action_to_action_id,
)

__version__ = "1.0.0"
__all__ = [
    "QuoridorEnv",
    "Board",
    "Player",
    "Wall",
    "BOARD_SIZE",
    "MAX_WALLS",
    "DIRECTIONS",
    "STRAIGHT_JUMP",
    "DIAGONAL_JUMP",
    "pos_to_xy",
    "xy_to_pos",
    "action_id_to_action",
    "action_to_action_id",
]
