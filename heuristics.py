from typing import Tuple
import math


def heuristic_manhatten_distance(source: Tuple, target: Tuple) -> float:
    (x1, y1) = source
    (x2, y2) = target
    manhatten_distance = abs(x1 - x2) + abs(y1 - y2)
    return manhatten_distance


def heuristic_euclidian_distance(source: Tuple, target: Tuple) -> float:
    (x1, y1) = source
    (x2, y2) = target
    euclidian_distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return euclidian_distance


def heuristic_manhatten_distance_cheat(source: Tuple, target: Tuple, x_weight: float = 1.0, y_weight: float = 1.0) -> float:
    (x1, y1) = source
    (x2, y2) = target
    manhatten_distance = x_weight * abs(x1 - x2) + y_weight * abs(y1 - y2)
    return manhatten_distance


def heuristic_minmax(source: Tuple, target: Tuple, alpha: float = 0.0) -> float:
    """
    Try to force agent to not change direction
    (e.g. look at the world "wumpus_world_custom_03.json").
    """
    (x1, y1) = source
    (x2, y2) = target

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    distance = alpha * max(dx, dy) + (1 - alpha) * min(dx, dy)
    return distance
