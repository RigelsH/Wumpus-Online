from typing import Dict, List, Tuple
import wumpus as wws

def rotate_clockwise(orientation):
    """
    [0,1] (north) -> [1,0] (east)
    [1,0] (east) -> [0,-1] (south)
    [0,-1] (south) -> [-1,0] (west)
    [-1,0] (west) -> [0,1] (north)
    """
    return (orientation[1], -orientation[0])

def rotate_counterclockwise(orientation):
    """
    [0,1] (north) -> [-1,0] (west)
    [-1,0] (west) -> [0,-1] (south)
    [0,-1] (south) -> [1,0] (east)
    [1,0] (east) -> [0,1] (north)
    """
    return (-orientation[1], orientation[0])

def find_lowest_indexes(lst):
    """
    Given list of values, returns ids of the lowest 
    values.
    """
    min_value = min(lst)
    indexes = []
    for i, value in enumerate(lst):
        if value == min_value:
            indexes.append(i)
    return indexes

def navigate(path, orientation):
    """
    Given path and orientation, returns the path 
    using LEFT, RIGHT and MOVE variables.
    """
    actions = []

    for i in range(len(path) - 1):
        current_pos = path[i]
        next_pos = path[i + 1]

        # Determine the relative direction to the next position
        direction = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])

        # Rotate clockwise or counterclockwise to face the next position
        while orientation != direction:
            if rotate_clockwise(orientation) == direction:
                actions.append(wws.Hunter.Actions.RIGHT)
                orientation = rotate_clockwise(orientation)
            else:
                actions.append(wws.Hunter.Actions.LEFT)
                orientation = rotate_counterclockwise(orientation)

        # Move forward
        actions.append(wws.Hunter.Actions.MOVE)

    return actions[0]