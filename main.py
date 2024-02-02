from __future__ import annotations
from random import randint
from enum import Enum

class Field:
    singleton = None
    
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height
        self.movers: list[Mover] = []
        
        Field.singleton = self
        
    def add_mover(self, mover:Mover) -> None:
        self.movers.append(mover)
        
    def give_perception(self, x:int, y:int, perception_range:tuple[int, int]) -> tuple[int, int] | None:
        for mover in self.movers:
            if isinstance(mover, Target):
                x_relative = mover.x - x
                y_relative = mover.y - y
                
                if abs(x_relative) <= perception_range[0] and abs(y_relative) <= perception_range[1]:
                    return (x_relative, y_relative)
                else:
                    return None
                
        raise Exception("No target found")
        
class Mover:
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y
    
    class Direction(Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4
        
    def move(self, direction: Direction) -> None:
        if direction == self.Direction.UP:
            self.y -= 1
        elif direction == self.Direction.DOWN:
            self.y += 1
        elif direction == self.Direction.LEFT:
            self.x -= 1
        elif direction == self.Direction.RIGHT:
            self.x += 1
    
class QLearner:
    pass
    
class Hunter(Mover):
    def __init__(self, x, y, perception_range:tuple[int,int], q_leaner:QLearner):
        super().__init__(x, y)
        self.perception_range = perception_range
        
    def move(self) -> None:
        super().move(self._decide_direction())
        
    def _decide_direction(self) -> Mover.Direction:
        return Mover.Direction(randint(1, 4))
        
class Target(Mover):
    def __init__(self, x, y):
        super().__init__(x, y)
        
    def move(self) -> None:
        super().move(self._decide_direction())
        
    def _decide_direction(self) -> Mover.Direction:
        return Mover.Direction(randint(1, 4))