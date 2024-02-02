from __future__ import annotations
from random import randint, random
from enum import Enum
import math


class Field:
    singleton = None

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.movers: list[Mover] = []

        Field.singleton = self

    def add_mover(self, mover: Mover) -> None:
        self.movers.append(mover)

    def give_perception(
        self, x: int, y: int, perception_range: tuple[int, int]
    ) -> tuple[int, int] | None:
        for mover in self.movers:
            if isinstance(mover, Target):
                x_relative = mover.x - x
                y_relative = mover.y - y

                if (
                    abs(x_relative) <= perception_range[0]
                    and abs(y_relative) <= perception_range[1]
                ):
                    return (x_relative, y_relative)
                else:
                    return None

        raise Exception("No target found")
    
    def judge_caught(self) -> bool:
        for mover in self.movers:
            if isinstance(mover, Hunter):
                for mover2 in self.movers:
                    if isinstance(mover2, Target):
                        if mover.x == mover2.x and mover.y == mover2.y:
                            return True
        return False


class Mover:
    def __init__(self, x: int, y: int):
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
    class State:
        def __init__(self, id: int, actions_n: int):
            self.id = id
            self.actions_n = actions_n
            self.q_values = [0] * actions_n

    def __init__(
        self,
        states: list[QLearner.State],
        leaning_rate: float = 0.8,
        discount_factor: float = 0.9,
        boltzmann_temperature_rate: float = 0.99,
    ):
        self.states = states
        self.learning_rate = leaning_rate
        self.discount_factor = discount_factor
        self.boltzmann_temperature = 1
        self.boltzmann_temperature_rate = boltzmann_temperature_rate

    def decide_action(self, state: int) -> int:
        # Boltzmann selection
        probs = [
            math.exp(q / self.boltzmann_temperature)
            for q in self.states[state].q_values
        ]

        # Normalize
        probs_sum = sum(probs)
        probs = [p / probs_sum for p in probs]

        # Choose action
        decided = -1
        r = random()
        for cnt in range(len(probs)):
            if r < probs[cnt]:
                decided = cnt
                break
            r -= probs[cnt]

        # Update temperature
        self.boltzmann_temperature *= self.boltzmann_temperature_rate

        return decided

    def learn(self, state: int, action_callback: callable) -> None:
        """
        :param action_callback: (action:int) -> reward:float, state_id: int
        """

        # take action
        action = self.decide_action(state)
        reward, state_next = action_callback(action)

        # update q value
        q_new = (1 - self.learning_rate) * self.states[state].q_values[action]
        q_new += self.learning_rate * (
            reward
            + self.discount_factor * max(self.states[state_next].q_values)
        )

        self.states[state].q_values[action] = q_new


class Hunter(Mover):
    def __init__(
        self, x, y, perception_range: tuple[int, int], q_leaner: QLearner
    ):
        super().__init__(x, y)
        self.perception_range = perception_range

        self._prepare_q_learning(q_leaner)

    def move(self) -> None:
        super().move(self._decide_direction())

    def _decide_direction(self) -> Mover.Direction:
        return self.q_leaner.decide_action(self._relative_to_stateId(self._perception()))
    
    def _prepare_q_learning(self, q_leaner: QLearner) -> None:
        states_n = (self.perception_range * 2 + 1) ** 2
        
        states = []
        for cnt in range(states_n):
            states = QLearner.State(cnt, 4)
            
        q_leaner.states = states
        
        self.q_leaner = q_leaner
        
    def _relative_to_stateId(self, relative: tuple[int, int]) -> int:
        return (relative[0] + self.perception_range) * (self.perception_range * 2 + 1) + (relative[1] + self.perception_range)

class Target(Mover):
    def __init__(self, x, y):
        super().__init__(x, y)

    def move(self) -> None:
        super().move(self._decide_direction())

    def _decide_direction(self) -> Mover.Direction:
        return Mover.Direction(randint(1, 4))
