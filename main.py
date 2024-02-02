from __future__ import annotations
from random import randint, random
from enum import Enum
import math
from matplotlib import pyplot as plt


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

    def reset(self):
        self.movers = []


class Mover:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    class Direction(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    def move(self, direction: Direction) -> None:
        if direction == self.Direction.UP:
            if self._is_in_field(self.x, self.y - 1):
                self.y -= 1
        elif direction == self.Direction.DOWN:
            if self._is_in_field(self.x, self.y + 1):
                self.y += 1
        elif direction == self.Direction.LEFT:
            if self._is_in_field(self.x - 1, self.y):
                self.x -= 1
        elif direction == self.Direction.RIGHT:
            if self._is_in_field(self.x + 1, self.y):
                self.x += 1

    def _is_in_field(self, x: int, y: int) -> bool:
        return (
            0 <= x < Field.singleton.width and 0 <= y < Field.singleton.height
        )


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
        boltzmann_temperature_rate: float = 0.9999,
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

    def learn(self) -> None:
        self.q_leaner.learn(
            self._relative_to_stateId(self._percept()), self._q_action_callback
        )

    def _decide_direction(self) -> Mover.Direction:
        perception = self._percept()
        return self.q_leaner.decide_action(
            self._relative_to_stateId(perception)
        )

    def _percept(self) -> tuple[int, int] | None:
        return Field.singleton.give_perception(
            self.x, self.y, self.perception_range
        )

    def _prepare_q_learning(self, q_leaner: QLearner) -> None:
        states_n = (self.perception_range[0] * 2 + 1) * (
            self.perception_range[1] * 2 + 1
        ) + 1

        states = []
        for cnt in range(states_n):
            states.append(QLearner.State(cnt, 4))

        q_leaner.states = states

        self.q_leaner = q_leaner

    def _relative_to_stateId(self, relative: tuple[int, int] | None) -> int:
        if relative is None:
            # last one is for "no target found"
            return len(self.q_leaner.states) - 1

        id = (relative[0] + self.perception_range[0]) * (
            self.perception_range[1] * 2 + 1
        ) + (relative[1] + self.perception_range[1])

        return id

    def _q_action_callback(self, action: int) -> tuple[float, int]:
        super().move(Mover.Direction(action))

        caught = Field.singleton.judge_caught()

        if caught:
            return 1, self._relative_to_stateId((0, 0))
        else:
            return -0.1, self._relative_to_stateId(self._percept())


class Target(Mover):
    def __init__(self, x, y):
        super().__init__(x, y)

    def move(self) -> None:
        super().move(self._decide_direction())

    def _decide_direction(self) -> Mover.Direction:
        return Mover.Direction(randint(0, 3))


def run_a_episode(
    field: Field, hunters: list[Hunter], targets: list[Target]
) -> int:
    steps = 0

    # put agents
    field.reset()
    for hunter in hunters:
        field.add_mover(hunter)
        hunter.x = randint(0, field.width - 1)
        hunter.y = randint(0, field.height - 1)

    for target in targets:
        field.add_mover(target)
        target.x = randint(0, field.width - 1)
        target.y = randint(0, field.height - 1)

    # run
    while not field.judge_caught():
        for hunter in hunters:
            hunter.learn()
        for target in targets:
            target.move()
        steps += 1

    return steps


def run_bunch_episodes(hunters, targets):
    field = Field(10, 10)
    episodes = 500
    steps = []

    for cnt in range(episodes):
        steps.append(run_a_episode(field, hunters, targets))

    return steps


def get_averages(x, by):
    averages = []
    start = 0
    starts = []
    while start < len(x):
        starts.append(start)
        end = min(start + by, len(x))
        averages.append(sum(x[start:end]) / (end - start))

        start = end

    return averages, starts


steps_2 = run_bunch_episodes(
    [Hunter(0, 0, (2, 2), QLearner([])), Hunter(0, 0, (2, 2), QLearner([]))],
    [Target(0, 0)],
)
averages_2, starts = get_averages(steps_2, 50)
steps_3 = run_bunch_episodes(
    [Hunter(0, 0, (3, 3), QLearner([])), Hunter(0, 0, (3, 3), QLearner([]))],
    [Target(0, 0)],
)
averages_3, _ = get_averages(steps_3, 50)
steps_4 = run_bunch_episodes(
    [Hunter(0, 0, (4, 4), QLearner([])), Hunter(0, 0, (4, 4), QLearner([]))],
    [Target(0, 0)],
)
averages_4, _ = get_averages(steps_4, 50)

plt.plot(starts, averages_2, label="perception 2")
plt.plot(starts, averages_3, label="perception 3")
plt.plot(starts, averages_4, label="perception 4")


class HunterRemembering(Hunter):
    def __init__(
        self, x, y, perception_range: tuple[int, int], q_leaner: QLearner
    ):
        super().__init__(x, y, perception_range, q_leaner)
        self.perception_former = None
        self.guessing = False

    def _percept(self) -> tuple[int, int]:
        perception = super()._percept()

        if perception is None:
            perception = self.perception_former
            self.guessing = True
        else:
            self.guessing = False

        self.perception_former = perception

        return perception
    
    def learn(self) -> None:
        perception = self._percept()
        if not self.guessing:
            super().learn()
        else:
            self.move()

steps_4_remembering = run_bunch_episodes(
    [HunterRemembering(0, 0, (4, 4), QLearner([])), HunterRemembering(0, 0, (4, 4), QLearner([]))],
    [Target(0, 0)],
)
averages_4_remembering, _ = get_averages(steps_4_remembering, 50)

plt.plot(starts, averages_4_remembering, label="perception 4 remembering")

plt.legend()
plt.show()
