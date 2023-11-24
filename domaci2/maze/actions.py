from enum import Enum, auto


class Direction(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()

    @staticmethod
    def get_all_directions():
        return [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]


class Action(Enum):
    ACTION_R = auto()  # On Level 1 assignment, this would collapse to direction RIGHT
    ACTION_L = auto()  # On Level 1 assignment, this would collapse to direction LEFT
    ACTION_U = auto()  # On Level 1 assignment, this would collapse to direction UP
    ACTION_D = auto()  # On Level 1 assignment, this would collapse to direction DOWN

    @staticmethod
    def get_all_actions():
        return [Action.ACTION_R, Action.ACTION_L, Action.ACTION_D, Action.ACTION_U]


if __name__ == "__main__":
    print(
        "Hi! Here you can find implementation of Action class, used for agent movement."
    )
