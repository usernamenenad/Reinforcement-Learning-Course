from enum import Enum, auto


class Action(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()

    @staticmethod
    def get_all_actions():
        return [Action.RIGHT, Action.LEFT, Action.DOWN, Action.UP]


if __name__ == '__main__':
    print('Hi! Here you can find implementation of Action class, used for agent movement.')
