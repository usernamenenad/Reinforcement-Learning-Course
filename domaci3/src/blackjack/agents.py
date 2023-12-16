from abc import abstractmethod

from .utils import *
from random import random


class Agent(ABC):
    """
    An interface for a type of Blackjack player,
    that can be a regular player or a dealer.
    """

    @property
    def state(self) -> State:
        return self.__state

    @property
    def experiences(self) -> dict[int, Experience]:
        return self.__experiences

    @property
    def name(self) -> str:
        return self.__name

    def __init__(self, state: State, name: str):
        self.__state = state
        self.__experiences: dict[int, Experience] = dict()
        self.__name = name

    def __repr__(self):
        return self.__name

    def update_total(self, card: Card):
        match card.number:
            case CardNumber.ACE:
                if self.__state.total + card.number.value <= 21:
                    self.__state.total += 11
                    self.__state.has_ace = True
                else:
                    self.__state.total += 1
            case _:
                self.__state.total += card.number.value
                if self.__state.total > 21:
                    if self.__state.has_ace:
                        self.__state.total -= 10
                        self.__state.has_ace = False

    def log_experience(self, round: int, exp: list[State | Action | float]):
        if round not in self.__experiences:
            self.__experiences[round] = Experience()
        self.__experiences[round].log(exp)

    def build_gains(self, round: int, result: float, gamma: float):
        self.__experiences[round].build(result, gamma)

    def reset(self):
        self.__state.reset()

    @abstractmethod
    def policy(self) -> Action:
        pass


class Dealer(Agent):
    """
    A dealer agent.
    """

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: DealerState = None, name: str = None):
        name = name if name else "Dealer"
        super().__init__(state if state else DealerState(), name)

    def policy(self) -> Action:
        return Action.HIT if self.state.total < 17 else Action.HOLD


class Player(Agent):
    """
    A player agent.
    """

    __no_players = 0

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: PlayerState = None, name: str = None):
        name = name if name else "Player" + str(Player.__no_players)
        Player.__no_players += 1
        super().__init__(state if state else PlayerState(), name)

    def policy(self) -> Action:
        return Action.HIT if random() < 0.5 else Action.HOLD
