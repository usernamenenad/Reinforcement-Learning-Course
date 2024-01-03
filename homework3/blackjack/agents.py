from abc import ABC, abstractmethod

from .utils import *


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

    @abstractmethod
    def __init__(self, state: State, name: str):
        self.__state = state
        self.__experiences: dict[int, Experience] = dict()
        self.__name = name

    def __str__(self):
        return self.__name

    def update_total(self, card: Card) -> None:
        match card.number:
            case CardNumber.ACE:
                if self.__state.total + 11 <= 21 and not self.__state.has_ace:
                    self.__state.total += 11
                    self.__state.has_ace = True
                else:
                    self.__state.total += 1
            case _:
                self.__state.total += card.value
                if self.__state.total > 21:
                    if self.__state.has_ace:
                        self.__state.total -= 10
                        self.__state.has_ace = False

    def log_experience(
        self, rnd: int, exp: list[State | Action | float | Card]
    ) -> None:
        """
        Used for adding new (State, Action, Gain) pair to the experience.
        """
        if rnd not in self.__experiences:
            self.__experiences[rnd] = Experience()
        self.__experiences[rnd].log(exp)

    def build_gains(self, rnd: int, result: float, gamma: float) -> None:
        """
        Used for "building gains"; determining the gains starting from every state.
        """
        self.__experiences[rnd].build(result, gamma)

    def reset(self):
        self.__state.reset()


class Dealer(Agent):
    """
    A dealer agent.
    This agent is deprecated from Level 2 onwards.
    """

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: DealerState = None, name: str = None):
        name = name if name else "Dealer"
        super().__init__(state if state else DealerState(), name)


class Player(Agent):
    """
    A player agent.
    """

    no_players = 0  # Using this just to name players, not much importance.

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: PlayerState = None, name: str = None):
        name = name if name else "Player" + str(Player.no_players)
        Player.no_players += 1
        super().__init__(state if state else PlayerState(), name)
