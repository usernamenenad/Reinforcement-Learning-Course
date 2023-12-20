from abc import abstractmethod

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

    def __init__(self, state: State, name: str):
        self.__state = state
        self.__experiences: dict[int, Experience] = dict()
        self.__name = name

    def __repr__(self):
        return self.__name

    def update_total(self, card: Card):
        match card.number:
            case CardNumber.ACE:
                # If we get two aces in a row, both of them can be counted as 1.
                # This is a little OP and will probably need revision.
                if self.__state.total + card.value <= 21 and not self.__state.has_ace:
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

    def log_experience(self, round: int, exp: list[State | Action | float]) -> None:
        """
        Used for adding new (State, Action, Gain) pair to the experience.
        """
        if round not in self.__experiences:
            self.__experiences[round] = Experience()
        self.__experiences[round].log(exp)

    def build_gains(self, round: int, result: float, gamma: float) -> None:
        """
        Used for "building gains"; determining the gains starting from every state.
        """
        self.__experiences[round].build(result, gamma)

    def reset(self):
        self.__state.reset()

    @abstractmethod
    def policy(self, q: Q, state: State) -> Action:
        pass


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

    def policy(self, q: Q, state: DealerState) -> Action:
        return Action.HIT if self.state.total < 17 else Action.HOLD


class Player(Agent):
    """
    A player agent.
    """

    __no_players = 0  # Using this just to name players, not much importance.

    @property
    def state(self) -> State:
        return super().state

    def __init__(self, state: PlayerState = None, name: str = None):
        name = name if name else "Player" + str(Player.__no_players)
        Player.__no_players += 1
        super().__init__(state if state else PlayerState(), name)

    def policy(self, q: Q, state: PlayerState) -> Action:
        """
        A greedy policy in this case.
        """
        return Action.HIT if q[(state, Action.HIT)] > q[(state, Action.HOLD)] else Action.HOLD
