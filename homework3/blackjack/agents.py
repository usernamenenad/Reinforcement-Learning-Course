from abc import ABC, abstractmethod

from blackjack.utils import *
from blackjack.policy import *


class Agent(ABC):
    """
    An interface for a type of Blackjack player,
    that can be a regular player or a dealer.
    """

    @property
    def state(self) -> State:
        return self.__state

    @property
    def policy(self) -> Policy:
        return self.__policy

    @property
    def name(self) -> str:
        return self.__name

    @abstractmethod
    def __init__(self, state: State, policy: Policy, name: str) -> None:
        self.__state = state
        self.__policy = policy
        self.__name = name

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

    def act(self, q: Q, s: State) -> Action:
        return self.__policy.act(q, s)

    def reset(self) -> None:
        self.__state.reset()


class Dealer(Agent):
    """
    A dealer agent.
    """

    @property
    def state(self) -> State:
        return super().state

    @property
    def policy(self) -> Policy:
        return super().policy

    @property
    def name(self) -> str:
        return super().name

    def __init__(
        self,
        state: State | None = None,
        policy: Policy | None = None,
        name: str = "Dealer",
    ) -> None:
        super().__init__(
            state if state is not None else State(),
            policy if policy is not None else DealerPolicy(),
            name,
        )


class Player(Agent):
    """
    A player agent.
    """

    # Using this just to name players, not much importance.
    no_players = 0

    @property
    def state(self) -> State:
        return super().state

    @property
    def policy(self) -> Policy:
        return super().policy

    @property
    def name(self) -> str:
        return super().name

    @property
    def experiences(self) -> dict[int, Experience]:
        return self.__experiences

    def __init__(
        self,
        state: State | None = None,
        policy: Policy | None = None,
        name: str | None = None,
    ) -> None:
        Player.no_players += 1
        super().__init__(
            state if state is not None else State(),
            policy if policy is not None else EpsGreedyPolicy(),
            name if name is not None else f"Player{Player.no_players}",
        )
        self.__experiences: dict[int, Experience] = {}

    def log_experience(
        self, rnd: int, exp: list[State | Action | float | Card | None]
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
