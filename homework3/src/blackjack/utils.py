from dataclasses import dataclass, astuple
from enum import Enum, StrEnum, auto
from random import shuffle, random

from tabulate import tabulate


class CardSuit(StrEnum):
    CLUB = "♣"
    DIAMOND = "♦"
    HEART = "♥"
    SPADE = "♠"

    def __repr__(self):
        return self


class CardNumber(Enum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 12
    DAME = 13
    KING = 14

    def __repr__(self):
        match self:
            case CardNumber.ACE:
                return "A"
            case CardNumber.JACK:
                return "J"
            case CardNumber.DAME:
                return "D"
            case CardNumber.KING:
                return "K"
            case _:
                return str(self.value)


@dataclass
class Card:
    number: CardNumber
    suit: CardSuit

    @property
    def value(self):
        if self.number == CardNumber.JACK or self.number == CardNumber.DAME or self.number == CardNumber.KING:
            return 10
        return self.number.value

    def __repr__(self):
        return f"{repr(self.number)}{repr(self.suit)}"


class CardDeck:

    def __init__(self, no_sets=5):
        self.__no_sets = no_sets
        self.__deck: list[Card] = list()
        self.__reshuffle()

    def __repr__(self):
        s = str()
        for card in self.__deck:
            s += repr(card) + " "

        return s

    def __reshuffle(self) -> None:
        """
        Used for (RE)creating and SHUFFLING the deck.
        """
        self.__deck = self.__no_sets * [Card(number=n, suit=s)
                                        for n in iter(CardNumber)
                                        for s in iter(CardSuit)]

        shuffle(self.__deck)

    def draw(self) -> Card:
        if not self.__deck:
            self.__reshuffle()
        return self.__deck.pop(0)


@dataclass
class State:
    total: int = 0
    has_ace: bool = False

    def reset(self):
        self.total = 0
        self.has_ace = False


@dataclass
class DealerState(State):
    pass


@dataclass
class PlayerState(State):

    def __hash__(self):
        return hash(astuple(self))


class Action(Enum):
    HIT = 0
    HOLD = 1


class Experience:

    @property
    def experience(self) -> list[list[State | Action | float | Card]]:
        return self.__experience

    def __init__(self):
        """
        Experiences will be represented as list of (State, Action, float) pairs.
        Every index of the list represents a round, i.e. 0th round - index 0 etc.
        """
        self.__experience: list[list[State | Action | float | Card]] = list()

    def __iter__(self):
        return iter(self.__experience)

    def __getitem__(self, index: int):
        return self.__experience[index]

    def __repr__(self):
        to_print = list()
        for exp in self.__experience:
            to_print.append(
                {
                    "State": exp[0],
                    "Action": exp[1],
                    "Gain": exp[2]
                }
            )

        return tabulate(to_print, headers="keys", tablefmt="rst") + "\r\n"

    def log(self, exp: [State | Action | float | Card]) -> None:
        """
        Used for adding new (State, Action, Gain) pair to the experience.
        """
        self.__experience.append(exp)

    def build(self, result: float, gamma: float = 1.0) -> None:
        """
        Used for "building gains"; determining the gains starting from every state.
        """
        self.experience[-1][2] = result

        for i, exp in enumerate(self.experience):
            gain = 0.0
            discount = 1.0
            for jexp in self.experience[i:]:
                gain += discount * jexp[2]
                discount *= gamma
            exp[2] = gain

    def clear(self):
        self.__experience.clear()


@dataclass
class Q:
    """
    Class for representing Q estimates.
    """

    def __init__(self):

        self.all_states: list[State] = list()
        for total in range(4, 22):
            if total not in range(4, 12):
                self.all_states.append(PlayerState(total=total, has_ace=False))
                self.all_states.append(PlayerState(total=total, has_ace=True))
                continue
            self.all_states.append(PlayerState(total=total, has_ace=False))

        self.all_actions: list[Action] = [Action.HOLD, Action.HIT]

        self.q: dict[tuple[State, Action], float] = {
            (s, a): random()
            for s in self.all_states
            for a in self.all_actions
        }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        """
        Returns the reward gotten when ending up in given state and taking the given action.
        """
        return self.q[key]

    def __setitem__(self, key: tuple[State, Action], gain: float):
        self.q[key] = gain

    def __repr__(self):
        to_repr = []
        for s, a in self.q:
            to_repr.append(
                {
                    "State": s,
                    "Action": a,
                    "Gain": self.q[(s, a)]
                }
            )

        return tabulate(to_repr, headers="keys", tablefmt="rst")

    def determine_v(self, s: State):
        return max([self.q[s, a] for a in self.all_actions])
