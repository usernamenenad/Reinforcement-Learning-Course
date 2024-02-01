from dataclasses import dataclass, astuple
from enum import Enum, StrEnum
from random import shuffle

from tabulate import tabulate


class CardSuit(StrEnum):
    CLUB = "♣"
    DIAMOND = "♦"
    HEART = "♥"
    SPADE = "♠"

    def __repr__(self):
        return self

    @staticmethod
    def get_all_suits():
        return [CardSuit.CLUB, CardSuit.DIAMOND, CardSuit.HEART, CardSuit.SPADE]


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

    @staticmethod
    def get_all_numbers():
        return [
            CardNumber.ACE,
            CardNumber.TWO,
            CardNumber.THREE,
            CardNumber.FOUR,
            CardNumber.FIVE,
            CardNumber.SIX,
            CardNumber.SEVEN,
            CardNumber.EIGHT,
            CardNumber.NINE,
            CardNumber.TEN,
            CardNumber.JACK,
            CardNumber.DAME,
            CardNumber.KING,
        ]


@dataclass
class Card:
    number: CardNumber
    suit: CardSuit

    @property
    def value(self) -> int:
        if (
            self.number == CardNumber.JACK
            or self.number == CardNumber.DAME
            or self.number == CardNumber.KING
        ):
            return 10
        return self.number.value

    def __repr__(self) -> str:
        return f"{repr(self.number)}{repr(self.suit)}"


class CardDeck:
    def __init__(self, no_sets: int = 5) -> None:
        self.__no_sets: int = no_sets
        self.__deck: list[Card] = list()
        self.__reshuffle()

    def __str__(self) -> str:
        s = ""
        for card in self.__deck:
            s += repr(card) + " "

        return s

    def __reshuffle(self) -> None:
        """
        Used for (RE)creating and SHUFFLING the deck.
        """
        self.__deck = self.__no_sets * [
            Card(number=n, suit=s)
            for n in CardNumber.get_all_numbers()
            for s in CardSuit.get_all_suits()
        ]

        shuffle(self.__deck)

    def draw(self) -> Card:
        if not self.__deck:
            self.__reshuffle()
        return self.__deck.pop(0)


@dataclass
class State:
    total: int = 0
    has_ace: bool = False

    def __hash__(self) -> int:
        return hash(astuple(self))

    def reset(self) -> None:
        self.total = 0
        self.has_ace = False


class Action(Enum):
    HIT = 0
    HOLD = 1

    def get_all_actions():
        return [Action.HIT, Action.HOLD]


class Experience:
    @property
    def experience(self) -> list[list[State | Action | float | Card | None]]:
        return self.__experience

    def __init__(self) -> None:
        """
        Experiences will be represented as list of (State, Action, float, Card) pairs.
        Every index of the list represents a round, i.e. 0th round - index 0 etc.
        This class is instantiated for each game, for each player.
        """
        self.__experience: list[list[State | Action | float | Card | None]] = []

    def __iter__(self):
        return iter(self.__experience)

    def __getitem__(self, index: int) -> list[State | Action | float | Card | None]:
        return self.__experience[index]

    def log(self, exp: list[State | Action | float | Card | None]) -> None:
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

    def clear(self) -> None:
        self.__experience.clear()


@dataclass
class Q:
    """
    Class for representing Q estimates.
    """

    @property
    def states(self) -> list[State]:
        return self.__states

    def __init__(self) -> None:
        self.__states: list[State] = list()
        for total in range(4, 22):
            if total not in range(4, 12):
                self.__states.append(State(total=total, has_ace=False))
                self.__states.append(State(total=total, has_ace=True))
                continue
            self.__states.append(State(total=total, has_ace=False))

        self.__actions: list[Action] = [Action.HOLD, Action.HIT]

        self.__q: dict[tuple[State, Action], float] = {
            (s, a): 0.0 for s in self.__states for a in self.__actions
        }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        """
        Returns the received reward when ending up in given state and taking the given action.
        """
        return self.__q[key]

    def __setitem__(self, key: tuple[State, Action], gain: float) -> None:
        self.__q[key] = gain

    def __str__(self) -> str:
        to_repr = []
        for s, a in self.__q:
            to_repr.append({"State": s, "Action": a, "Value": self.__q[(s, a)]})

        return tabulate(to_repr, headers="keys", tablefmt="rst")

    def determine_v(self, s: State) -> float:
        return max([self.__q[s, a] for a in self.__actions])
