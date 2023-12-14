from abc import ABC
from enum import Enum, StrEnum
from dataclasses import dataclass
from random import shuffle


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
class Card():
    number: CardNumber
    suit: CardSuit

    def __repr__(self):
        return f"{repr(self.number)}{repr(self.suit)}"


class CardDeck():

    def __init__(self, no_sets=5):
        self.no_sets = no_sets
        self.deck: list[Card] = list()
        self.__reshuffle()

    def __repr__(self):
        s = str()
        for card in self.deck:
            s += repr(card) + " "

        return s

    def __reshuffle(self) -> None:
        """
        Used for (RE)creating and SHUFFLING the deck.
        """
        self.deck = self.no_sets * [Card(number=n, suit=s)
                                    for n in iter(CardNumber)
                                    for s in iter(CardSuit)]

        shuffle(self.deck)

    def draw(self) -> Card:
        if not self.deck:
            self.__reshuffle()
        return self.deck.pop(0)


@dataclass
class State(ABC):
    total: int = 0
    has_ace: bool = False if total != 11 else True


@dataclass
class DealerState(State):
    pass


@dataclass
class PlayerState(State):
    dealer_total: int = 0


class Action(Enum):
    HIT = 0
    HOLD = 1


class Experience():

    def __init__(self):
        self.experience: list[list[State, Action, float]] = list()

    def __add__(self, exp: list[State, Action, float]):
        self.experience.append(exp)
        return self

    def compute_experience(self, result: float, gamma: float = 1.0):
        self.experience[-1][2] = result

        for i, exp in enumerate(self.experience):
            gain = 0
            discount = 1
            for jexp in self.experience[i:]:
                gain += discount * jexp[2]
                discount *= gamma
            exp[2] = gain
