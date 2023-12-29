from blackjack import *


def test_agents():
    card_dealer = Card(CardNumber.SIX, CardSuit.CLUB)
    print(f"Dealer's card: {card_dealer}")

    dealer = Dealer(state=DealerState())
    dealer.update_total(card=card_dealer)
    print(f"Agent: {dealer}, state: {dealer.state}\r\n")

    card_player = Card(CardNumber.ACE, CardSuit.DIAMOND)
    print(f"Player's card: {card_player}")

    player = Player(state=PlayerState())
    player.update_total(card=card_player)
    print(f"Agent: {player}, state: {player.state}\r\n")

    another_ace = Card(CardNumber.ACE, CardSuit.HEART)
    print(f"Player drew another ace card: {another_ace}")
    player.update_total(card=another_ace)
    print(f"Agent: {player}, state: {player.state}\r\n")
