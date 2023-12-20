from blackjack import *

import unittest


class TestBlackjack(unittest.TestCase):
    def test_agents(self):
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

    def test_game(self):
        no_players = 3
        players = [Player() for _ in range(no_players)]
        game = Game(players)

        game.play(gamma=0.9, q=Q())
        # Info.draw_experience(game, round=1)
        # Info.log_experiences(game)

    def test_incr_monte_carlo(self):
        no_players = 2
        players = [Player() for _ in range(no_players)]
        game = Game(players)

        q = Q()

        imc = IncrMonteCarlo(q=q, gamma=0.9, iterations=20000)
        imc.run(game)

        print(q)

        random_player = Player()
        to_print = []
        for state in q.all_states:
            action = random_player.policy(q, state)
            to_print.append(
                {
                    "State": state,
                    "Action": action
                }
            )

        print("Optimal policy is:")
        print(tabulate(to_print, headers="keys", tablefmt="rst"))


def main() -> None:
    unittest.main()
