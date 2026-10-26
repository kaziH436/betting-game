
import random

STARTING_CHIPS = 2_000
TARGET_BETS = 10_000

WIN_PROBABILITY = 0.40
NUM_SIMULATIONS = 50_000


def play_game(bet_fraction):
    chips = STARTING_CHIPS
    total_bet = 0

    while chips > 0 and total_bet < TARGET_BETS:
        bet = max(1, round(chips * bet_fraction))
        bet = min(bet, chips, TARGET_BETS - total_bet)

        # Reaching the cumulative betting target means winning.
        if total_bet + bet >= TARGET_BETS:
            return True

        total_bet += bet

        if random.random() < WIN_PROBABILITY:
            chips += bet
        else:
            chips -= bet

    return total_bet >= TARGET_BETS


def estimate_win_probability(bet_fraction):
    wins = sum(
        play_game(bet_fraction)
        for _ in range(NUM_SIMULATIONS)
    )

    return wins / NUM_SIMULATIONS


for fraction in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    probability = estimate_win_probability(fraction)

    print(
        f"Bet {fraction:.0%} of chips: "
        f"{probability:.2%} estimated win probability"
    )
