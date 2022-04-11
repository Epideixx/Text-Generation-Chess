from generate_moves import save_games

n_slots = 100
nb_games = 50

for _ in range(n_slots):
    save_games(nb_games, nb_simu=50, filename="fen.txt",
               rate_mcts_vs_mcts=1, rate_mcts_vs_rnd=0, rate_rnd_vs_rnd=0)
