from generate_moves import save_games

n_slots = 500
nb_games = 100

for i in range(n_slots):
    random_rate = (n_slots - i)/n_slots
    save_games(nb_games, nb_simu=50, filename="fen.txt",
               random_rate=random_rate)
