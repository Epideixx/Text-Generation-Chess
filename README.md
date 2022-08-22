# Objective

The full objective is to use a Transformer architecture to learn how to play Chess and other games such as Ultimate Tic-Tac-Toe

Sub-objectives :
- Create a Transformer model from scratch based on the Paper "Attention Is All You Need"
- Adapt the model to focus on the main objective
- Train the algorithm on Data generated by a Monte Carlo Tree Search (MCTS) algorithm
- Analyse the parameters of the Attention Mechanism to understand how the algorithm learnt the game and what links he makes between the different elements

# Results

The transformer-based model has been successfully coded and has been trained for the game of Ultimate Tic-Tac-Toe. 
It is able then to play less randomly that a non-initiated player. However, it is the limit reached...

Indeed, Transformers are really big models, able to do many tasks, but need generally a very big pre-training, which is not the case here. 
