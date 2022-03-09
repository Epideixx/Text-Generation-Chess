Ceci sont les différentes idées que j'ai et qu'il faut implémenter:
- Mettre en place un transformer "classique"
- En entrée de l'encoder, il y aura le plateau actuel, représenter sous la forme ...r.Q..//.PRBR...// etc où il faut que chaque lettre soit tokenize
- En entrée du décodeur, il y aura la succession des coups alors joués
- En sortie, sachant qu'il y a au maximum 218 coups possibles, le vocab de sortie est ainsi limité à ce nombre et il faut estimé quel coup à le plus de proba de sortir done:
    - Sortie 1 : vecteur de 218 pour savoir quel coup jouer
    - Sortie 2 : vecteur de taille 1 pour savoir la proba de gagner
- Pour les données utilisées, on récupère toutes les parties et on s'assure que tout se qui passe à l'entraînement sert à faire gagner le joueur (pour l'instant)

Autres idées :
- On va utiliser 3 Transformers différents : 
    - Un pour les ouvertures
    - Un pour le milieu de jeu
    - Un pour la fin de jeu
- Pour le mécanisme d'attention, faire de l'attention positive et négative, genre positive pour ce qu'on peut faire, les opportunités et négative pour ce qui est un risque
- Changer l'encodage de la position pour être sur 2 axes (tandis que pour le momment c'est que sur 1 vu le texte est linéaire)


Pour transformers.py :
- Comprendre les formats des données DONE
- Save and Load
    - Encoder + decoder + maybe vocab
- Add Wandb


Papiers à voir :
- Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

Problèmes :
- gradients trop faibles voire nuls
- loss exactement PAREIL à chaque fois 8.303