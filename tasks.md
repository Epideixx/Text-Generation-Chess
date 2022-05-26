NOUVEAU :

- Regarder le padding mask, et vérifier qu'il est ok


ANCIENS (à trier):

Ceci sont les différentes idées que j'ai et qu'il faut implémenter:
- Mettre en place un transformer "classique" ==> DONE
- En entrée de l'encoder, il y aura le plateau actuel, représenter sous la forme ...r.Q..//.PRBR...// etc où il faut que chaque lettre soit tokenize ==> DONE
- En entrée du décodeur, il y aura la succession des coups alors joués ==> DONE
- En sortie, sachant qu'il y a au maximum 218 coups possibles, le vocab de sortie est ainsi limité à ce nombre et il faut estimé quel coup à le plus de proba de sortir done: FALSE
    - Sortie 1 : vecteur de ... pour savoir quel coup jouer DONE
    - Sortie 2 : vecteur de taille 1 pour savoir la proba de gagner NOT DONE
- Pour les données utilisées, on récupère toutes les parties et on s'assure que tout se qui passe à l'entraînement sert à faire gagner le joueur (pour l'instant) DONE

Autres idées : NOT DONE
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
- Add Wandb DONE

Pour DEV (05/04) :
- Changer Accuracy pour avoir max 1
- Taille vocab automatique ?
- Utiliser mask pour contrecarrer padding dans Tokenizer
- Couches Dropout
- Training complet sur cluster


Pour Parallélisme :
- Demander Guillaume Joslin (donner contact à José)
- Possible que la partie BP soit plus courte (GPU) que MCTS (RAM)
- Tester avec de plus gros trucs sur GPU et voir si en fait c'est pas la partie RAM qui bloque parce que pas parallélisable 
- Ou alors GRAM sature



POURQUOI J'ARRIVE PLUS A PARTIR D'UNE HAUTE LOSS ???
Genre c'est comme si ça s'initialisait toujours au même endroit ...

TO DO:
- Vérifier le bon fonctionnement du MCTS DONE
- Implémenter une représentation string pour le Transformer DONE
- Créer un générateur de données DONE
- Entraîner le Transformer custom pour TTT DONE

- Générer plus de données plus rapidement ==> IDEE : MCTS avec une part d'aléatoire variable DONE
- En fait, à chaque fois qu'on regénère des données, il supprimait les anciennes ... DONE
- Faire un validation_split DONE
- Faire un entraînement massif IN PROGRESS
- Pouvoir jouer contre lui IN PROGRESS -> Il faudrait save parameters of transformer ... (Later)


- VOIR SI ON PEUT changer FLATTEN parce que là y a QUE la classification finale qui s'entraîne quasiment !!!!!!! DONE
EXPLICATION : 
En fait, on cherche pas à prédire juste le dernier coup mais l'ensemble des coups qui ont été joués jusque là plus le futur coup...
C'est pour ça qu'il y a le "start" et le "stop" pour l'entraînement normalement.
Et possiblement faudra changer la fonction de coût
- Transformer fonction de coût pour qu'elle target beaucoup plus le dernier coup que les autres au bout d'un moment d'entraînement ABANDON FOR THE MOMENT
- Prendre meilleure accuracy DONE
- Regarder le Embedding "Gradients do not exist" DONE
- Installer Cuda DONE

- On va voir s'il y a des segments sur lesquels il apprend mieux

- Et si ça marche, on passe sur la partie analyse du réseau

Projet : Deadline générale = 1 Juin
- Trouver ce qui marchait avant ==> C'est sans aucun doute le shuffle(32000) !!!!!!!!!!!!!!!!!!
- Réussir un entraînement
- Faire apprendre au fur et à mesure des parties de plus en plus hard
- Mettre en place un interface graphique 
- Mettre en place un joueur Transformer
- Obtenir des résultats

NOTES :
- Ben en fait, faut peut-être lui dire pour qui il joue le coco, parce que du coup il sait pas qui il fait gagner ...
D'ailleurs, ici, on est toujours X normalement ! (grâce à la façon dont le jeu est encodé)
- Changer Transformer pour vocab board parce que pas d'espace entre les lettres du plateau DONE