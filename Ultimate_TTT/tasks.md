TO DO:
- Vérifier le bon fonctionnement du MCTS DONE
- Implémenter une représentation string pour le Transformer DONE
- Créer un générateur de données DONE
- Entraîner le Transformer custom pour TTT DONE

- Générer plus de données plus rapidement ==> IDEE : MCTS avec une part d'aléatoire variable DONE
- En fait, à chaque fois qu'on regénère des données, il supprimait les anciennes ... DONE
- Faire un validation_split DONE
- Faire un entraînement massif IN PROGRESS
- Pouvoir jouer contre lui -> Il faudrait save parameters of transformer ... (Later)


- VOIR SI ON PEUT changer FLATTEN parce que là y a QUE la classification finale qui s'entraîne quasiment !!!!!!! DONE
EXPLICATION : 
En fait, on cherche pas à prédire juste le dernier coup mais l'ensemble des coups qui ont été joués jusque là plus le futur coup...
C'est pour ça qu'il y a le "start" et le "stop" pour l'entraînement normalement.
Et possiblement faudra changer la fonction de coût
- Transformer fonction de coût pour qu'elle target beaucoup plus le dernier coup que les autres au bout d'un moment d'entraînement ABANDON FOR THE MOMENT
- Prendre meilleure accuracy DONE

- Et si ça marche, on passe sur la partie analyse du réseau

NOTES :
- Ben en fait, faut peut-être lui dire pour qui il joue le coco, parce que du coup il sait pas qui il fait gagner ...
D'ailleurs, ici, on est toujours X normalement ! (grâce à la façon dont le jeu est encodé)
- Changer Transformer pour vocab board parce que pas d'espace entre les lettres du plateau DONE