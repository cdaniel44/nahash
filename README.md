# nahash
##########################################
######### etapes du programme ############
##########################################

### I. importer les musiques

1. importer les musiques
1. convertir en format midi
1. exporter les midi dans des matrices

### II. Definir les Hyperparameters

1. taille de la couche visible
1. taille de la couche cachee
1. num epochs -> nombre de fois où on entraine notre modele
1. batch size nombre de sons lancer en une fois 
1. learning rate -> vitesse d'apprentissage du modèle -> descente de gradient

### III varibales 

Creer les variables qui vont contenir les donnees, les poids, les bias

### IV Training code 

### V eval code 

1. lancer la session tensorflow
1. nourrir les couches de neurones





#comment creer de la musique


#HyperParameters



#fonctions tensorflow

tf.placeholder:

prepare un espace reservé pour un tensor qui sera fed plus tard avec "feed_dict"

Args:

-dtype : type des éléments du tensor
-shape : taille de la matrice
-name  : nom du placeholder

return : un tensor

tf.variable:

créer un tensor

Args:
x
name=


tf.floor

retourne un tensor où chaque élément est le plus grand possible mais inférieur à x

Args:
x
name=

tf.constant

retourne un tensor constant

Args:
x
dtype=
shape=
name=
verify_shape=

tf.matmul

multiplie 2 tensors

Args:
x
y

tf.sigmoid

fonction d'activation qui renvoie un tensor


tf.Session

### GIBBS


