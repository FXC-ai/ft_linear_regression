# Installations nécessaires au projet

Pour installer matplotlib et numpy :

```bash
python3 -m pip install numpy
python3 -m pip install matplotlib
```

En utilisant **`python3 -m pip install numpy`**, on s’assure que **`numpy`** est installé pour la version correcte de Python utilisée par notre script. C'est une pratique recommandée lorsque nous avons plusieurs versions de Python installées sur votre système.

# Sujet

## Partie obligatoire

Vous allez mettre en œuvre une régression linéaire simple avec une seule caractéristique - dans ce cas, le kilométrage de la voiture.

Pour ce faire, vous devez créer deux programmes :

- Le premier programme sera utilisé pour prédire le prix d'une voiture pour un kilométrage donné. Lorsque vous lancez le programme, il doit vous demander un kilométrage, puis vous donner le prix estimé pour ce kilométrage. Le programme utilisera l'hypothèse suivante pour prédire le prix :

$$
estimatePrice(mileage) = \theta_0 + (\theta_1 * mileage)
$$

Avant l'exécution du programme d'entraînement, theta0 et theta1 seront mis à 0.

- Le second programme sera utilisé pour entraîner votre modèle. Il lira votre fichier de données et effectue une régression linéaire sur les données. Une fois la régression linéaire terminée, vous enregistrerez les variables theta0 et theta1 pour les utiliser dans le premier programme.
Vous utiliserez les formules suivantes :

$$
tmp\theta_0 = learningRate * \frac{1}{m} * \sum\limits_{i=0}^{m-1}estimate(mileage[i]) - price[i]
$$

$$
tmp\theta_1 = learningRate * \frac{1}{m} * \sum\limits_{i=0}^{m-1}(estimate(mileage[i]) - price[i]) * mileage[i]
$$

Je vous laisse deviner la valeur de m :) Notez que l'estimatePrice est la même que dans notre premier programme, mais ici elle utilise votre theta0 et theta1 temporaires, calculés en dernier. N'oubliez pas non plus de mettre à jour simultanément theta0 et theta1.

## Bonus

Voici quelques bonus qui pourraient s'avérer très utiles :

- Tracer les données sur un graphique pour voir leur répartition.
- Tracer la droite résultant de votre régression linéaire sur le même graphique, pour voir le résultat de votre travail !
- Un programme qui calcule la précision de votre algorithme.

# Théorie

## Le dataset

Le sujet fournit le dataset suivant :

| km | price |
| --- | --- |
| 240000 | 3650 |
| 139800 | 3800 |
| 150500 | 4400 |
| 185530 | 4450 |
| 176000 | 5250 |
| 114800 | 5350 |
| 166800 | 5800 |
| 89000 | 5990 |
| 144500 | 5999 |
| 84000 | 6200 |
| 82029 | 6390 |
| 63060 | 6390 |
| 74000 | 6600 |
| 97500 | 6800 |
| 67000 | 6800 |
| 76025 | 6900 |
| 48235 | 6900 |
| 93000 | 6990 |
| 60949 | 7490 |
| 65674 | 7555 |
| 54000 | 7990 |
| 68500 | 7990 |
| 22899 | 7990 |
| 61789 | 8290 |

Voici la représentation graphique de ce dataset :

![Figure_2.png](https://github.com/FXC-ai/ft_linear_regression/blob/master/Figure_2.png)

## Le modèle

En second lieu, le sujet impose d’optimiser le modèle suivant : 

$$
estimatePrice(mileage) = \theta_0 + (\theta_1 * mileage)
$$

Le but de l’exercice est donc de déterminer pour quelles valeurs de theta0 et theta1 notre modèle est le plus efficace pour prédire le prix d’une voiture selon son kilométrage. Ces 2 paramètres doivent être optimisés en utilisant l’algorithme de la descente de gradient optimisé. On peut déjà estimé à l’œil sur le graphique que :

$$
8000 < \theta_0 < 9000
$$

$$
-0,1 <\theta_1<-0,3
$$

Pour déterminer si le modèle est efficace dans ces prédictions, nous comparons ses prédictions aux valeurs fournies par le dataset. Cette comparaison est permise grâce à la fonction de coût.

## La fonction de coût

Elle n’est pas donné par le sujet mais après quelques recherches sur internet, la voici :

$$
cost(\theta_0, \theta_1) = \frac{1}{m} * \sum\limits_{i=0}^{m-1}(estimatePrice(mileage[i]) - price[i])^2
$$

Ou encore…

$$
cost(\theta_0, \theta_1) = \frac{1}{m} * \sum\limits_{i=0}^{m-1}((\theta_0 + \theta_1 *mileage[i]) - price[i])^2
$$

Cette fonction ressemble au calcul de la variance. Le résultat est la moyenne du carré des erreurs entre les valeurs prédites par le modèle et les valeurs observées. Ainsi plus la valeur de `cost` est faible, plus le modèle est performant pour faire des prédictions.

Il est possible d’en faire une représentation graphique en mettant les valeurs de theta0 en abscisse et les valeurs de theta1 en ordonnée. Le résultat du calcul de la fonction sera sur l’axe z. On obtient le résultat suivant :

![Figure_1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/4bc43969-1fca-4a09-8d55-3219240d78af/2b71439a-32fb-48b0-9a92-87bc3b300860/Figure_1.png)

N.B. : les valeurs des paramètres sur le graphiques ne correspondent pas à la réalité car elles ont été normalisées. Nous verrons pourquoi plus loin.

Le but de l’exercice est donc de trouver pour quelles valeurs de theta0 et theta1 cette fonction atteint son minimum.

## Dérivées partielles de la fonction de coût

Le sujet impose de débuter l’entraînement de notre modèle en initialisant theta0 = 0 et theta1 = 0. Nous pouvons d’ores et déjà calculer la valeur de notre fonction de coût pour ce cas là. On obtient : cost(0,0) = 41761038.5 ou encore cost(0,0) = 41761038.58 sur les valeurs normalisées du dataset. Comme nous pouvons le constater la fonction de coût a une valeur très élevée dans ce cas.

La question qui se pose maintenant est : `comment choisir d’autres valeurs pour theta0 et theta1 pour que notre fonction de coût présente un meilleur résultat ?`

La réponse est simple : nous allons utiliser les dérivés de cette fonction de coût ! De façon très basique la dérivé est une quantité qui nous fournit 2 informations très utiles sur une fonction :

- Elle nous indique si cette fonction est croissante ou décroissante. Si la dérivée est positive la fonction est croissante. Si la dérivée est négative la fonction est décroissante.
- Elle nous indique également si la fonction croit (ou décroit) rapidement. Plus la valeur de la dérivée est élevée plus la fonction croit (ou décroit) rapidement.

Pour plus de détail, je conseille de regarder la suite de vidéo de Lé Nguyen Hoang sur le sujet (cf. sources).

Comme nous avons 2 paramètres à optimiser, nous avons besoin de 2 dérivées : la dérivée partielle de cost selon theta0 et la dérivée partielle de cost selon theta1. Qu’à cela ne tienne nous allons les calculer immédiatement !

Commençons par développer l’équation de notre fonction de coût :

$$
cost(\theta_0, \theta_1) = \frac{1}{m} * \sum\limits_{i=0}^{m-1}((\theta_0 + \theta_1 *mileage[i]) - price[i])^2
$$

Nous obtenons :

$$
cost(\theta_0, \theta_1) = \frac{1}{m} * \sum\limits_{i=0}^{m-1}\theta_0^2+2*\theta_0*\theta_1*mileage[i]+\theta_1^2*mileage[i]^2-2\theta_0*price[i]-2*\theta_1*price[i]*mileage[i]+price[i]^2
$$

### Dérivée partielle selon theta0

Nous pouvons calculer la dérivée de notre fonction de coût terme à terme selon theta0, nous obtenons :

$$
\frac {\partial cost(\theta_0, \theta_1)}{\partial \theta_0} = \frac{1}{m} * \sum\limits_{i=0}^{m-1}2 * \theta_0+2*\theta_1*mileage[i]+0-2*price[i]-0+0
$$

$$
\frac {\partial cost(\theta_0, \theta_1)}{\partial \theta_0} = \frac{2}{m} * \sum\limits_{i=0}^{m-1} \theta_0+\theta_1*mileage[i]-price[i]
$$

$$
\frac{\partial cost}{\partial \theta_0} =  \frac{2}{m} * \sum\limits_{i=0}^{m-1}estimatePrice(mileage[i]) - price[i]
$$

Or, le sujet nous impose de mettre à jour nos paramètres a l’aide de la formule suivante :

$$
tmp\theta_0 = learningRate * \frac{1}{m} * \sum\limits_{i=0}^{m-1}estimate(mileage[i]) - price[i]
$$

Nous pouvons constater que cela est cohérent car nous retrouvons la formule de notre dérivée précédemment calculée.

### Dérivée partielle selon theta1

De même nous pouvons calculer la dérivée terme à terme de notre fonction de coût selon theta1 :

$$
\frac {\partial cost(\theta_0, \theta_1)}{\partial \theta_1} = \frac{1}{m} * \sum\limits_{i=0}^{m-1}0 + 2 * \theta_0 * mileage[i] + 2 * \theta_1 * mileage[i]^2 - 0 - 2 * price[i] * mileage[i] + 0
$$

$$
\frac {\partial cost(\theta_0, \theta_1)}{\partial \theta_1} = \frac{2}{m} * \sum\limits_{i=0}^{m-1}(\theta_0 +\theta_1 * mileage[i] - price[i] )* mileage[i]
$$

$$
\frac{\partial cost}{\partial \theta_1} = \frac{2}{m} * \sum\limits_{i=0}^{m-1}[estimatePrice(mileage[i]) - price[i]] * mileage[i]
$$

De la même manière, nous retrouvons la formule de notre dérivée dans la formule imposée par le sujet pour mettre à jour theta1.

$$
tmp\theta_1 = learningRate * \frac{1}{m} * \sum\limits_{i=0}^{m-1}(estimate(mileage[i]) - price[i]) * mileage[i]
$$

Nous pouvons cependant remarquer que le facteur 2 des dérivées que nous avons calculer à été remplacé par le learningRate dans la formule fournis par le sujet. Il est temps de se pencher sur ce paramètre…

## Le taux d’apprentissage

Grâce au calcul de nos dérivés nous savons si nous devons augmenter ou diminuer la valeur de theta0 et de theta1 à chaque itération. Parfait ! Mais de combien ? C’est là que le taux d’apprentissage entre en jeu ! Le taux d’apprentissage est le pourcentage de la dérivée que nous soustrairont à theta0 et theta1 afin de les mettre à jour à chaque itération.
Nouvelle question : `quelle valeur donner à ce learningRate ?`

Ici nous sommes face à un dilemme :

- soit nous choisissons un learningRate élevé : dans ce cas notre algorithme va converger après un faible nombre d’itération, cependant le revers de la médaille est que nous risquons de passer à côté du minimum de notre fonction de coût. Dit autrement si nous faisons de trop grands pas, nous risquons d’enjamber le minimum de la fonction sans nous en rendre compte.
- soit nous choisissons un learningRate faible : dans ce cas nous allons faire de tous petits pas qui nous permettrons plus certainement de tomber sur le minimum de notre fonction mais au prix d’un très grand nombre de calcul

Alors comment fait-on ? Eh bien, je n’ai pas de recette miracle à part celle de tester notre algo avec différents taux d’apprentissage et de voir quel learningRate permet une convergence satisfaisante en un temps raisonnable.

## Normalisation du dataset

Avant de lancer notre algorithme à la recherche du minimum de notre fonction de coût, nous allons lui simplifier la tâche grâce à la normalisation des données. 

`Qu’est ce que normaliser les données ?`

Comme nous l’avons dit, les 2 paramètres à optimiser sont theta0 et theta1. Theta0 doit avoir une valeur optimale entre 8000 et 9000. Quand à theta1, il semblerait que sa valeur optimale soit comprise entre -0,1 et -0,3. Nous pouvons constater que l’ordre de grandeur de nos 2 paramètres est différent (il y a un facteur de x10000) entre les 2.

Normaliser les données permets de régler ce problème. Dans notre cas j’ai utilisé (arbitrairement) une normalisation min-max.

$$
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

En normalisant ainsi les données, on obtient le dataset suivant :

| km normalisés | price normalisés |
| --- | --- |
| 1 | 0 |
| 0.538463664 | 0.032327586 |
| 0.587749481 | 0.161637931 |
| 0.749102952 | 0.172413793 |
| 0.705206333 | 0.344827586 |
| 0.423309888 | 0.36637931 |
| 0.662829743 | 0.463362069 |
| 0.304471191 | 0.504310345 |
| 0.560112574 | 0.50625 |
| 0.281440436 | 0.549568966 |
| 0.272361712 | 0.590517241 |
| 0.184987632 | 0.590517241 |
| 0.235378925 | 0.635775862 |
| 0.343623475 | 0.67887931 |
| 0.203135868 | 0.67887931 |
| 0.244706381 | 0.700431034 |
| 0.116701443 | 0.700431034 |
| 0.322895795 | 0.719827586 |
| 0.175264048 | 0.827586207 |
| 0.197028111 | 0.841594828 |
| 0.143255904 | 0.935344828 |
| 0.210045094 | 0.935344828 |
| 0 | 0.935344828 |
| 0.179133214 | 1 |

`Pourquoi normaliser les données ?`

Cela est nécessaire car les valeurs des dérivées partielles de theta0 et theta1 sont très différentes. Par à la première itération, on obtiendrait :

- drv_theta0 (0,0) = -6331.833333333333
- drv_theta1 (0,0) = -582902525.4166666

Comme nous pouvons le constater il y a un facteur 100000 entre la valeur de la dérivée pour theta0 et la valeur de la dérivée pour theta1. Or le learningRate pour la mise à jour de theta0 et theta1 est le même pour nos deux paramètres. Par conséquent, l’algorithme ne fonctionne pas. Les 2 paramètres sont mis à jour à des “vitesses” différentes. On pourrait régler ce problème en utilisant un learningRate différent pour chaque paramètre mais ce serait un bricolage peu fiable.

# Le code du projet

## Les fonctions

```python
import numpy as np
import csv
import matplotlib.pyplot as plt

def read_datas_to_array (file_name) :
	try :
		file = open(file_name, 'r')
	except Exception as exc:
		print("File error : {}".format(exc.__class__))
		exit(0)
	reader = csv.reader(file)
	datas = list(reader)
	del(datas[0])
	arr_datas = np.array(datas, dtype = 'i')
	return arr_datas

def normalize_minmax (value, arr_data) :
	return (value - arr_data.min()) / (arr_data.max() - arr_data.min())

def unnormalize_minmax (value, arr_data) :
	return value * (arr_data.max() - arr_data.min()) + arr_data.min()

def normalize_minmax_arr (arr_data):
	return (arr_data - arr_data.min()) / (arr_data.max() - arr_data.min())

def unnormalize_minmax_arr (arr_data, arr_normalized_data):
	return arr_normalized_data * (arr_data.max() - arr_data.min()) + arr_data.min()

def estimatePrice(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

def cost_fct(arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1])**2
	result = arr_errors.mean()
	return result

def drv_cost_fct_theta0 (arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1])
	return arr_errors.mean()

def drv_cost_fct_theta1 (arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1]) * arr_datas[:,0]
	return arr_errors.mean()

def read_model_parameters():
	with open("model_parameters.txt", "r") as model_parameters_file :
		model_parameters_file = open("model_parameters.txt", "r")
		list_str_parameters = model_parameters_file.readlines()
	theta0 = float(list_str_parameters[0])
	theta1 = float(list_str_parameters[1])
	return {"theta0" : theta0, "theta1" : theta1}

def display_values(arr_mileage, arr_price, id_graph):
	fig = plt.figure(id_graph)
	plt.scatter(arr_mileage, arr_price, marker = 'P')
	fig.suptitle("Observations only")
	plt.grid()
	plt.show()
	plt.close()

def display_model(arr_mileage,  arr_estimated_price, id_graph):
	fig = plt.figure(id_graph)
	plt.plot(arr_mileage, arr_estimated_price, c = "green")
	fig.suptitle("Predictions only")
	plt.grid()
	plt.show()
	plt.close()

def display_values_and_model(arr_mileage, arr_price, arr_estimated_price, id_graph):
	fig = plt.figure(id_graph)
	plt.scatter(arr_mileage, arr_price, marker = 'P')
	plt.plot(arr_mileage, arr_estimated_price, c = "green")
	fig.suptitle("Observations and predictions")
	plt.grid()
	plt.show()
	plt.close()

def display_cost_fct(arr_normalized_datas):
    fig = plt.figure()
    # 3D Surface Plot
    ax1 = plt.axes(projection='3d')
    arr_theta0 = np.linspace(-2, 4, 100)
    arr_theta1 = np.linspace(-4, 2, 100)
    theta0_grid, theta1_grid = np.meshgrid(arr_theta0, arr_theta1)
    values = np.zeros_like(theta0_grid)
    for i in range(len(arr_theta0)):
        for j in range(len(arr_theta1)):
            values[i, j] = cost_fct(arr_normalized_datas, theta0_grid[i, j], theta1_grid[i, j])
    ax1.plot_surface(theta0_grid, theta1_grid, values, cmap="viridis", edgecolor="none")
    ax1.set_title("Cost Function Surface")
    ax1.set_xlabel("Theta0")
    ax1.set_ylabel("Theta1")
    ax1.set_zlabel("Cost")
    plt.show()
```

## Le programme de prédiction

```python
import sys
import libft_linear_regression as lr

args = sys.argv
if len(args) != 2 or int(args[-1]) < 0 or int(args[-1]) > 1000000:
	print("Arguments provided are inconsistents. Please enter a number between 0 and 1000000.")
	exit(0)

mileage = int(args[-1])

dict_params = lr.read_model_parameters()
theta0 = dict_params["theta0"]
theta1 = dict_params["theta1"]

estimated_price = theta0 + theta1 * mileage

estimated_price = 0 if estimated_price < 0 else estimated_price

print("The estimated price of the model for a mileage of {} is : {} ".format(mileage, estimated_price))
```

## Le programme d’entraînement

```python
import sys
import numpy as np

from libft_linear_regression import *

args = sys.argv
if len(args) != 3 :
	print("Please enter valid args : python3 ft_linear_regression.py [file name].csv [flag bonus 0 or 1]")
	exit(0)

file_name = args[1]
flag = int(args[-1])

arr_datas = read_datas_to_array(file_name)

theta0 = 0
theta1 = 0
learningRate = 0.01
limit = 30000

print("Initial values :\ntheta0 = {}\ntheta1 = {}\nlearningRate = {}\ntraining_iterations = {}".format(
	theta0,
	theta1,
	learningRate,
	limit
))

# Min Max Normalization
arr_mileage_normalized = normalize_minmax_arr(arr_datas[:,0]).reshape((len(arr_datas[:,0])),1)
arr_price_normalized = normalize_minmax_arr(arr_datas[:,1]).reshape((len(arr_datas[:,1]),1))
arr_normalized_datas = np.concatenate([arr_mileage_normalized,arr_price_normalized], axis = 1)

count = 0
while (count < limit) :
	if flag == 1 and count % 10 == 0:
		cost = cost_fct(arr_normalized_datas, theta0,theta1)
		print(count, " | Fonction de cout : ", cost)
	tmp_theta0 = theta0 - learningRate * drv_cost_fct_theta0 (arr_normalized_datas, theta0,theta1)
	tmp_theta1 = theta1 - learningRate * drv_cost_fct_theta1 (arr_normalized_datas , theta0, theta1)
	theta0 = tmp_theta0
	theta1 = tmp_theta1
	count+=1

print("\nLinear regression : OK")
print("Normalized theta0 = {}\nNormalized theta1 = {}".format(theta0, theta1))

arr_estimated_price = estimatePrice(arr_normalized_datas[:,0], theta0, theta1)
arr_estimated_price_unormalized = unnormalize_minmax_arr(arr_datas[:,1] ,arr_estimated_price)

estimated_norm_price_max = estimatePrice(1, theta0, theta1)
estimated_norm_price_min = estimatePrice(0, theta0, theta1)

final_theta1 = (unnormalize_minmax(estimated_norm_price_max, arr_datas[:,1]) - unnormalize_minmax(estimated_norm_price_min, arr_datas[:,1])) / (arr_datas[:,0].max() - arr_datas[:,0].min())
final_theta0 = unnormalize_minmax(estimated_norm_price_min, arr_datas[:,1]) - final_theta1 * arr_datas[:,0].min()

with open("model_parameters.txt", 'w') as model_parameters_file :
	model_parameters_file.writelines([str(final_theta0), "\n", str(final_theta1)])

print("theta0 = ", final_theta0)
print("theta1 = ", final_theta1)

if flag == 1 :
	print("Fonction de coût = ", cost)
	display_values(arr_datas[:,0], arr_datas[:,1], 2)
	display_model(arr_datas[:,0], arr_estimated_price_unormalized,3)
	display_values_and_model(arr_mileage_normalized, arr_price_normalized, arr_estimated_price, 1)
	display_values_and_model(arr_datas[:,0], arr_datas[:,1], arr_estimated_price_unormalized, 4)
	display_cost_fct(arr_normalized_datas)
```

# Les résultats

## Quelques exemples de résultats de la fonction de coût avec différents learningRate et différents nombre d’itération

| learningRate | nbr d’itérations | Valeur de la fonction de coût |
| --- | --- | --- |
| 0.1 | 1300 | 0.020699401199204836 |
| 0.01 | 13000 | 0.0206993886698352 |
| 0.001 | 130000 | 0.020699387460695752 |

Les valeurs optimales des paramètres obtenus par le modèle sont :

- theta0 =  8499.598743566039
- theta1 =  -0.021448954971897145

## Représentation graphique

![Figure_4.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/4bc43969-1fca-4a09-8d55-3219240d78af/e473a9e5-1350-4a09-bf43-93b3fc463d25/Figure_4.png)

# Sources

## Theorie mathématique

[Qu'est ce que la dérivée ? Lê Nguyên Hoang](https://www.youtube.com/watch?v=mYJ5iPi9mfk)

[Formules de dérivation | StudySmarter](https://www.studysmarter.fr/resumes/mathematiques/analyse-mathematiques/formules-de-derivation/)

[STAD98_7](http://grasland.script.univ-paris-diderot.fr/STAT98/stat98_7/stat98_7.htm)

[Algorithme de descente de gradient](https://datascientest.com/descente-de-gradient)

[Comprendre la descente de gradient en 3 étapes et 12 dessins](https://www.charlesbordet.com/fr/gradient-descent/#et-les-maths-dans-tout-ca-)

[Machine Learning : développez votre première régression linéaire avec la descente de gradient](https://fr.blog.businessdecision.com/tutoriel-regression-lineaire-et-descente-de-gradient-en-machine-learning/)

## Tutoriaux Python

[Sort NumPy Arrays By Columns or Rows](https://opensourceoptions.com/sort-numpy-arrays-by-columns-or-rows/#google_vignette)

## Projets d’autres students

[GitHub - k-off/ft_linear_regression: An Algo Project at Codam (42) - Linear Regression](https://github.com/k-off/ft_linear_regression?tab=readme-ov-file)
