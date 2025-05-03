# Machine Learning Industrialization

## TD1: Industrialization de la pipeline data

Nous allons créer un data pipeline industrialisée. <br/>
Nous travaillons pour un vendeur de légumes, qui récupère les ventes hebdomadaires par une source externe. Nous nous intéressons, pour notre modèle, aux ventes mensuelles.

Nous allons créer une API avec les entry points:
- /post_sales/: reçoit une request POST avec data: une liste de dictionnaire {"date": ..., "vegetable": ...., "kilo_sold": ....} <br/>
Cette entrée est idempotente. On stockera ces données dans la table bronze
- /get_raw_sales/: renvoie les données brutes que nous avons reçues
- /get_monthly_sales/: renvoie les données cleanées (nom du légume standardisé. Cet entry point a l'option "remove_outliers". Si "remove_outliers=False", entraîne le modèle sur toutes les données. Si "remove_outliers=True", entraîne sur le modèle sur les données safe, qui n'ont pas été tagguées comme "unsafe" par nos algorithmes d'outlier detection.

Je fournis le code app.py, une app Flask basique avec les entry points /post_sales/, /get_raw_sales/ et /get_monthly_sales/. <br/>
Je fournis aussi le code client.py, qui va ping chacun des entry points. <br/>
Enfin, je fournis [un CSV sur ce lien](https://drive.google.com/file/d/1WJPZQEijYsfTga6il8Ls3pgjdsGhCrq0/view?usp=sharing) avec des ventes weekly de 2020 à 2023. Les noms des légumes peuvent être en anglais, français ou espagnol. Il peut y avoir une faute d'orthographe. Dans la base cleaned, ils doivent être en anglais, sans faute d'orthographe.

- Créer le code pour "post_sales" qui va stocker les données dans un CSV, est idempotent.
- Créer le code pour "get_raw_sales" qui retourne les données ingérées.
- Créer le code pour "compute_monthly_sales" qui prend des ventes weekly et les transforme en vente monthly. <br/>
Pour une semaine avec n jours sur un mois et 7-n jours sur le mois suivant, on considère que (n / 7)% des ventes étaient sur le mois précédent et ((7 - n) / 7)% sont sur le mois suivant.<br/>
Il est **fortement** suggéré de créer le test unitaire avec les cas problématiques (donnée en entrée, ce qu'on attend en sortie).
- Créer le code "tag_outlier" qui ajoute is_outlier=True si la vente est supérieure à la moyenne plus 5 fois l'écart-type. <br/>
Faut-il prendre moyenne globale ou légume par légume ? Pourquoi ?
- Créer la pipeline qui reçoit les données dans "post_sales", les écrit dans la table bronze, traduit les "vegetable" en anglais clean (table silver) et les transforme en monthly sales, avec le tag is_outlier True ou False, les écrit dans la table gold.
- Créer le code "/get_monthly_sales/" qui retourne les données gold, avec ou sans outliers
- Changer le modèle pour ne plus écrire dans un CSV, mais dans une table SQL Lite.<br/>
Si vous avez bien travaillé, le changement isolé et n'impacte pas les entry points ni la pipeline.<br/>
Si non, qu'auriez-vous dû changer dans votre implémentation pour pouvoir facilement changer de base de données ?
- Ajouter un entry point "/init_database" pour créer la database ou la vider. <br/>
- Utiliser le package "locust" pour faire un test de charge sur votre API.

Je testerai votre code en le faisant tourner dans un container, en appelant post_sales et les différents get_sales sur mes données. <br/>
Si vous êtes allés jusqu'au tables SQL, j'utiliserai "/init_database" pour ré-initialiser la base quand j'en aurais besoin.

Je testerai votre pipeline en la faisant tourner dans un Docker. Votre pipeline doit supporter:
- Ne pas inscrire des données où des champs sont faux (pas de vegetable)
- post_sales est bien idempotent pour une (year_month,vegetable)
- Inscrire toutes les données valables d'une liste. Si la liste contient des données "A: valide", "B: valide", "C: non valide", "D: valide", je dois retrouver dans la base les données A, B, et D.
- Tagger comme "outlier" les ventes à 5 écart-types de leur référence.
- Votre API doit supporter 1000 requêtes par seconde (500 post, 250 get_raw_data, 250 get_monthly_data).
- Vos tests d'intégration doivent couvrir les cas mentionnés ici.

**A rendre**: le code, avec un fichier app_csv.py où l'app fonctionne en enregistrant les données sur des CSV et app_sql.py où l'app enregistre sur une base de données sql lite<br/>
**Code à rendre à la fin du cour** (à 13h pour les IABD2, 17h15 pour les IABD1)
Envoyer à foucheta@gmail.com avec l'objet "[ESGI][ML_INDUS]TP1"

## TD2: Iteration sur un modèle

Dans ce TD, nous allons voir un problème d'apprentissage supervisé, sur lequel on va rajouter des sources de données et des features au fur et à mesure. <br/>
Nous voulons faire du code industrialisé, où nous pouvons itérer rapidement. <br/>
Il est fortement conseillé:
- De faire les étapes une par une, de "jouer le jeu" d'un projet qui évolue au cour du temps
- Pour chaque étape, de coder une solution, puis voir les refactos intéressantes. Attention: une erreur serait de se perdre en cherchant la perfection. 
- De coder des tests unitaires pour toutes les transformations de données "un poil" complexes
- De faire du code modulaire, avec:
  - Un module téléchargeant les données (un data catalogue)
  - Un module construisant les features. Chaque "feature" a son module
  - Un module générant le model. Tous les modèles ont les méthodes ".fit(X, y)", ".predict"

Vous avez le fichier de test tests/test_model.py avec les tests que je ferai. <br/>
Les tests appellent, dans main.py, la fonction "make_predictions(config: dict) -> df_pred: pd.Dataframe"


Télécharger [le dataset](https://drive.google.com/file/d/1OFDGVqlmx-5-hE3Bnn-996LGpumScwOV/view?usp=sharing). <br/>
Il s'agît de ventes mensuelles d'une industrie fictive.

### 1: Coder un modèle "SameMonthLastYearSales"

Le modèle prédit, pour les ventes de item N pour un mois, les mêmes ventes qu'il a faites l'année dernière au même mois (pour août 2024 les mêmes ventes que l'item N a eu en août 2023)

### 2: Coder un modèle auto-regressif.
Les données ont été générées comme une combinaison des ventes le même mois l'année dernière, des ventes moyennes sur l'année dernière, et des ventes du même mois l'année dernière fois la croissance du quarter Q-5 au quarter Q-1

$$sales(M) = a \times sales(M-12) + b \times sales(M-1:M-12) / 12 \\ + c \times sales(M-12) \frac{sales(M-1:M-3)}{sales(M-13:M-15)}$$

Coder le "build_feature" qui va générer ces différentes features autoregressive. <br/>
Utiliser le modèle sklearn Ridge()

### 3: Ajouter les données marketing.

Les mois où il y a eu des dépenses marketing, cela a impacté les ventes.

Les données ont été générées ainsi

$$ sales(M) = ...past model... * (1 + marketing * d) $$

### 4: Ajouter les données de prix

Les clients, des grossistes, sont prévenus en avance d'un changement de prix. <br/>
Si le prix va augmenter le mois suivant M+1, ils commandent plus que d'habitude au mois M, et moins au mois M+1. <br/<
A l'inverse, si le prix va baisser, ils commandent moins au mois M et plus à M+1.

### 5: Ajouter les données de stock

Certains mois, l'industriel a eu des ruptures de stocks et donc a vendu moins que ce qu'il aurait pu. Le mois suivant, il a plus vendu car les clients ont racheté ce qu'ils devaient pour leur consommation. <br/>

stock.csv contient les "refill" de stock quotidien. On suppose que le stock initial était 0. <br/>
Il y a rupture de stock si le stock est 0 à la fin du mois. <br/>
En ayant identifié les ruptures de stock, vous pouvez décider de ne pas entraîner sur les mois où les ruptures de stocks ont eu un effet (le mois de la rupture et le mois suivant). <br/>

On sait en avance les refill de stocks qu'on aura. <br/>
Donc, on peut améliorer nos prédictions de cette façon:

$$ \tilde{pred}(item_i, month_M) = \min(stock(item_i, month_M), pred(item_i, month_M)) $$

### 6: Ajouter les objectifs des commerciaux.

Les commerciaux ont des objectifs de vente à l'année. L'année fiscal se terminant en juin, c'est ce mois, et le mois suivant, qui sont impactés. <br/>
Si l'item a déjà fait son objectif, où est loin de le faire (resterait 20% des ventes à faire), il n'y a pas d'impact. <br/>
Sinon, l'équipe commercial va faire tout son possible pour arriver à l'objectif, demandant à leurs clients de sur-acheter en juin. Du coup, il y a un sous-achat en juillet compensant la sur-vente de juin.

Intégrer les données des objectifs à votre pipeline de prédiction.

### 7: Faire un modèle custom

La génération des données a été faite ainsi. J'ai généré des données autoregressées ainsi:

$$sales\_v1(M) = a * sales(M-12) + b * sales(M-1:M-12) / 12 + c * sales(M-12) \frac{sales(M-1:M-3)}{sales(M-13:M-15)}$$

Ca fait, j'ai rajouté les effets:

$$ sales\_v2(M)  = sales\_v1(M) * (1 + d * marketing ) * (1 + e * price\_change) $$

J'ai ensuite ajouté, au hasard sur certains mois, des contraintes "objectifs commerciaux", puis des contraintes de stock.

Vous pouvez faire votre propre modèle qui reprend ces équations, avec les paramètres a, b, c....,e, et utilser scipy.optimize pour trouver les paramètres idéaux.

Side note: les "items" ont des ventes moyennes différentes. <br/>
Sans scaling, votre modèle "fit" surtout le top1 item ou top10 item. <br/>
Peut-être qu'un scaling permettra de mieux estimer les paramètres a, b, c...

## TD3: Logging & monitoring

Dans ce TD, nous allons prendre une app, sans log ni monitoring, prédisant le sentiment d'un utilisateur selon sa review.<br/>
Le but est d'ajouter des logs et du monitoring pour comprendre ce qui s'est mal passé. <br/>
Nous n'allons pas installer une toolbox, juste essayer de poser du "logging.info/debug/error(...logging message...)". <br/>
Si on joue le jeu, on essaie de lire, dans les logs (qui peuvent être écrit dans un fichier .log) ce qui a posé problème.

Dans le dossier TD3, vous trouverez 2 fichiers: app.py et test_app.py.<br/>
test_app va vraiment appeler l'API. <br/>
Il faut
```python app.py```
Pour rendre l'API disponible.<br/>
Ensuite on peut lancer les tests.

Dans ce TD, on va essayer de comprendre quels cas posent quels problèmes avec le logging et le monitoring. <br/>
Pour l'instant, "test_app.py" ne lance que "test_normal_case", qui est sensé marcher. <br/>

Essayez, sans découvrir les autres cas de tests, d'ajouter du logging sur app.py qui vous paraît normal:
- Avoir le temps de réponse
- Mesurer CPU / RAM / Memory (ici, on va juste mesurer "Memory". Regarder la fonction "health_check" pour voir ce que je considère comme "Memory"
- Ajouter un ID à chaque requête, pour pouvoir tracer ce qui s'est passé
- Avoir dans les logs les infos pour pouvoir rejouer le cas qui a échoué
- Essayer de mesurer data drift (la différence entre les données d'entraînement et les données en prod)

Une fois votre logging mis en place, vous pouvez débloquer les test cases du fichier "test_app.py" en transformant les noms de fonctions "def tst_case_xxx" en "def test_case_xxx".

Livrable: A la fin du TD, envoyez-moi un document avec les problèmes qui étaient contenus dans cette app et, pour chaque problème, quel logging aurait permis de le découvrir. (si vous ne savez pas quel logging aurait aider, vous pouvez le mettre). <br/>
(Ce document n'a pas besoin d'être très long.)

## TD4: Refactoring dirty ML code

Dans ce TD, nous allons supposer qu'un data scientist est arrivé à un bon modèle, sans se soucier des règles d'industrialisation. Nous allons refactor ce code pour qu'il soit testé et modulaire **avec méthode!**

Voici [un zip avec des data](https://drive.google.com/file/d/1c6lRYoXXRQW9T4i1nKZC4H2qmd1CiYrE/view?usp=sharing)

Le code à refactorer est dans td4/script.py

### Contexte

Nous travaillons dans une start-up d'adtech. Lorsqu'un utilisateur arrive sur une page, on nous envoie une requête d'enchère (bid) pour un "user" sur une "page". Nous devons envoyer une "ad" ainsi que le "bid" (combien nous sommes prêts à payer pour diffuser notre pub à cet utilisateur sur cette page).

#### Data

Nous disposons:
- D'un historique des requêtes (user, page, timestamp) que nous avons reçu (bid_request.csv)
- D'information sur les utilisateurs (user_data.csv)
- Du text contenu dans les pages (page_data.csv)
- Pour les enchères que nous avons remportées, nous savons si l'utilisateur a cliqué sur la pub ou non (click_data.csv)

#### Modèle

L'idée du data scientist, pour avoir un bon prédicteur, et d'essayer de clusteriser les pages et les utilisateurs. Il doit exister plusieurs catégories de pages (pages de sport, de news, de théâtre, de gossip...). En clusterisant les pages selon les utilisateurs qui les ont visitées, on doit retrouvé des centres d'intérêts.

De même, on peut clusteriser les utilisateurs selon les pages qu'ils visitent.

En plus, pour les pages, une fois clusterisées, il apprend une régression logistique (texte de la page) -> (probabilité d'appartenir à chacun des clusters).

Nous entraînons ensuite une régression logistique sur les clicks data (user_id, page_id, ad_id, ad clicked or not) sur les features suivantes:
- Le nombre de publicités que l'utilisateur a vu jusqu'ici
- le cluster de l'utilisateur
- la probabilité d'être dans chacun des clusters pour la page

#### Problèmes du code

- C'est un long script
- On a des variables globales inscrites en dur
- Les modèles sont dumpés en local dans des fichiers inscrits en dur.

### Vos tâches

#### End to end test

Tout d'abord, on va s'assurer qu'on ne casse rien. Nous allons donc créer un end-to-end test runnant toute la pipeline, et le résultat ne doit pas changer.

Créer un test qui aura cette structure là:

```
def test_end_to_end():
    data_test = ...load_data_test...
    result_expected = ...load_expected_result...
    
    train_model()
    result = predict(data_test)

    ...assert result == result_expected...
```

Pour créer "result expected", faites tourner la pipeline et dumper, dans des CSVs ou des pickles les outcomes que vous voulez comparer. <br/>
Quels outcomes voulez-vous comparer ?<br/>
A vous de déterminer. En tant que CTO de l'adtech, je veux **au moins** garder les mêmes predictions sur user_id, page_id, ad_id. On peut se dire qu'on sauvegarde des résultats intermédiaires (page_cluster_proba) mais notre seul vrai besoin, ce sont les prédictions (user_id, page_id, ad_id) -> probability to be clicked. <br/>
Je préfère assert sur les données résultats (CSVs ou .parquet) que sur les .pickles (les modèles sont les moyens d'avoir le résultat. Ce ne sont pas eux qui m'intéressent).

Vous pouvez refactorer "script.py" pour qu'il ait un entry point clair "train_model" et "predict(test_data)".

Pour que le test d'intégration prenne peu de temps, vous pouvez créer un dataset_train de petite taille (et tous les calculs seront plus rapides).

#### Modularité

Maintenant que nous sommes assurés de ne rien casser, nous allons séparer le code en module:
- Un module pour loader les données
- Un module qui sort les user clusters
- Un module qui sort les page clusters proba
- Un module qui construit le dataframe pour notre prédicteur (user_id, page_id, ad_id) -> proba to be clicked

#### Configuration

Créer un système de configuration gérant:
- Les chemins aux données
- Les chemins aux modèles
- Les variables paramètres

#### Unit-test

Une erreur s'est glissée dans le calcul d'une feature. <br/>
Créer des tests unitaires pour vous assurer que chaque feature est bien calculée et trouvez l'erreur.

Si vous trouvez cette partie compliquée et que vous vous demandez comment vous y prendre, c'est un bon moment pour demander de l'aide au professeur. <br/>
Les tests sont un vaste sujet, mais ils sont cruciaux pour l'industrialisation de machine learning.

### Livrable

Envoyez-moi votre code zippé à foucheta@gmail.com, avec le sujet [ESGI][ML_INDUS] TD4.
Envoyez moi aussi un **court** document expliquant ce que vous avez fait pour les parties "modularités", "configuration", "unit-test" (où était l'erreur)
