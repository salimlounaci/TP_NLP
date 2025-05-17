import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os


class ClickPredictor:
    def __init__(self, feature_builder, seed=42):
        self.feature_builder = feature_builder
        self.seed = seed
        self.model = None

    def train(self):
        """Entraîne le modèle de prédiction de clic"""
        click_features = self.feature_builder.build_click_features()

        # Séparation des features et de la cible
        X = click_features.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
        y = click_features['clicked']

        # Entraînement du modèle
        lr = LogisticRegression(max_iter=1000, random_state=self.seed)
        lr.fit(X, y)

        self.model = lr

        return lr

    def evaluate(self):
        """Évalue les performances du modèle"""
        click_features = self.feature_builder.build_click_features()

        # Split train/test aléatoire
        msk = np.random.rand(len(click_features)) < 0.8
        train = click_features[msk]
        test = click_features[~msk]

        X_train = train.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
        y_train = train['clicked']

        X_test = test.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
        y_test = test['clicked']

        # Entraînement sur le subset de training
        lr = LogisticRegression(max_iter=1000, random_state=self.seed)
        lr.fit(X_train, y_train)

        # Évaluation sur le subset de test
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def predict_batch(self, test_features):
        """Effectue la prédiction sur un lot de données"""
        if self.model is None:
            self.model = self.train()

        # Préparation des features pour la prédiction
        X_test = test_features.drop(['user_id', 'page_id', 'timestamp'], axis=1, errors='ignore')

        # Vérification que toutes les colonnes attendues sont présentes
        for feature in self.model.feature_names_in_:
            if feature not in X_test.columns:
                X_test[feature] = 0  # Ajout des colonnes manquantes avec une valeur par défaut

        # Réorganisation des colonnes pour correspondre au modèle
        X_test = X_test[self.model.feature_names_in_]

        # Prédiction
        probas = self.model.predict_proba(X_test)[:, 1]
        predictions = (probas >= 0.5).astype(int)

        return predictions, probas

    def save_model(self):
        """Sauvegarde le modèle de prédiction"""
        if self.model is None:
            self.train()

        if not os.path.exists("models"):
            os.makedirs("models")

        with open("models/click_predictor.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self):
        """Charge le modèle sauvegardé"""
        with open("models/click_predictor.pkl", "rb") as f:
            self.model = pickle.load(f)