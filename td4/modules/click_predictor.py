import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from modules.config import Config


class ClickPredictor:
    def __init__(self, feature_builder, config_path=None):
        self.feature_builder = feature_builder
        self.config = Config(config_path)
        self.seed = self.config.get_param("model_params", "seed")
        self.max_iter = self.config.get_param("model_params", "max_iter")
        self.threshold = self.config.get_param("training", "threshold")
        self.test_split = self.config.get_param("training", "test_split")
        self.model = None

    def train(self):
        """Entraîne le modèle de prédiction de clic"""
        # Construction des features
        click_features = self.feature_builder.build_click_features()

        # Vérification si les features ont été correctement construites
        if click_features is None:
            print("ERROR: build_click_features() returned None - cannot train model")
            return None

        print(click_features.head())

        # Séparation des features et de la cible
        X = click_features.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
        y = click_features['clicked']

        # Entraînement du modèle
        lr = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)
        lr.fit(X, y)

        self.model = lr

        return lr

    def evaluate(self):
        """Évalue les performances du modèle"""
        # Construction des features
        click_features = self.feature_builder.build_click_features()

        # Vérification si les features ont été correctement construites
        if click_features is None:
            print("ERROR: build_click_features() returned None - cannot evaluate model")
            return 0.0

        # Split train/test aléatoire
        msk = np.random.rand(len(click_features)) < (1 - self.test_split)
        train = click_features[msk]
        test = click_features[~msk]

        X_train = train.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
        y_train = train['clicked']

        X_test = test.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
        y_test = test['clicked']

        # Entraînement sur le subset de training
        lr = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)
        lr.fit(X_train, y_train)

        # Évaluation sur le subset de test
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def predict_batch(self, test_features):
        """Effectue la prédiction sur un lot de données"""
        # Vérification que le modèle est chargé
        if self.model is None:
            print("Model not loaded, attempting to train...")
            self.model = self.train()

            if self.model is None:
                print("ERROR: Failed to train model")
                return None, None

        # Préparation des features pour la prédiction
        X_test = test_features.drop(['user_id', 'page_id', 'timestamp'], axis=1, errors='ignore')

        # Vérification que toutes les colonnes attendues sont présentes
        missing_columns = set(self.model.feature_names_in_) - set(X_test.columns)
        if missing_columns:
            print(f"Adding missing columns: {missing_columns}")
            for col in missing_columns:
                X_test[col] = 0

        # S'assurer que les colonnes sont dans le même ordre que lors de l'entraînement
        X_test = X_test[self.model.feature_names_in_]

        # Prédiction des probabilités
        probas = self.model.predict_proba(X_test)[:, 1]

        # Application du seuil de classification
        predictions = (probas >= self.threshold).astype(int)

        return predictions, probas

    def predict_single(self, user_id, page_id, ad_id=None):
        """Prédit la probabilité de clic pour une seule instance"""
        # Vérification que le modèle est chargé
        if self.model is None:
            self.load_model()

        # Récupération des caractéristiques de l'utilisateur et de la page
        user_clusterer = self.feature_builder.user_clusterer
        page_clusterer = self.feature_builder.page_clusterer

        user_clusters, _ = user_clusterer.clusterize_users()

        # Trouver le cluster de l'utilisateur
        try:
            user_cluster = user_clusters[user_clusters['user_id'] == user_id]['cluster'].values[0]
        except IndexError:
            user_cluster = 0  # Cluster par défaut si l'utilisateur n'est pas trouvé

        # Obtenir les probabilités de cluster de page
        page_probs = page_clusterer.get_page_cluster_probabilities(page_id)

        # Construire le vecteur de features
        features = {}
        features['user_cluster'] = user_cluster
        features['user_ads_seen'] = 1  # Valeur par défaut

        # Ajouter les probabilités de cluster de page
        for i, prob in enumerate(page_probs):
            features[f'page_cluster_prob_{i}'] = prob

        # Convertir en DataFrame
        features_df = pd.DataFrame([features])

        # S'assurer que toutes les colonnes attendues sont présentes
        for col in self.model.feature_names_in_:
            if col not in features_df.columns:
                features_df[col] = 0

        # Réorganiser les colonnes selon l'ordre attendu par le modèle
        features_df = features_df[self.model.feature_names_in_]

        # Prédiction
        prob = self.model.predict_proba(features_df)[0][1]

        return prob

    def save_model(self):
        """Sauvegarde le modèle de prédiction"""
        if self.model is None:
            print("No model to save. Training first...")
            self.train()

            if self.model is None:
                print("ERROR: Failed to train model. Cannot save.")
                return

        models_dir = self.config.config["paths"]["models_dir"]
        os.makedirs(models_dir, exist_ok=True)
        print(f"Saving model to {models_dir}")

        model_path = self.config.get_model_path("click_predictor")
        print(f"Model path: {model_path}")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Model saved successfully to {model_path}")

    def load_model(self):
        """Charge le modèle de prédiction sauvegardé"""
        try:
            model_path = self.config.get_model_path("click_predictor")
            print(f"Loading model from {model_path}")

            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at {model_path}")
                print("Training a new model instead...")
                self.train()
                return

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            print("Model loaded successfully")
        except Exception as e:
            import traceback
            print(f"ERROR loading model: {e}")
            print(traceback.format_exc())
            print("Training a new model instead...")
            self.train()