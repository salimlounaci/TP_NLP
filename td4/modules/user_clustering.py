import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os


class UserClusterer:
    def __init__(self, data_loader, n_clusters=5, seed=42):
        self.data_loader = data_loader
        self.n_clusters = n_clusters
        self.seed = seed
        self._cache = {}

    def process_user_data(self):
        """Prépare les données utilisateur pour le clustering"""
        if "processed_user_data" in self._cache:
            return self._cache["processed_user_data"]

        # Récupération des données
        user_data, _, bid_data, _ = self.data_loader.get_data()

        # One-hot encoding des features catégorielles
        user_processed = pd.get_dummies(user_data, columns=['sex', 'city', 'device'])

        # Intégration des interactions utilisateur-page
        user_visits = (
            bid_data.groupby(["user_id", "page_id"])
            .size()
            .unstack(1)
            .fillna(0)
        )
        user_visits.columns = [str(c) for c in user_visits.columns]
        user_processed = user_processed.merge(user_visits, on='user_id', how='left')

        # Mise en cache
        self._cache["processed_user_data"] = user_processed

        return user_processed

    def clusterize_users(self):
        """Crée les clusters d'utilisateurs"""
        if "user_clusters" in self._cache:
            return self._cache["user_clusters"], self._cache["user_cluster_model"]

        user_processed = self.process_user_data()

        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        user_clusters = km.fit_predict(user_processed.drop('user_id', axis=1))

        user_processed = user_processed.copy()
        user_processed['cluster'] = user_clusters

        self._cache["user_clusters"] = user_processed
        self._cache["user_cluster_model"] = km

        return user_processed, km

    def save_models(self):
        """Sauvegarde le modèle de clustering"""
        if not os.path.exists("models"):
            os.makedirs("models")

        with open("models/user_cluster_model.pkl", "wb") as f:
            pickle.dump(self._cache["user_cluster_model"], f)

    def load_models(self):
        """Charge le modèle sauvegardé"""
        with open("models/user_cluster_model.pkl", "rb") as f:
            self._cache["user_cluster_model"] = pickle.load(f)