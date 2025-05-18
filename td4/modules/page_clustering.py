import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from functools import lru_cache
from modules.config import Config


class PageClusterer:
    def __init__(self, data_loader, config_path=None):
        self.data_loader = data_loader
        self.config = Config(config_path)
        self.n_clusters = self.config.get_param("model_params", "page_clusters")
        self.seed = self.config.get_param("model_params", "seed")
        self.max_features = self.config.get_param("model_params", "max_features")
        self.max_iter = self.config.get_param("model_params", "max_iter")
        self._cache = {}

    def preprocess_text(self, text_series):
        """Prétraitement du texte"""
        text_series = text_series.fillna("")
        text_series = text_series.str.lower()
        return text_series

    def clusterize_pages(self):
        """Crée les clusters de pages"""
        if "page_clusters" in self._cache:
            return self._cache["page_clusters"], self._cache["page_cluster_model"], self._cache["page_vectorizer"]

        _, page_data, _, _ = self.data_loader.get_data()

        vect = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        X_pages = vect.fit_transform(self.preprocess_text(page_data['page_text']))

        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        page_clusters = km.fit_predict(X_pages)

        page_data = page_data.copy()
        page_data['cluster'] = page_clusters

        self._cache["page_clusters"] = page_data
        self._cache["page_cluster_model"] = km
        self._cache["page_vectorizer"] = vect

        return page_data, km, vect

    def train_page_cluster_predictor(self):
        """Entraîne un prédicteur de cluster de page"""
        page_data, _, vect = self.clusterize_pages()

        X_pages = vect.transform(self.preprocess_text(page_data['page_text']))
        y = page_data['cluster']

        lr = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)
        lr.fit(X_pages, y)

        self._cache["page_cluster_predictor"] = lr

        return lr

    @lru_cache(maxsize=1000)
    def get_page_cluster_probabilities(self, page_id):
        """Renvoie les probabilités d'appartenance d'une page à chaque cluster"""
        page_data, _, vect = self.clusterize_pages()

        lr = self._cache.get("page_cluster_predictor")
        if not lr:
            lr = self.train_page_cluster_predictor()

        try:
            page_text = page_data[page_data['page_id'] == page_id]['page_text'].values[0]

            X = vect.transform([self.preprocess_text(pd.Series([page_text]))[0]])

            probs = lr.predict_proba(X)[0]

            return probs
        except (IndexError, KeyError):
            # Si page_id n'existe pas, retourne des zéros
            return np.zeros(self.n_clusters)

    def save_models(self):
        """Sauvegarde les modèles"""
        models_dir = self.config.config["paths"]["models_dir"]
        os.makedirs(models_dir, exist_ok=True)

        with open(self.config.get_model_path("page_cluster_model"), "wb") as f:
            pickle.dump(self._cache["page_cluster_model"], f)

        with open(self.config.get_model_path("page_vectorizer"), "wb") as f:
            pickle.dump(self._cache["page_vectorizer"], f)

        with open(self.config.get_model_path("page_cluster_predictor"), "wb") as f:
            pickle.dump(self._cache["page_cluster_predictor"], f)

    def load_models(self):
        """Charge les modèles sauvegardés"""
        with open(self.config.get_model_path("page_cluster_model"), "rb") as f:
            self._cache["page_cluster_model"] = pickle.load(f)

        with open(self.config.get_model_path("page_vectorizer"), "rb") as f:
            self._cache["page_vectorizer"] = pickle.load(f)

        with open(self.config.get_model_path("page_cluster_predictor"), "rb") as f:
            self._cache["page_cluster_predictor"] = pickle.load(f)