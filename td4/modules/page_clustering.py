import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from functools import lru_cache


class PageClusterer:
    def __init__(self, data_loader, n_clusters=7, seed=42):
        self.data_loader = data_loader
        self.n_clusters = n_clusters
        self.seed = seed
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

        vect = TfidfVectorizer(max_features=1000, stop_words='english')
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

        lr = LogisticRegression(max_iter=1000, random_state=self.seed)
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
        if not os.path.exists("models"):
            os.makedirs("models")

        with open("models/page_cluster_model.pkl", "wb") as f:
            pickle.dump(self._cache["page_cluster_model"], f)

        with open("models/page_vectorizer.pkl", "wb") as f:
            pickle.dump(self._cache["page_vectorizer"], f)

        with open("models/page_cluster_predictor.pkl", "wb") as f:
            pickle.dump(self._cache["page_cluster_predictor"], f)

    def load_models(self):
        """Charge les modèles sauvegardés"""
        with open("models/page_cluster_model.pkl", "rb") as f:
            self._cache["page_cluster_model"] = pickle.load(f)

        with open("models/page_vectorizer.pkl", "rb") as f:
            self._cache["page_vectorizer"] = pickle.load(f)

        with open("models/page_cluster_predictor.pkl", "rb") as f:
            self._cache["page_cluster_predictor"] = pickle.load(f)