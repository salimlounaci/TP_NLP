import pandas as pd
import numpy as np


class FeatureBuilder:
    def __init__(self, data_loader, page_clusterer, user_clusterer):
        self.data_loader = data_loader
        self.page_clusterer = page_clusterer
        self.user_clusterer = user_clusterer
        self._cache = {}

    def build_click_features(self):
        """Construit les features pour la prédiction de clic"""
        if "click_features" in self._cache:
            return self._cache["click_features"]

        # Récupération des données
        _, _, _, click_data = self.data_loader.get_data()

        # Calcul des annonces vues par jour
        click_data = click_data.copy()
        click_data["date"] = click_data["timestamp"].apply(lambda txt: txt[:10])
        click_data["count"] = 1
        click_data["user_ads_seen"] = (
            click_data.groupby(["user_id", "date"])["count"]
            .cumsum()
        )

        click_data = click_data[["user_id", "page_id", "ad_id", "user_ads_seen", "clicked"]]

        # Récupération des clusters
        user_clusters, _ = self.user_clusterer.clusterize_users()

        # Association des clusters utilisateur
        click_features = click_data.merge(user_clusters[['user_id', 'cluster']], on='user_id', how='left')
        click_features = click_features.rename(columns={'cluster': 'user_cluster'})

        # Ajout des probabilités de cluster de page
        page_ids = click_features["page_id"].unique()
        page_to_cluster_prob = {}

        for page_id in page_ids:
            page_to_cluster_prob[page_id] = self.page_clusterer.get_page_cluster_probabilities(page_id)

        cluster_probs = [page_to_cluster_prob[page_id] for page_id in click_features["page_id"]]

        cluster_prob_df = pd.DataFrame(
            cluster_probs,
            columns=[f'page_cluster_prob_{i}' for i in range(self.page_clusterer.n_clusters)]
        )

        click_features = pd.concat(
            [click_features.reset_index(drop=True), cluster_prob_df.reset_index(drop=True)],
            axis=1,
        )

        self._cache["click_features"] = click_features

        return click_features

    def build_test_features(self, bid_requests_path):
        """Construit les features pour les données de test"""
        # Chargement des données de test
        bid_requests = self.data_loader.get_test_data(bid_requests_path)

        # Récupération des clusters utilisateur
        user_clusters, _ = self.user_clusterer.clusterize_users()

        # Association des clusters utilisateur
        # Utilisation d'un merge avec left pour conserver tous les utilisateurs, même ceux sans cluster
        test_features = bid_requests.merge(user_clusters[['user_id', 'cluster']], on='user_id', how='left')
        test_features = test_features.rename(columns={'cluster': 'user_cluster'})
        test_features['user_cluster'] = test_features['user_cluster'].fillna(0)  # Cluster par défaut

        # Ajout d'une valeur par défaut pour user_ads_seen
        test_features["user_ads_seen"] = 1

        # Ajout des probabilités de cluster de page
        page_ids = test_features["page_id"].unique()
        cluster_probs = []

        for page_id in test_features["page_id"]:
            probs = self.page_clusterer.get_page_cluster_probabilities(page_id)
            cluster_probs.append(probs)

        cluster_prob_df = pd.DataFrame(
            cluster_probs,
            columns=[f'page_cluster_prob_{i}' for i in range(self.page_clusterer.n_clusters)]
        )

        test_features = pd.concat(
            [test_features.reset_index(drop=True), cluster_prob_df.reset_index(drop=True)],
            axis=1,
        )

        return test_features