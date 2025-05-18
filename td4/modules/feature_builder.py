import pandas as pd
import numpy as np
from modules.config import Config


class FeatureBuilder:
    def __init__(self, data_loader, page_clusterer, user_clusterer, config_path=None):
        self.data_loader = data_loader
        self.page_clusterer = page_clusterer
        self.user_clusterer = user_clusterer
        self.config = Config(config_path) if config_path is not None else None
        self._cache = {}

    def build_click_features(self):
        """Construit les features pour la prédiction de clic"""
        # Utilisation du cache si disponible
        if "click_features" in self._cache:
            return self._cache["click_features"]

        # Récupération des données
        try:
            user_data, page_data, bid_data, click_data = self.data_loader.get_data()

            # Vérification que les données sont bien chargées
            if click_data is None:
                print("ERROR: click_data is None from data_loader.get_data()")
                return None

            # Calcul des annonces vues par jour
            click_data = click_data.copy()
            click_data["date"] = click_data["timestamp"].apply(lambda txt: str(txt)[:10])
            click_data["count"] = 1
            click_data["user_ads_seen"] = (
                click_data.groupby(["user_id", "date"])["count"]
                .cumsum()
            )

            # Sélection des colonnes pertinentes
            click_data = click_data[["user_id", "page_id", "ad_id", "user_ads_seen", "clicked"]]

            # Récupération des clusters utilisateur
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

            n_clusters = self.page_clusterer.n_clusters
            cluster_prob_df = pd.DataFrame(
                cluster_probs,
                columns=[f'page_cluster_prob_{i}' for i in range(n_clusters)]
            )

            # Concaténation des features
            click_features = pd.concat(
                [click_features.reset_index(drop=True), cluster_prob_df.reset_index(drop=True)],
                axis=1,
            )

            # Mise en cache du résultat
            self._cache["click_features"] = click_features

            return click_features

        except Exception as e:
            import traceback
            print(f"ERROR in build_click_features: {e}")
            print(traceback.format_exc())
            return None

    def build_test_features(self, bid_requests_path):
        """Construit les features pour les données de test"""
        try:
            print(f"Loading test data from: {bid_requests_path}")

            # Vérifier si le fichier existe
            import os
            if not os.path.exists(bid_requests_path):
                print(f"ERROR: Test data file not found at {bid_requests_path}")
                return None

            # Chargement des données de test
            bid_requests = self.data_loader.get_test_data(bid_requests_path)

            if bid_requests is None:
                print("ERROR: Test data loading returned None")
                return None

            print(f"Test data loaded with {len(bid_requests)} rows")
            print(f"Columns: {bid_requests.columns.tolist()}")

            # Récupération des clusters utilisateur
            print("Getting user clusters...")
            user_clusters, _ = self.user_clusterer.clusterize_users()

            if user_clusters is None:
                print("ERROR: User clustering returned None")
                return None

            print(f"User clusters obtained with {len(user_clusters)} users")

            # Association des clusters utilisateur
            print("Merging with user clusters...")
            test_features = bid_requests.merge(user_clusters[['user_id', 'cluster']], on='user_id', how='left')
            test_features = test_features.rename(columns={'cluster': 'user_cluster'})

            # Gestion des utilisateurs absents des données d'entraînement
            null_count = test_features['user_cluster'].isnull().sum()
            if null_count > 0:
                print(f"WARNING: {null_count} rows have null user_cluster after merge")
                test_features['user_cluster'] = test_features['user_cluster'].fillna(0)

            # Ajout d'une valeur par défaut pour user_ads_seen
            test_features["user_ads_seen"] = 1

            # Ajout des probabilités de cluster de page
            print("Adding page cluster probabilities...")
            page_ids = test_features["page_id"].unique()
            print(f"Found {len(page_ids)} unique page IDs")

            # Création d'un dictionnaire de probabilités par page
            page_to_cluster_prob = {}
            for page_id in page_ids:
                page_to_cluster_prob[page_id] = self.page_clusterer.get_page_cluster_probabilities(page_id)

            # Création de la liste des probabilités pour chaque ligne
            print("Creating probability vectors for each page...")
            cluster_probs = []
            for page_id in test_features["page_id"]:
                if page_id in page_to_cluster_prob:
                    cluster_probs.append(page_to_cluster_prob[page_id])
                else:
                    # Page non trouvée, utiliser des zéros
                    print(f"WARNING: Page ID {page_id} not found in probability dictionary")
                    cluster_probs.append(np.zeros(self.page_clusterer.n_clusters))

            # Création du DataFrame de probabilités
            print("Creating probability DataFrame...")
            cols = [f'page_cluster_prob_{i}' for i in range(self.page_clusterer.n_clusters)]
            cluster_prob_df = pd.DataFrame(cluster_probs, columns=cols)

            # Réinitialisation des index pour la concaténation
            test_features = test_features.reset_index(drop=True)
            cluster_prob_df = cluster_prob_df.reset_index(drop=True)

            # Concaténation
            print("Concatenating features...")
            test_features = pd.concat([test_features, cluster_prob_df], axis=1)

            print(f"Final test features: {test_features.shape}")

            return test_features
        except Exception as e:
            import traceback
            print(f"ERROR in build_test_features: {e}")
            print(traceback.format_exc())
            return None