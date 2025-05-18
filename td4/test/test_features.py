import unittest
import pandas as pd
import numpy as np
import os
import sys
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.config import Config


class TestFeatureBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuration initiale exécutée une fois avant tous les tests"""
        cls.config = Config()
        cls.data_loader = DataLoader()
        cls.page_clusterer = PageClusterer(cls.data_loader)
        cls.user_clusterer = UserClusterer(cls.data_loader)
        cls.feature_builder = FeatureBuilder(cls.data_loader, cls.page_clusterer, cls.user_clusterer)

        # Création des données de test manuellement si nécessaire
        cls.create_test_data()

    @classmethod
    def create_test_data(cls):
        """Création de données de test si elles n'existent pas"""
        data_dir = Path(cls.config.get_param("paths", "data_dir"))
        data_dir.mkdir(exist_ok=True)

        # Vérifier si les fichiers existent déjà
        if (data_dir / cls.config.get_param("files", "user_data")).exists():
            # Les fichiers existent déjà, pas besoin de les recréer
            return

        # Créer des données de test minimales
        user_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 40],
            'sex': ['M', 'F', 'M'],
            'city': ['Paris', 'Lyon', 'Marseille'],
            'device': ['mobile', 'desktop', 'tablet']
        })

        page_data = pd.DataFrame({
            'page_id': [101, 102, 103],
            'page_text': ['sports news football', 'finance markets stocks', 'food recipes cooking']
        })

        bid_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'page_id': [101, 102, 101, 103, 102],
            'timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00',
                          '2023-01-01 12:00:00', '2023-01-01 13:00:00',
                          '2023-01-01 14:00:00']
        })

        click_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'page_id': [101, 102, 101, 103, 102],
            'ad_id': [201, 202, 201, 203, 202],
            'timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00',
                          '2023-01-01 12:00:00', '2023-01-01 13:00:00',
                          '2023-01-01 14:00:00'],
            'clicked': [1, 0, 1, 0, 1]
        })

        # Sauvegarder les données
        user_data.to_csv(data_dir / cls.config.get_param("files", "user_data"), index=False)
        page_data.to_csv(data_dir / cls.config.get_param("files", "page_data"), index=False)
        bid_data.to_csv(data_dir / cls.config.get_param("files", "bid_data_train"), index=False)
        click_data.to_csv(data_dir / cls.config.get_param("files", "click_data_train"), index=False)

        # Créer des données de test similaires
        bid_data_test = pd.DataFrame({
            'user_id': [1, 2, 3],
            'page_id': [101, 102, 103],
            'timestamp': ['2023-01-02 10:00:00', '2023-01-02 11:00:00', '2023-01-02 12:00:00']
        })

        bid_data_test.to_csv(data_dir / cls.config.get_param("files", "bid_data_test"), index=False)


    def test_click_features_user_ads_seen(self):
        """Test que la colonne user_ads_seen est correctement calculée"""
        self.feature_builder._cache = {}
        self.data_loader._cache = {}

        click_features = self.feature_builder.build_click_features()

        self.assertIn('user_ads_seen', click_features.columns,
                      "La colonne 'user_ads_seen' n'existe pas dans les features")

        user_data, _, _, click_data = self.data_loader.get_data()

        click_data_test = click_data.copy()
        click_data_test["date"] = pd.to_datetime(click_data_test["timestamp"]).dt.date
        click_data_test["count"] = 1
        manual_user_ads_seen = click_data_test.groupby(["user_id", "date"])["count"].cumsum()

        # Joindre ces valeurs manuelles au DataFrame original pour comparaison
        click_data_test['manual_ads_seen'] = manual_user_ads_seen

        # Fusionner avec le DataFrame de features pour comparer
        joined = click_features.merge(
            click_data_test[['user_id', 'page_id', 'ad_id', 'manual_ads_seen']],
            on=['user_id', 'page_id', 'ad_id'],
            how='left'
        )

        # Vérifier que la feature calculée correspond à notre calcul manuel
        valid_rows = joined.dropna(subset=['manual_ads_seen'])

        # Vérifier l'égalité élément par élément
        for _, row in valid_rows.iterrows():
            self.assertEqual(
                row['user_ads_seen'], row['manual_ads_seen'],
                f"user_ads_seen incorrect pour user_id={row['user_id']}, "
                f"expected={row['manual_ads_seen']}, got={row['user_ads_seen']}"
            )
    def test_click_features_user_cluster(self):
        """Test que la colonne user_cluster est correctement calculée"""
        click_features = self.feature_builder.build_click_features()

        # Vérifier que la colonne existe
        self.assertIn('user_cluster', click_features.columns,
                      "La colonne 'user_cluster' n'existe pas dans les features")

        # Vérifier que chaque utilisateur a un cluster assigné
        user_ids = click_features['user_id'].unique()
        for user_id in user_ids:
            # Tous les enregistrements pour cet utilisateur doivent avoir le même cluster
            user_rows = click_features[click_features['user_id'] == user_id]
            unique_clusters = user_rows['user_cluster'].unique()
            self.assertEqual(len(unique_clusters), 1,
                             f"L'utilisateur {user_id} a plusieurs clusters: {unique_clusters}")

    def test_click_features_page_cluster_probabilities(self):
        """Test que les probabilités de cluster de page sont correctement calculées"""
        click_features = self.feature_builder.build_click_features()

        # Vérifier que les colonnes de probabilité existent
        n_clusters = self.page_clusterer.n_clusters
        for i in range(n_clusters):
            col_name = f'page_cluster_prob_{i}'
            self.assertIn(col_name, click_features.columns,
                          f"La colonne '{col_name}' n'existe pas dans les features")

        # Vérifier que les probabilités somment à 1 pour chaque page
        page_ids = click_features['page_id'].unique()
        for page_id in page_ids:
            page_rows = click_features[click_features['page_id'] == page_id].iloc[0]

            # Extraire les probabilités de cluster
            probs = [page_rows[f'page_cluster_prob_{i}'] for i in range(n_clusters)]

            # Vérifier que la somme est proche de 1 (à cause des erreurs d'arrondi)
            self.assertAlmostEqual(sum(probs), 1.0, places=5,
                                   msg=f"La somme des probabilités pour la page {page_id} n'est pas égale à 1")

    def test_build_test_features(self):
        """Test que les features de test sont correctement construites"""
        bid_requests_path = self.config.get_data_path("bid_data_test")
        test_features = self.feature_builder.build_test_features(bid_requests_path)

        # Vérifier que le résultat n'est pas None
        self.assertIsNotNone(test_features, "Les features de test sont None")

        # Vérifier que toutes les colonnes nécessaires sont présentes
        required_columns = ['user_id', 'page_id', 'user_cluster', 'user_ads_seen']
        for col in required_columns:
            self.assertIn(col, test_features.columns, f"La colonne '{col}' manque dans les features de test")

        # Vérifier les colonnes de probabilité de cluster
        n_clusters = self.page_clusterer.n_clusters
        for i in range(n_clusters):
            col_name = f'page_cluster_prob_{i}'
            self.assertIn(col_name, test_features.columns,
                          f"La colonne '{col_name}' manque dans les features de test")


    def test_user_ads_seen_calculation(self):
        """Test spécifique pour vérifier le calcul de user_ads_seen"""
        click_data = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 2, 2, 2],
            'page_id': [101, 102, 103, 104, 201, 202, 203],
            'ad_id': [201, 202, 203, 204, 301, 302, 303],
            'timestamp': [
                "2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00",
                "2023-01-02 09:00:00",  # Nouvelle date pour utilisateur 1
                "2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-02 09:00:00"  # Utilisateur 2
            ],
            'clicked': [0, 1, 0, 1, 0, 1, 0]
        })

        data_dir = Path(self.config.get_param("paths", "data_dir"))
        temp_file = data_dir / "temp_click_data.csv"
        click_data.to_csv(temp_file, index=False)

        try:
            # Charger les données dans le data_loader
            self.data_loader._cache["click_data"] = click_data

            click_data_copy = click_data.copy()
            click_data_copy["date"] = pd.to_datetime(click_data_copy["timestamp"]).dt.date
            click_data_copy["count"] = 1
            user_ads_seen_correct = click_data_copy.groupby(["user_id", "date"])["count"].cumsum().tolist()

            user_ads_seen_incorrect = click_data_copy.groupby(["user_id"])["count"].cumsum().tolist()

            self.feature_builder._cache = {}
            click_features = self.feature_builder.build_click_features()

            click_features_sorted = click_features.sort_values(["user_id", "page_id"]).reset_index(drop=True)

            calculated_values = click_features_sorted['user_ads_seen'].tolist()

            # Ces valeurs devraient être différentes
            self.assertNotEqual(user_ads_seen_correct, user_ads_seen_incorrect,
                                "Les données de test devraient montrer une différence entre les méthodes")

            # Le résultat calculé devrait correspondre à la méthode correcte
            self.assertEqual(calculated_values, user_ads_seen_correct,
                             f"Calcul de user_ads_seen incorrect. Attendu: {user_ads_seen_correct}, Obtenu: {calculated_values}")

        finally:
            if temp_file.exists():
                temp_file.unlink()

if __name__ == '__main__':
    unittest.main()