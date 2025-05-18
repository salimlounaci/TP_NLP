import pandas as pd
from pathlib import Path
from modules.config import Config


class DataLoader:
    def __init__(self, config_path=None):
        self.config = Config(config_path)
        # Trouve automatiquement les fichiers de données
        self.config.find_data_files()
        self._cache = {}

    def get_data(self):
        """Charge toutes les données nécessaires"""
        if all(k in self._cache for k in ["user_data", "page_data", "bid_data", "click_data"]):
            return (
                self._cache["user_data"],
                self._cache["page_data"],
                self._cache["bid_data"],
                self._cache["click_data"]
            )

        # Récupération des chemins depuis la configuration
        user_data_path = self.config.get_data_path("user_data")
        print(user_data_path)
        page_data_path = self.config.get_data_path("page_data")
        bid_data_path = self.config.get_data_path("bid_data_train")
        click_data_path = self.config.get_data_path("click_data_train")


        # Chargement des données
        try:
            tmp_user_data = pd.read_csv(user_data_path)
            tmp_page_data = pd.read_csv(page_data_path)
            tmp_bid_data = pd.read_csv(bid_data_path)
            tmp_click_data = pd.read_csv(click_data_path)
            print(tmp_click_data)
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Trying to find data files automatically...")

            # Recherche automatique
            data_dir = self.config.find_data_files()

            # Nouvelle tentative de chargement
            user_data_path = Path(data_dir) / self.config.config["files"]["user_data"]
            page_data_path = Path(data_dir) / self.config.config["files"]["page_data"]
            bid_data_path = Path(data_dir) / self.config.config["files"]["bid_data_train"]
            click_data_path = Path(data_dir) / self.config.config["files"]["click_data_train"]

            tmp_user_data = pd.read_csv(user_data_path)
            tmp_page_data = pd.read_csv(page_data_path)
            tmp_bid_data = pd.read_csv(bid_data_path)
            tmp_click_data = pd.read_csv(click_data_path)

        # Mise en cache des données
        self._cache["user_data"] = tmp_user_data
        self._cache["page_data"] = tmp_page_data
        self._cache["bid_data"] = tmp_bid_data
        self._cache["click_data"] = tmp_click_data

        return tmp_user_data, tmp_page_data, tmp_bid_data, tmp_click_data

    def get_test_data(self, bid_requests_path=None):
        """Charge les données de test"""
        if bid_requests_path is None:
            bid_requests_path = self.config.get_data_path("bid_data_test")

        return pd.read_csv(bid_requests_path)