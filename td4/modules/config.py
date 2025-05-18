import os
import yaml
from pathlib import Path


class Config:
    """Gestionnaire de configuration centralisé pour le projet"""

    _instance = None

    def __new__(cls, config_path=None):
        """Implémentation du pattern Singleton pour assurer une configuration unique"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        """Initialise la configuration à partir d'un fichier yaml ou des valeurs par défaut"""
        if self._initialized:
            return


        self.default_config = {
            "paths": {
                "data_dir": "./data/",
                "models_dir": "./models/",
            },
            "model_params": {
                "user_clusters": 5,
                "page_clusters": 7,
                "seed": 42,
                "max_features": 1000,
                "max_iter": 1000,
            },
            "files": {
                "user_data": "user_data.csv",
                "page_data": "page_data.csv",
                "bid_data_train": "bid_requests_train.csv",
                "click_data_train": "click_data_train.csv",
                "bid_data_test": "bid_requests_test.csv",
                "output_data ": "prediction.csv",
            },
            "training": {
                "test_split": 0.2,
                "threshold": 0.5,
            }
        }

        self.config = self.default_config.copy()
        if config_path:
            self.load_config(config_path)

        self._create_directories()

        self._initialized = True

    def load_config(self, config_path):
        """Charge la configuration depuis un fichier YAML"""
        try:
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)

            self._update_dict(self.config, yaml_config)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration")

    def _update_dict(self, d, u):
        """Met à jour un dictionnaire de manière récursive"""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _create_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        os.makedirs(self.config["paths"]["models_dir"], exist_ok=True)

    def get_data_path(self, file_key=None):
        """Retourne le chemin complet vers un fichier de données"""
        data_dir = Path(self.config["paths"]["data_dir"])

        if file_key:
            return data_dir / self.config["files"][file_key]
        return data_dir

    def get_model_path(self, model_name):
        """Retourne le chemin complet vers un fichier de modèle"""
        return Path(self.config["paths"]["models_dir"]) / f"{model_name}.pkl"

    def get_param(self, section, key):
        """Récupère un paramètre spécifique de la configuration"""
        return self.config[section][key]

    def save_config(self, config_path="config.yaml"):
        """Sauvegarde la configuration actuelle dans un fichier YAML"""
        try:
            with open(config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {e}")

    def find_data_files(self):
        """Recherche les fichiers de données dans différents emplacements possibles"""
        possible_dirs = [
            "./",
            "./data/",
            "./td4/",
            "./td4/data/"
        ]

        for file_key, filename in self.config["files"].items():
            file_found = False
            for directory in possible_dirs:
                path = Path(directory) / filename
                if path.exists():
                    self.config["paths"]["data_dir"] = directory
                    file_found = True
                    print(f"Found {filename} in {directory}")
                    break

            if not file_found:
                print(f"Warning: {filename} not found in any expected location")

        return self.config["paths"]["data_dir"]