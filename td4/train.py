import warnings

warnings.filterwarnings('ignore')
import argparse

from modules.config import Config
from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.click_predictor import ClickPredictor


def main(config_path=None):
    print("Starting ad prediction system...")

    # Initialisation de la configuration
    config = Config(config_path)

    # Sauvegarde de la configuration (optionnel)
    config.save_config()

    print("iaasn",config_path)

    # Initialisation des composants avec la configuration
    data_loader = DataLoader(config_path)
    page_clusterer = PageClusterer(data_loader, config_path)
    user_clusterer = UserClusterer(data_loader, config_path)
    feature_builder = FeatureBuilder(data_loader, page_clusterer, user_clusterer, config_path)
    click_predictor = ClickPredictor(feature_builder, config_path)

    print("\n== Loading data ==")
    data_loader.get_data()

    print("\n== Building page clusters ==")
    page_clusterer.clusterize_pages()

    print("\n== Training page cluster predictor ==")
    page_clusterer.train_page_cluster_predictor()

    print("\n== Building user clusters ==")
    user_clusterer.clusterize_users()

    print("\n== Training click predictor ==")
    click_predictor.train()

    print("\n== Evaluating model ==")
    accuracy = click_predictor.evaluate()
    print(f"Model accuracy: {accuracy:.4f}")

    print("\n== Saving models ==")
    page_clusterer.save_models()
    user_clusterer.save_models()
    click_predictor.save_model()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the ad prediction system')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    print(f"Config path from args: {args.config}")  # Ajout pour débogage
    main(args.config)