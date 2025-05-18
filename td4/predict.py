from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.click_predictor import ClickPredictor
from modules.config import Config
import os
import pandas as pd


def predict(config_path=None, bid_requests_path=None, output_path=None):
    """
    Makes predictions on test data using the trained models.
    """
    print("Starting prediction process...")

    # Initialisation de la configuration
    config = Config(config_path)

    # Recherche automatique des fichiers
    data_dir = config.find_data_files()
    print(f"Using data directory: {data_dir}")

    # Création du dossier data
    os.makedirs("./data", exist_ok=True)

    # Si les chemins ne sont pas spécifiés, utilisez la configuration
    if bid_requests_path is None:
        bid_data_test = config.config["files"]["bid_data_test"]
        bid_requests_path = os.path.join(data_dir, bid_data_test)
        print(f"Using test data from config: {bid_requests_path}")

    # Si le chemin de sortie n'est pas spécifié, utilisez la configuration
    if output_path is None:
        output_file = config.config["files"]["output_data"]
        output_path = os.path.join(data_dir, output_file)
    print(f"Output will be saved to: {output_path}")

    # Initialisation des composants
    data_loader = DataLoader(config_path)
    page_clusterer = PageClusterer(data_loader, config_path)
    user_clusterer = UserClusterer(data_loader, config_path)
    feature_builder = FeatureBuilder(data_loader, page_clusterer, user_clusterer, config_path)
    click_predictor = ClickPredictor(feature_builder, config_path)

    # Chargement ou entraînement des modèles
    print("Loading or training models...")
    page_clusterer.clusterize_pages()
    page_clusterer.train_page_cluster_predictor()
    user_clusterer.clusterize_users()
    click_predictor.train()

    print("Building features for prediction...")
    test_features = feature_builder.build_test_features(bid_requests_path)

    print("Making predictions...")
    click_predictions, _ = click_predictor.predict_batch(test_features)

    # Création du DataFrame de résultats
    results = pd.DataFrame({
        'user_id': test_features['user_id'],
        'page_id': test_features['page_id'],
        'timestamp': test_features['timestamp'],
        'click': click_predictions
    })

    # Sauvegarde des prédictions
    results.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Make predictions with the trained model')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Path to bid requests test data')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    args = parser.parse_args()


    predict(args.config, args.input, args.output)