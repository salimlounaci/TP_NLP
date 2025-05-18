import warnings

warnings.filterwarnings('ignore')
import argparse

from modules.config import Config
from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.click_predictor import ClickPredictor


def predict(config_path=None, bid_requests_path=None, output_path=None):
    print("Starting prediction process...")

    # Initialisation de la configuration
    config = Config(config_path)

    # Si les chemins ne sont pas spécifiés, utilisez la configuration
    if bid_requests_path is None:
        bid_requests_path = config.get_data_path("bid_data_test")

    if output_path is None:
        output_path = config.get_data_path() / "predictions.csv"

    # Initialisation des composants avec la configuration
    data_loader = DataLoader(config_path)
    page_clusterer = PageClusterer(data_loader, config_path)
    user_clusterer = UserClusterer(data_loader, config_path)
    feature_builder = FeatureBuilder(data_loader, page_clusterer, user_clusterer, config_path)
    click_predictor = ClickPredictor(feature_builder, config_path)

    # Chargement des modèles
    print("Loading models...")
    page_clusterer.load_models()
    user_clusterer.load_models()
    click_predictor.load_model()

    # Création des features pour la prédiction
    print("Building features for prediction...")
    test_features = feature_builder.build_test_features(bid_requests_path)

    # Prédiction
    print("Making predictions...")
    click_predictions, _ = click_predictor.predict_batch(test_features)

    # Création du DataFrame de résultats
    import pandas as pd
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
    # Ajout d'arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Make predictions with the trained model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Path to bid requests test data')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    args = parser.parse_args()

    predict(args.config, args.input, args.output)