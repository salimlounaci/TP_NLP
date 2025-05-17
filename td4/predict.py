from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.click_predictor import ClickPredictor
import warnings

warnings.filterwarnings('ignore')


def predict(bid_requests_path, output_path="./data/predictions.csv"):
    """
    Makes predictions on test data using the trained models, using the modular approach.
    """
    print("Loading models and data...")

    # Initialisation des composants
    data_loader = DataLoader()
    page_clusterer = PageClusterer(data_loader, n_clusters=7, seed=42)
    user_clusterer = UserClusterer(data_loader, n_clusters=5, seed=42)
    feature_builder = FeatureBuilder(data_loader, page_clusterer, user_clusterer)
    click_predictor = ClickPredictor(feature_builder, seed=42)

    # Chargement des modèles
    page_clusterer.load_models()
    user_clusterer.load_models()
    click_predictor.load_model()

    print("Building features for prediction...")
    test_features = feature_builder.build_test_features(bid_requests_path)

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
    import sys

    # Vous pouvez choisir quelle méthode utiliser ici
    use_modular = False  # Mettre à True pour utiliser l'approche modulaire

    if len(sys.argv) > 1:
        bid_requests_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    else:
        # Valeurs par défaut si aucun argument n'est fourni
        bid_requests_path = './data/bid_requests_test.csv'
        output_path = './data/predictions.csv'

    if use_modular:
        predict(bid_requests_path, output_path)
    else:
        from predict import predict

        predict(bid_requests_path, output_path)