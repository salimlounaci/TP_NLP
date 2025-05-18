from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.click_predictor import ClickPredictor
from modules.config import Config
import warnings
import os
import sys
import pandas as pd

warnings.filterwarnings('ignore')


def predict(config_path=None, bid_requests_path=None, output_path="./data/prediction.csv"):
    """
    Makes predictions on test data using the trained models, using the modular approach.
    """
    print("Starting prediction process...")

    # Initialisation de la configuration
    config = Config(config_path)

    # Recherche automatique des fichiers
    data_dir = config.find_data_files()
    print(f"Using data directory: {data_dir}")

    # Création du dossier data s'il n'existe pas
    os.makedirs("./data", exist_ok=True)

    # Si les chemins ne sont pas spécifiés, utilisez la configuration
    if bid_requests_path is None:
        bid_data_test = config.config["files"]["bid_data_test"]
        bid_requests_path = os.path.join(data_dir, bid_data_test)
        print(f"Using test data from config: {bid_requests_path}")

    # S'assurer que le chemin de sortie est dans le dossier data
    output_path = "./data/prediction.csv"
    print(f"Output will be saved to: {output_path}")

    # Initialisation des composants
    data_loader = DataLoader(config_path)

    # Récupérer les paramètres du modèle depuis la config
    page_clusters = config.get_param("model_params", "page_clusters")
    seed = config.get_param("model_params", "seed")
    user_clusters = config.get_param("model_params", "user_clusters")

    page_clusterer = PageClusterer(data_loader, config_path)
    user_clusterer = UserClusterer(data_loader, config_path)
    feature_builder = FeatureBuilder(data_loader, page_clusterer, user_clusterer, config_path)
    click_predictor = ClickPredictor(feature_builder, config_path)

    # Chargement des modèles
    try:
        print("Loading models...")
        page_clusterer.load_models()
        user_clusterer.load_models()
        click_predictor.load_model()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Training new models instead...")

        # Entraînement des modèles si nécessaire
        print("\n== Building page clusters ==")
        page_clusterer.clusterize_pages()

        print("\n== Training page cluster predictor ==")
        page_clusterer.train_page_cluster_predictor()

        print("\n== Building user clusters ==")
        user_clusterer.clusterize_users()

        print("\n== Training click predictor ==")
        click_predictor.train()
        # Sauvegarde des modèles
        page_clusterer.save_models()
        user_clusterer.save_models()
        click_predictor.save_model()

    print("Building features for prediction...")
    test_features = feature_builder.build_test_features(bid_requests_path)

    if test_features is None:
        print("ERROR: Failed to build test features")
        return None

    print("Making predictions...")
    click_predictions, probas = click_predictor.predict_batch(test_features)

    # Création du DataFrame de résultats
    results = pd.DataFrame({
        'user_id': test_features['user_id'],
        'page_id': test_features['page_id'],
        'timestamp': test_features['timestamp'],
        'click': click_predictions
    })

    # Sauvegarde des prédictions dans le dossier data
    results.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

    # Vérification que le fichier a bien été créé
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Verified: File exists with size {file_size} bytes")
    else:
        print(f"WARNING: File was not created at {output_path}")

    return results


if __name__ == "__main__":
    import sys
    import argparse

    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Make predictions with the trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Path to bid requests test data')
    parser.add_argument('--output', type=str, default='./data/prediction.csv', help='Path to save predictions')
    args = parser.parse_args()

    # Exécution de la prédiction
    predict(args.config, args.input, args.output)