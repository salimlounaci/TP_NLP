import os
import time
import warnings

warnings.filterwarnings('ignore')

from modules.data_loader import DataLoader
from modules.page_clustering import PageClusterer
from modules.user_clustering import UserClusterer
from modules.feature_builder import FeatureBuilder
from modules.click_predictor import ClickPredictor


def main():
    print("Starting ad prediction system...")

    # Initialize components
    data_loader = DataLoader(data_dir="./data")  # Adapt path as needed
    page_clusterer = PageClusterer(data_loader, n_clusters=7, seed=42)
    user_clusterer = UserClusterer(data_loader, n_clusters=5, seed=42)
    feature_builder = FeatureBuilder(data_loader, page_clusterer, user_clusterer)
    click_predictor = ClickPredictor(feature_builder, seed=42)

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
    main()