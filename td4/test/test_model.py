import sys
import os
# Ajouter le répertoire parent au chemin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import predict
import pandas as pd
from modules.config import Config


def test_end_to_end():
    # Charger les données de test
    data_test = pd.read_csv('./data/bid_requests_test.csv')

    # Charger les résultats attendus
    result_expected = pd.read_csv('./data/expected_results.csv')

    config_path = './config.yaml'
    input_path = './data/bid_requests_test.csv'
    output_path = './data/prediction.csv'

    result = predict(config_path, input_path, output_path)

    pd.testing.assert_frame_equal(result, result_expected)
    print("End-to-End test passed!")