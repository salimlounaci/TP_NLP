from script import predict
import pandas as pd
def test_end_to_end():
    # Charger les données de test
    data_test = pd.read_csv('bid_requests_test.csv')  # Changez ce chemin selon votre structure de dossier
    # Charger les résultats attendus
    result_expected = pd.read_csv('expected_results.csv')  # Changez ce chemin selon votre structure de dossier


    #train_page_cluster_predictor()  # Entraîner le prédicteur de clusters de pages
    #train_click_predictor()  # Entraîner le prédicteur de clics


    result = predict('bid_requests_test.csv')
    #data_test)  # Cette fonction doit renvoyer les résultats prédits

    pd.testing.assert_frame_equal(result, result_expected)
    print("End-to-End test passed!")
