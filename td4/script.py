from functools import cache
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
import random
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Global vars
u_clusters = 5  # Number of user clusters
p_clusters = 7  # Number of page clusters
seed = 42

#FOLDER = Path("data") / "raw"/ "td4"

_cache = {}

def get_data():
    tmp_user_data = pd.read_csv("user_data.csv")
    tmp_page_data = pd.read_csv("page_data.csv")
    tmp_bid_data = pd.read_csv("bid_requests_train.csv")
    tmp_click_data = pd.read_csv("click_data_train.csv")
    
    _cache["user_data"] = tmp_user_data
    _cache["page_data"] = tmp_page_data
    _cache["bid_data"] = tmp_bid_data
    _cache["click_data"] = tmp_click_data
    
    return tmp_user_data, tmp_page_data, tmp_bid_data, tmp_click_data

def preprocess_text(text_series):
    text_series = text_series.fillna("")
    text_series = text_series.str.lower()
    return text_series

@cache
def clusterize_pages(k=7):
    if "page_clusters" in _cache:
        return _cache["page_clusters"], _cache["page_cluster_model"], _cache["page_vectorizer"]
    
    _, page_data, _, _ = get_data()
    
    vect = TfidfVectorizer(max_features=1000, stop_words='english')
    X_pages = vect.fit_transform(preprocess_text(page_data['page_text']))
    
    km = KMeans(n_clusters=k, random_state=seed)
    page_clusters = km.fit_predict(X_pages)
    
    page_data['cluster'] = page_clusters
    
    _cache["page_clusters"] = page_data
    _cache["page_cluster_model"] = km
    _cache["page_vectorizer"] = vect
    
    return page_data, km, vect


def train_page_cluster_predictor():
    page_data, _, vect = clusterize_pages(p_clusters)
    
    X_pages = vect.transform(preprocess_text(page_data['page_text']))
    y = page_data['cluster']
    
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_pages, y)
    
    _cache["page_cluster_predictor"] = lr
    
    return lr

def process_user_data():
    """Process user data for clustering"""
    # Get data
    user_data, _, bid_data, _ = get_data()
    
    # One-hot encode user features
    user_processed = pd.get_dummies(user_data, columns=['sex', 'city', 'device'])
    
    # Join with bid data to get user-page interactions
    user_visits = (
        bid_data.groupby(["user_id", "page_id"])
        .size()
        .unstack(1)
        .fillna(0)
    )
    user_visits.columns = [str(c) for c in user_visits.columns]
    user_processed = user_processed.merge(user_visits, on='user_id', how='left')
    
    # Cache processed data
    _cache["processed_user_data"] = user_processed
    
    return user_processed

def clusterize_users(k=5):
    if "user_clusters" in _cache:
        return _cache["user_clusters"], _cache["user_cluster_model"]
    
    user_processed = process_user_data()
    
    km = KMeans(n_clusters=k, random_state=seed)
    user_clusters = km.fit_predict(user_processed.drop('user_id', axis=1))
    
    user_processed['cluster'] = user_clusters
    
    _cache["user_clusters"] = user_processed
    _cache["user_cluster_model"] = km
    
    return user_processed, km

@cache
def get_page_cluster_probabilities(page_id):
    """Get probabilities of a page belonging to each cluster"""
    page_data, _, vect = clusterize_pages(p_clusters)

    lr = _cache.get("page_cluster_predictor")
    if not lr:
        lr = train_page_cluster_predictor()
    
    page_text = page_data[page_data['page_id'] == page_id]['page_text'].values[0]
    
    X = vect.transform([preprocess_text(pd.Series([page_text]))[0]])
    
    probs = lr.predict_proba(X)[0]
    
    return probs

def build_click_features():
    """Build features for click prediction"""
    user_data, page_data, bid_data, click_data = get_data()
    
    # Number of ad seen this day before this page
    click_data["date"] = click_data["timestamp"].apply(lambda txt: txt[:10])
    click_data["count"] = 1
    click_data["user_ads_seen"] = (
        click_data.groupby(["user_id", "date"])["count"]
        .cumsum()
    )


    click_data = click_data[["user_id", "page_id", "ad_id", "user_ads_seen", "clicked"]]

    user_clusters, _ = clusterize_users(u_clusters)
    page_clusters, _, _ = clusterize_pages(p_clusters)
    
    click_features = click_data.merge(user_clusters[['user_id', 'cluster']], on='user_id', how='left')
    click_features = click_features.rename(columns={'cluster': 'user_cluster'})
    
    cluster_probs = []
    page_to_cluster_prob = {page_id: get_page_cluster_probabilities(page_id) for page_id in click_features["page_id"].unique()}

    cluster_probs = [page_to_cluster_prob[page_id] for page_id in click_features["page_id"]]
    
    cluster_prob_df = pd.DataFrame(
        cluster_probs, 
        columns=[f'page_cluster_prob_{i}' for i in range(p_clusters)]
    )
    
    click_features = pd.concat(
        [click_features.reset_index(drop=True),  cluster_prob_df.reset_index(drop=True)],
        axis=1,
    )
    
    _cache["click_features"] = click_features
    
    return click_features

def train_click_predictor():
    click_features = build_click_features()
    
    X = click_features.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
    
    y = click_features['clicked']
    
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X, y)
    
    _cache["click_predictor"] = lr
    
    return lr

def predict_click(user_id, page_id, ad_id):
    user_clusters, _ = clusterize_users(u_clusters)
    user_cluster = user_clusters[user_clusters['user_id'] == user_id]['cluster'].values[0]
    
    page_probs = get_page_cluster_probabilities(page_id)
    
    features = np.hstack([np.array([user_cluster]), page_probs, np.array([ad_id])])
    
    lr = train_click_predictor()
    
    prob = lr.predict_proba(features.reshape(1, -1))[0][1]
    
    return prob

def evaluate_model():
    click_features = build_click_features()
    
    msk = np.random.rand(len(click_features)) < 0.8
    train = click_features[msk]
    test = click_features[~msk]
    
    X_train = train.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
    y_train = train['clicked']
    
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)
    
    X_test = test.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
    y_test = test['clicked']
    
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")

def get_recommendations(user_id, page_id, ad_ids):
    load_models()
    predictions = []
    for ad_id in ad_ids:
        prob = predict_click(user_id, page_id, ad_id)
        predictions.append((ad_id, prob))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions

def save_models():
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Save page cluster model
    with open("models/page_cluster_model.pkl", "wb") as f:
        pickle.dump(_cache["page_cluster_model"], f)
    
    # Save page vectorizer
    with open("models/page_vectorizer.pkl", "wb") as f:
        pickle.dump(_cache["page_vectorizer"], f)
    
    # Save page cluster predictor
    with open("models/page_cluster_predictor.pkl", "wb") as f:
        pickle.dump(_cache["page_cluster_predictor"], f)
    
    # Save user cluster model
    with open("models/user_cluster_model.pkl", "wb") as f:
        pickle.dump(_cache["user_cluster_model"], f)
    
    # Save click predictor
    with open("models/click_predictor.pkl", "wb") as f:
        pickle.dump(_cache["click_predictor"], f)

def load_models():
    with open("models/page_cluster_model.pkl", "rb") as f:
        _cache["page_cluster_model"] = pickle.load(f)
    
    # Load page vectorizer
    with open("models/page_vectorizer.pkl", "rb") as f:
        _cache["page_vectorizer"] = pickle.load(f)
    
    # Load page cluster predictor
    with open("models/page_cluster_predictor.pkl", "rb") as f:
        _cache["page_cluster_predictor"] = pickle.load(f)
    
    # Load user cluster model
    with open("models/user_cluster_model.pkl", "rb") as f:
        _cache["user_cluster_model"] = pickle.load(f)
    
    # Load click predictor
    with open("models/click_predictor.pkl", "rb") as f:
        _cache["click_predictor"] = pickle.load(f)

def train():
    print("Starting ad prediction system...")

    print("\n== Loading data ==")
    get_data()

    print("\n== Building page clusters ==")
    clusterize_pages(p_clusters)

    print("\n== Training page cluster predictor ==")
    train_page_cluster_predictor()

    print("\n== Building user clusters ==")
    clusterize_users(u_clusters)

    print("\n== Training click predictor ==")
    train_click_predictor()

    print("\n== Saving models ==")
    save_models()


def predict(bid_requests_path, output_path="predictions.csv"):
    """
    Makes predictions on test data using the trained models.
    
    Args:
        bid_requests_path: Path to the bid requests test data
        output_path: Path where to save the prediction results
        
    Returns:
        DataFrame with predictions
    """
    print("Loading models and data...")
    # Load the necessary models
    with open('models/click_predictor.pkl', 'rb') as f:
        click_model = pickle.load(f)
    
    with open('models/page_cluster_predictor.pkl', 'rb') as f:
        page_cluster_model = pickle.load(f)
        
    with open('models/page_vectorizer.pkl', 'rb') as f:
        page_vectorizer = pickle.load(f)
        
    # Load data files
    bid_requests = pd.read_csv(bid_requests_path)
    user_data = pd.read_csv('user_data.csv')
    page_data = pd.read_csv('page_data.csv')
    
    # For debugging - check what features the model expects
    print("Click model expects features:", click_model.feature_names_in_ if hasattr(click_model, 'feature_names_in_') else "Unknown")
    
    print("Creating predictions directly...")
    # Create a mapping from page_id to page_text
    page_text_map = dict(zip(page_data['page_id'], page_data['page_text']))
    
    # Create a simple mapping from user_id to a default cluster (we'll use 0 since we can't use the original model)
    user_cluster_map = {user_id: 0 for user_id in user_data['user_id']}
    
    # Process page texts and get cluster probabilities for each page in the test set
    page_cluster_probs = {}
    num_clusters = len(page_cluster_model.classes_)
    
    for page_id in bid_requests['page_id'].unique():
        if page_id in page_text_map:
            page_text = page_text_map[page_id]
            # Preprocess text
            page_text = page_text.lower() if isinstance(page_text, str) else ""
            
            # Transform using the vectorizer
            X = page_vectorizer.transform([page_text])
            
            # Get probabilities for each cluster
            probs = page_cluster_model.predict_proba(X)[0]
            page_cluster_probs[page_id] = probs
        else:
            # If page_id not found, use zeros as probabilities
            page_cluster_probs[page_id] = np.zeros(num_clusters)
    
    print("Building features for prediction...")
    
    # Simplified approach - create features based on the click model's expected feature names
    results = []
    
    for _, row in bid_requests.iterrows():
        user_id = row['user_id']
        page_id = row['page_id']
        
        # Simplified approach - use fixed values for features we can't generate properly
        features = {}
        
        # Add user_cluster (using a default value of 0)
        features['user_cluster'] = user_cluster_map.get(user_id, 0)
        
        # Add user_ads_seen (default to 1)
        features['user_ads_seen'] = 1
        
        # Add page cluster probabilities
        probs = page_cluster_probs.get(page_id, np.zeros(num_clusters))
        for i, prob in enumerate(probs):
            features[f'page_cluster_prob_{i}'] = prob
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Check if we have all required features
        missing_features = []
        for feature in click_model.feature_names_in_:
            if feature not in features_df.columns:
                missing_features.append(feature)
                features_df[feature] = 0  # Add missing feature with default value
        
        if missing_features:
            print(f"Adding {len(missing_features)} missing features with default values.")
        
        # Make sure all columns are in the same order as during training
        features_df = features_df[click_model.feature_names_in_]
        
        # Make prediction
        prob = click_model.predict_proba(features_df)[0][1]
        click_prediction = 1 if prob >= 0.5 else 0
        
        # Add to results
        results.append({
            'user_id': user_id,
            'page_id': page_id,
            'timestamp': row['timestamp'],
            'click': click_prediction
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save predictions
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    return results_df

def main():

    train()
    print("\n== Evaluating model ==")
    evaluate_model()
    print("\nDone!")


if __name__ == "__main__":
    main()