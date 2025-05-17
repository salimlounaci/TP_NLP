import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def get_data(self):
        """Load all datasets from data directory"""
        if all(k in self._cache for k in ["user_data", "page_data", "bid_data", "click_data"]):
            return (
                self._cache["user_data"],
                self._cache["page_data"],
                self._cache["bid_data"],
                self._cache["click_data"]
            )

        # Load data
        tmp_user_data = pd.read_csv(self.data_dir / "user_data.csv")
        tmp_page_data = pd.read_csv(self.data_dir / "page_data.csv")
        tmp_bid_data = pd.read_csv(self.data_dir / "bid_requests_train.csv")
        tmp_click_data = pd.read_csv(self.data_dir / "click_data_train.csv")

        # Cache data
        self._cache["user_data"] = tmp_user_data
        self._cache["page_data"] = tmp_page_data
        self._cache["bid_data"] = tmp_bid_data
        self._cache["click_data"] = tmp_click_data

        return tmp_user_data, tmp_page_data, tmp_bid_data, tmp_click_data

    def get_test_data(self, bid_requests_path):
        """Load test data"""
        return pd.read_csv(bid_requests_path)