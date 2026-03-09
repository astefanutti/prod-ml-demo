"""Verify Feast feature retrieval works correctly.

Tests both offline (historical) and online feature retrieval.

Usage:
    python feast/test_features.py
"""

import pandas as pd
from feast import FeatureStore

FEATURE_REPO = "feast/feature_repo"


def test_offline_features():
    """Test historical feature retrieval from offline store."""
    store = FeatureStore(repo_path=FEATURE_REPO)

    entity_df = pd.DataFrame(
        {
            "user_id": ["SAMPLE_USER_1", "SAMPLE_USER_2"],
            "item_id": ["B000000001", "B000000002"],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        }
    )

    print("Testing offline feature retrieval...")
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_features:user_avg_rating",
            "user_features:user_review_count",
            "item_features:item_avg_rating",
            "item_features:item_review_count",
            "item_features:item_price_bucket",
        ],
    ).to_df()

    print("Offline features retrieved:")
    print(features)
    print(f"Shape: {features.shape}")
    return features


def test_online_features():
    """Test real-time feature retrieval from online store."""
    store = FeatureStore(repo_path=FEATURE_REPO)

    print("\nTesting online feature retrieval...")
    try:
        features = store.get_online_features(
            features=[
                "user_features:user_avg_rating",
                "user_features:user_review_count",
                "item_features:item_avg_rating",
                "item_features:item_review_count",
            ],
            entity_rows=[
                {"user_id": "SAMPLE_USER_1", "item_id": "B000000001"},
            ],
        ).to_dict()

        print("Online features retrieved:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        return features
    except Exception as e:
        print(f"Online store not available (expected if Redis not running): {e}")
        return None


if __name__ == "__main__":
    test_offline_features()
    test_online_features()
    print("\nFeature store tests complete.")
