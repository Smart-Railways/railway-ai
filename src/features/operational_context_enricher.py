import pandas as pd

from src.features.railkit_feature_bridge import RailKitFeatureBridge


class OperationalContextEnricher:
    """
    Enriches an operational-impact dataset with normalized
    RailKit operational features.

    The enrichment is deliberately additive: existing ML
    features are preserved unchanged.
    """

    def __init__(self):
        self.railkit_bridge = RailKitFeatureBridge()

    def enrich(self, df, railkit_context):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        railkit_features = self.railkit_bridge.transform(
            railkit_context
        )

        enriched = df.copy()

        for column in railkit_features.columns:
            enriched[column] = railkit_features.iloc[0][column]

        return enriched
