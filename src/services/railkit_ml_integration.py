import pandas as pd

from src.data.railkit_operational_context import RailKitOperationalContext
from src.features.operational_context_enricher import (
    OperationalContextEnricher,
)
from src.features.railkit_operational_features import (
    RailKitOperationalFeatures,
)


class RailKitMLIntegration:
    """
    End-to-end integration between live RailKit operational
    context and the existing ML feature pipeline.
    """

    def __init__(self):
        self.context_service = RailKitOperationalContext()
        self.enricher = OperationalContextEnricher()
        self.operational_features = RailKitOperationalFeatures()

    def fetch_context(self, train_number, date):
        return self.context_service.fetch_train_context(
            train_number,
            date,
        )

    def enrich(self, df, train_number, date):
        context = self.fetch_context(
            train_number,
            date,
        )

        enriched = self.enricher.enrich(
            df,
            context,
        )

        enriched = self.operational_features.transform(
            enriched
        )

        return enriched

    def enrich_with_context(self, df, context):
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")

        enriched = self.enricher.enrich(
            df,
            context,
        )

        return self.operational_features.transform(
            enriched
        )
