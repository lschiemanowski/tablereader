"""A recognizer that returns saved responses, for tests and for re-running extraction without a model."""

from collections.abc import Mapping

from document_prep.models import PreparedRegion
from extraction.results import ComponentFailure, OcrResult


class SavedResponses:
    """Returns the saved response for each region ID; a missing ID is a non-retryable failure."""

    def __init__(self, responses: Mapping[str, OcrResult]) -> None:
        self.responses = dict(responses)

    async def recognize(self, region: PreparedRegion) -> OcrResult:
        try:
            return self.responses[region.region.region_id]
        except KeyError:
            raise ComponentFailure("no_saved_response", region.region.region_id, retryable=False) from None
