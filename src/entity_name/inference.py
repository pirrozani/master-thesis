"""End-to-end inference pipeline for entity name extraction."""

from src.entity_name.extraction import (
    DEFAULT_ENTITY_LABELS,
    DEFAULT_SPACY_MODEL,
    EntityNameExtractor,
)
from src.entity_name.models import EntityName
from src.entity_name.normalization import EntityNameNormalizer


class EntityNameExtractionPipeline:
    """Compose ``EntityNameExtractor`` and ``EntityNameNormalizer`` end-to-end.

    The pipeline runs spaCy NER over the input to obtain a candidate span,
    then runs the rule normalizer over that span to produce the cleaned
    entity name. Returns ``EntityName`` objects so the public surface
    matches the address pipeline's contract.
    """

    def __init__(
        self,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        entity_labels: tuple[str, ...] = DEFAULT_ENTITY_LABELS,
    ):
        """Initialize pipeline.

        Args:
            spacy_model: Name of the spaCy pipeline to load.
            entity_labels: NER labels to consider as entity names.
        """
        self.extractor = EntityNameExtractor(
            spacy_model=spacy_model,
            entity_labels=entity_labels,
        )
        self.normalizer = EntityNameNormalizer()

    @property
    def model(self):
        """Expose the spaCy model handle (used by the evaluator readiness check)."""
        return self.extractor.nlp

    def load_model(self) -> None:
        """Load the spaCy pipeline."""
        self.extractor.load_model()

    def predict(self, text: str) -> EntityName:
        """Extract and normalize a single entity name.

        Args:
            text: Raw input text.

        Returns:
            ``EntityName`` with the cleaned name (empty if nothing matched).
        """
        raw_span = self.extractor.extract(text)
        cleaned = self.normalizer.normalize(raw_span)
        return EntityName(name=cleaned)

    def predict_batch(self, texts: list[str]) -> list[EntityName]:
        """Extract and normalize a batch of entity names.

        Args:
            texts: List of raw input texts.

        Returns:
            List of ``EntityName`` objects (one per input).
        """
        raw_spans = self.extractor.extract_batch(texts)
        cleaned = self.normalizer.normalize_batch(raw_spans)
        return [EntityName(name=c) for c in cleaned]

    # Compatibility shims so this pipeline can stand in for the LLM-style
    # extractor used by the evaluator (``extract`` / ``extract_batch``).

    def extract(self, text: str) -> EntityName:
        """Alias for ``predict`` (compatibility with evaluator)."""
        return self.predict(text)

    def extract_batch(self, texts: list[str]) -> list[EntityName]:
        """Alias for ``predict_batch`` (compatibility with evaluator)."""
        return self.predict_batch(texts)
