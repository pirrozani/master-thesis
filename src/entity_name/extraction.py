"""spaCy-based entity name extraction."""

from typing import Iterable

import spacy
from spacy.language import Language
from spacy.tokens import Doc

DEFAULT_SPACY_MODEL = 'en_core_web_trf'
DEFAULT_ENTITY_LABELS: tuple[str, ...] = ('ORG', 'PERSON')


class EntityNameExtractor:
    """Extract entity-name spans from raw text using a pretrained spaCy pipeline.

    The extractor returns the most prominent span whose label matches one of
    the configured entity labels. If no labelled entity is present, it falls
    back to the first noun chunk; if that also fails, it returns an empty
    string.
    """

    def __init__(
        self,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        entity_labels: Iterable[str] = DEFAULT_ENTITY_LABELS,
    ):
        """Initialize extractor.

        Args:
            spacy_model: Name of the spaCy pipeline to load
                (e.g., 'en_core_web_trf', 'en_core_web_lg').
            entity_labels: Iterable of NER labels to consider as entity names.
        """
        self.spacy_model = spacy_model
        self.entity_labels = tuple(entity_labels)
        self.nlp: Language | None = None

    def load_model(self) -> None:
        """Load the spaCy pipeline.

        Raises:
            RuntimeError: If the spaCy model is not installed.
        """
        try:
            self.nlp = spacy.load(self.spacy_model)
        except OSError as e:
            raise RuntimeError(
                f'spaCy model {self.spacy_model!r} is not installed. '
                f'Install it with: '
                f'uv run python -m spacy download {self.spacy_model}'
            ) from e

        print(f'spaCy model {self.spacy_model!r} loaded successfully')

    def _select_span(self, doc: Doc) -> str:
        """Select the best entity span from a parsed Doc.

        Strategy:
            1. Among entities whose label is in ``self.entity_labels``,
               return the one with the longest character span.
            2. If no matching labeled entity is found, return the first
               noun chunk text.
            3. Otherwise, return an empty string.

        Args:
            doc: Parsed spaCy Doc.

        Returns:
            Selected entity span text (raw, not normalized).
        """
        candidates = [ent for ent in doc.ents if ent.label_ in self.entity_labels]

        if candidates:
            best = max(candidates, key=lambda ent: ent.end_char - ent.start_char)
            return best.text.strip()

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if text:
                return text

        return ''

    def extract(self, text: str) -> str:
        """Extract a single entity-name span from text.

        Args:
            text: Raw input text potentially containing an entity name.

        Returns:
            Extracted span string (empty if nothing matches).

        Raises:
            RuntimeError: If the spaCy model has not been loaded.
        """
        if self.nlp is None:
            raise RuntimeError(
                'spaCy model not loaded. Call load_model() before extraction.'
            )

        if not text or not text.strip():
            return ''

        doc = self.nlp(text)
        return self._select_span(doc)

    def extract_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[str]:
        """Extract entity-name spans from multiple texts.

        Uses ``nlp.pipe`` for efficient batched processing.

        Args:
            texts: List of raw input texts.
            batch_size: Batch size for spaCy pipe.

        Returns:
            List of extracted span strings (one per input).

        Raises:
            RuntimeError: If the spaCy model has not been loaded.
        """
        if self.nlp is None:
            raise RuntimeError(
                'spaCy model not loaded. Call load_model() before extraction.'
            )

        if not texts:
            return []

        # Replace empty inputs with a single space so spaCy returns a Doc;
        # we'll still emit '' for them.
        sanitised = [t if (t and t.strip()) else ' ' for t in texts]
        empty_mask = [not (t and t.strip()) for t in texts]

        results: list[str] = []
        for i, doc in enumerate(self.nlp.pipe(sanitised, batch_size=batch_size)):
            if empty_mask[i]:
                results.append('')
            else:
                results.append(self._select_span(doc))

        return results
