import sys

from src.address.inference import AddressExtractor
from src.entity_name.classification.inference import EntityTypeClassifier
from src.entity_name.classification.models import EntityType
from src.entity_name.extraction.inference import EntityNameExtractor
from src.pipeline.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFICATION_TASK,
    resolve_stage_configs,
)
from src.pipeline.models import PipelineResult


class ExtractionPipeline:
    """Three-stage inference pipeline producing combined flat records."""

    def __init__(
        self,
        address_model: str | None = None,
        address_adapter_path: str | None = None,
        extraction_model: str | None = None,
        extraction_adapter_path: str | None = None,
        classification_model: str | None = None,
        classification_adapter_path: str | None = None,
        classification_task: str = DEFAULT_CLASSIFICATION_TASK,
        device: str = 'cuda',
    ):
        """Initialize the pipeline stages without loading any model.

        Args:
            address_model: Model alias for the address stage (None uses
                the stage default)
            address_adapter_path: Adapter dir for the address stage
                (overrides the model config)
            extraction_model: Model alias for the name extraction stage
                (None uses the stage default)
            extraction_adapter_path: Adapter dir for the name extraction
                stage (overrides the model config)
            classification_model: Model alias for the classification
                stage (None uses the stage default)
            classification_adapter_path: Adapter dir for the
                classification stage (overrides the model config)
            classification_task: Classification task variant whose
                adapters to use
            device: Device for inference (cuda/cpu)

        Raises:
            ValueError: If classification_task or any model alias is
                unknown
        """
        configs = resolve_stage_configs(
            address_model=address_model,
            extraction_model=extraction_model,
            classification_model=classification_model,
            classification_task=classification_task,
        )

        self.address_extractor = AddressExtractor(
            model=configs['address'],
            adapter_path=address_adapter_path,
            device=device,
        )
        self.name_extractor = EntityNameExtractor(
            model=configs['extraction'],
            adapter_path=extraction_adapter_path,
            device=device,
        )
        self.type_classifier = EntityTypeClassifier(
            model=configs['classification'],
            adapter_path=classification_adapter_path,
            device=device,
        )
        self.classification_task = classification_task
        self.device = device

    def load_models(self) -> None:
        """Load the three stage models sequentially.

        Load order is fixed (address, extraction, classification) so a
        failure is attributable to a single stage.

        Raises:
            RuntimeError: If any stage model fails to load
        """
        self.address_extractor.load_model()
        self.name_extractor.load_model()
        self.type_classifier.load_model()

    def run(self, text: str) -> PipelineResult:
        """Run the full pipeline on a single raw combined line.

        Args:
            text: Raw combined line (entity name + address)

        Returns:
            PipelineResult with address fields, cleaned entity name, and
            entity type

        Raises:
            RuntimeError: If any stage model is not loaded
        """
        address = self.address_extractor.extract(text)
        entity_name = self.name_extractor.extract(text)
        entity_type = self.type_classifier.classify(entity_name.name)

        return PipelineResult.from_stages(address, entity_name, entity_type)

    def run_batch(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[PipelineResult]:
        """Run the full pipeline over multiple raw combined lines.

        Processes the inputs in chunks of batch_size, preserving input
        order. Empty extracted names skip the classification stage and
        yield an empty entity type.

        Args:
            texts: Raw combined lines to process
            batch_size: Number of inputs per chunk

        Returns:
            List of PipelineResult objects in input order

        Raises:
            ValueError: If batch_size is smaller than 1
            RuntimeError: If any stage model is not loaded
        """
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        if not texts:
            return []

        results: list[PipelineResult] = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            chunk = texts[start_idx:end_idx]

            print(
                f'Pipeline batch {batch_idx + 1}/{num_batches}',
                file=sys.stderr,
            )

            addresses = self.address_extractor.extract_batch(chunk)
            names = self.name_extractor.extract_batch(chunk)

            # classify_batch has no per-item empty guard, so empty names
            # keep the invalid sentinel and never reach the model
            entity_types = [EntityType() for _ in chunk]
            to_classify = [
                (index, name.name)
                for index, name in enumerate(names)
                if name.name.strip()
            ]
            if to_classify:
                predicted = self.type_classifier.classify_batch(
                    [name for _, name in to_classify]
                )
                for (index, _), entity_type in zip(to_classify, predicted):
                    entity_types[index] = entity_type

            results.extend(
                PipelineResult.from_stages(address, name, entity_type)
                for address, name, entity_type in zip(addresses, names, entity_types)
            )

        return results
