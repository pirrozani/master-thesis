from pydantic import BaseModel, Field

from src.address.models import Address
from src.entity_name.classification.models import EntityType
from src.entity_name.extraction.models import EntityName


class PipelineResult(BaseModel):
    """Combined flat record produced by the three-stage pipeline.

    Field declaration order matches the JSON key order of the emitted
    records (model_dump preserves it).
    """

    address: str = Field(default='', description='Street address')
    city: str = Field(default='', description='City name')
    state: str = Field(default='', description='State or province')
    zip_code: str = Field(default='', description='ZIP or postal code')
    country: str = Field(default='', description='Country name')
    entity_name: str = Field(default='', description='Cleaned entity name')
    entity_type: str = Field(
        default='',
        description="Entity type: 'company', 'person', or '' when invalid",
    )

    @classmethod
    def from_stages(
        cls,
        address: Address,
        entity_name: EntityName,
        entity_type: EntityType,
    ) -> 'PipelineResult':
        """Compose a result from the three stage outputs.

        Args:
            address: Address stage output (street is emitted as 'address')
            entity_name: Name extraction stage output
            entity_type: Classification stage output (invalid labels are
                emitted as an empty string)

        Returns:
            PipelineResult combining the three stage outputs
        """
        return cls(
            address=address.street,
            city=address.city,
            state=address.state,
            zip_code=address.zip_code,
            country=address.country,
            entity_name=entity_name.name,
            entity_type=entity_type.label if entity_type.is_valid() else '',
        )
