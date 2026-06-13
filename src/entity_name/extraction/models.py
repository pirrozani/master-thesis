"""Pydantic models for entity name extraction."""

from pydantic import BaseModel, Field


class EntityName(BaseModel):
    """Entity name model for name extraction and cleaning."""

    name: str = Field(default='', description='Cleaned entity name')

    def to_output(self) -> str:
        """Serialize to plain string output.

        Returns:
            Cleaned entity name string
        """
        return self.name

    @classmethod
    def from_output(cls, output: str) -> 'EntityName':
        """Parse from model output string.

        Args:
            output: Raw model output string

        Returns:
            EntityName instance with parsed value
        """
        if not output or not output.strip():
            return cls()

        return cls(name=output.strip())
