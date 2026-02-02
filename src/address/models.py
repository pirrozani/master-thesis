"""Pydantic models for address extraction."""

from pydantic import BaseModel, Field


class Address(BaseModel):
    """Address model with pipe-delimited serialization support."""

    street: str = Field(default='', description='Street address')
    city: str = Field(default='', description='City name')
    state: str = Field(default='', description='State or province')
    zip_code: str = Field(default='', description='ZIP or postal code')
    country: str = Field(default='', description='Country name')

    def to_pipe(self) -> str:
        """Serialize to pipe-delimited format.

        Returns:
            Pipe-delimited string: street|city|state|zip|country
        """
        return f'{self.street}|{self.city}|{self.state}|{self.zip_code}|{self.country}'

    @classmethod
    def from_pipe(cls, pipe_string: str) -> 'Address':
        """Parse from a pipe-delimited format string.

        Args:
            pipe_string: Pipe-delimited string in format:
                        street|city|state|zip|country

        Returns:
            Address instance with parsed values

        Raises:
            ValueError: If pipe_string doesn't have an expected format
        """
        if not pipe_string or not pipe_string.strip():
            return cls()

        # Split by pipe delimiter
        parts = pipe_string.strip().split('|')

        # Pad with empty strings if fewer than 5 parts
        while len(parts) < 5:
            parts.append('')

        return cls(
            street=parts[0].strip(),
            city=parts[1].strip(),
            state=parts[2].strip(),
            zip_code=parts[3].strip(),
            country=parts[4].strip(),
        )
