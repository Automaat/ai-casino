"""Custom SQLAlchemy type decorators for cross-database compatibility."""

import json
import uuid
from typing import Any

from sqlalchemy import TypeDecorator
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import CHAR, TEXT, TypeEngine


class UUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL UUID where available, otherwise CHAR(36) for SQLite.
    Stores as string, loads as uuid.UUID.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> TypeEngine[Any]:
        """Select UUID for PostgreSQL, CHAR(36) for others."""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Any, dialect: Any) -> str | uuid.UUID | None:
        """Convert UUID to string for binding."""
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value) if dialect.name != "postgresql" else value
        if isinstance(value, str):
            return value
        msg = f"Expected UUID or str, got {type(value)}"
        raise TypeError(msg)

    def process_result_value(self, value: Any, dialect: Any) -> uuid.UUID | None:
        """Convert string to UUID when loading."""
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


class JSONB(TypeDecorator):
    """Platform-independent JSONB type.

    Uses PostgreSQL JSONB where available, otherwise TEXT+JSON for SQLite.
    Stores as JSON string, loads as dict/list.
    """

    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> TypeEngine[Any]:
        """Select JSONB for PostgreSQL, TEXT for others."""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_JSONB())
        return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        """Convert dict/list to JSON string for SQLite."""
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        """Convert JSON string to dict/list for SQLite."""
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.loads(value)


class ARRAY(TypeDecorator):
    """Platform-independent ARRAY type.

    Uses PostgreSQL ARRAY where available, otherwise TEXT+JSON for SQLite.
    Stores as JSON array string, loads as list.
    """

    impl = TEXT
    cache_ok = True

    def __init__(self, item_type: Any = None, *args: Any, **kwargs: Any) -> None:
        """Initialize ARRAY type."""
        super().__init__(*args, **kwargs)
        self.item_type = item_type

    def load_dialect_impl(self, dialect: Any) -> TypeEngine[Any]:
        """Select ARRAY for PostgreSQL, TEXT for others."""
        if dialect.name == "postgresql":
            pg_item_type = self.item_type or TEXT()
            return dialect.type_descriptor(PG_ARRAY(pg_item_type))
        return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        """Convert list to JSON array string for SQLite."""
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        """Convert JSON array string to list for SQLite."""
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.loads(value)
