"""Coordinator memory for persistent learning observations."""

import asyncio
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field


class ObservationRecord(BaseModel):
    """Single observation record."""

    timestamp: datetime
    observation: str
    category: str = Field(description="Observation category")


class CoordinatorMemory:
    """Append-only memory for coordinator learning observations."""

    def __init__(self, memory_file: Path | None = None) -> None:
        """Initialize coordinator memory.

        Args:
            memory_file: Path to JSONL memory file (default: ~/.ai-casino/coordinator-memory.jsonl)
        """
        self._memory_file = memory_file or Path("~/.ai-casino/coordinator-memory.jsonl").expanduser()
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self._memory_file.exists():
            self._memory_file.touch()
            logger.info(f"Created coordinator memory at {self._memory_file}")

    async def save(self, observation: str, category: str = "general") -> None:
        """Save observation to memory file.

        Args:
            observation: Observation text
            category: Category (market/pattern/error/success/general)
        """
        record = ObservationRecord(
            timestamp=datetime.now(UTC),
            observation=observation,
            category=category,
        )

        # Append to JSONL file (offload to thread)
        await asyncio.to_thread(self._append_record, record)
        logger.debug(f"Saved observation: {category}")

    def _append_record(self, record: ObservationRecord) -> None:
        """Append record to JSONL file.

        Args:
            record: Observation record to append
        """
        with self._memory_file.open("a") as f:
            f.write(record.model_dump_json() + "\n")

    async def retrieve_recent(
        self,
        limit: int = 50,
        category: str | None = None,
    ) -> list[ObservationRecord]:
        """Retrieve recent observations from memory.

        Args:
            limit: Maximum number of records to retrieve
            category: Optional category filter

        Returns:
            List of observation records (most recent first)
        """
        return await asyncio.to_thread(self._read_records, limit, category)

    def _read_records(self, limit: int, category: str | None) -> list[ObservationRecord]:
        """Read records from JSONL file.

        Args:
            limit: Maximum number of records
            category: Optional category filter

        Returns:
            List of observation records (most recent first)
        """
        if not self._memory_file.exists():
            return []

        try:
            # Use deque to keep only last N matching records, bounded memory
            records: deque[ObservationRecord] = deque(maxlen=limit)

            with self._memory_file.open() as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        record = ObservationRecord.model_validate_json(line)
                        if category is None or record.category == category:
                            records.append(record)
                    except Exception as e:
                        logger.warning(f"Failed to parse observation record: {e}")
                        continue

            # Return in reverse order (most recent first)
            return list(reversed(records))

        except Exception as e:
            logger.error(f"Failed to read observations: {e}")
            return []

    def __repr__(self) -> str:
        """String representation."""
        return f"CoordinatorMemory(file={self._memory_file})"
