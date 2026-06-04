from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, UniqueConstraint, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from offline_llm_eval.db import Base

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "dataset_version", name="uq_datasets_name_version"),)

    dataset_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[JsonObject | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
    )


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    dataset_id: int
    name: str
    dataset_version: str
    metadata_json: JsonObject | None


class DatasetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_name_and_version(
        self,
        name: str,
        dataset_version: str,
    ) -> DatasetRecord | None:
        result = await self._session.execute(
            select(Dataset).where(
                Dataset.name == name,
                Dataset.dataset_version == dataset_version,
            )
        )
        dataset = result.scalar_one_or_none()
        if dataset is None:
            return None

        return _to_record(dataset)

    async def create_dataset(
        self,
        name: str,
        dataset_version: str,
        *,
        metadata_json: JsonObject | None = None,
    ) -> DatasetRecord:
        dataset = Dataset(
            name=name,
            dataset_version=dataset_version,
            metadata_json=_copy_metadata(metadata_json),
        )
        self._session.add(dataset)
        await self._session.flush()

        return _to_record(dataset)

    async def get_or_create_dataset(
        self,
        name: str,
        dataset_version: str,
        *,
        metadata_json: JsonObject | None = None,
    ) -> DatasetRecord:
        existing = await self.get_by_name_and_version(name, dataset_version)
        if existing is not None:
            return existing

        return await self.create_dataset(
            name,
            dataset_version,
            metadata_json=metadata_json,
        )


def _copy_metadata(metadata_json: JsonObject | None) -> JsonObject | None:
    if metadata_json is None:
        return None

    return dict(metadata_json)


def _to_record(dataset: Dataset) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        dataset_version=dataset.dataset_version,
        metadata_json=dataset.metadata_json,
    )
