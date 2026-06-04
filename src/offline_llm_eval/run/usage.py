from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from offline_llm_eval.dataset.repository import JsonObject, JsonValue


class UsageSource(StrEnum):
    UNAVAILABLE = "unavailable"
    PROVIDER_REPORTED = "provider_reported"


@dataclass(frozen=True, slots=True)
class UsageSummary:
    usage_source: UsageSource
    usage_json: JsonObject | None


def resolve_usage_summary(usage_json: Mapping[str, JsonValue] | None) -> UsageSummary:
    if usage_json is None:
        return UsageSummary(
            usage_source=UsageSource.UNAVAILABLE,
            usage_json=None,
        )

    return UsageSummary(
        usage_source=UsageSource.PROVIDER_REPORTED,
        usage_json=dict(usage_json),
    )
