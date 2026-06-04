from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
)

MIN_RATE_THRESHOLD = 0.0
MAX_RATE_THRESHOLD = 1.0
MIN_COUNT_THRESHOLD = 0

type RateThreshold = Annotated[
    StrictFloat,
    Field(ge=MIN_RATE_THRESHOLD, le=MAX_RATE_THRESHOLD),
]
type CountThreshold = Annotated[
    StrictInt,
    Field(ge=MIN_COUNT_THRESHOLD),
]


class GateConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass_rate_min: RateThreshold | None = None
    fail_rate_max: RateThreshold | None = None
    max_pass_rate_delta: RateThreshold | None = None
    max_fail_rate_delta: RateThreshold | None = None
    max_skipped_ratio_delta: RateThreshold | None = None
    high_severity_must_pass: StrictBool = False
    fail_on_secret_leak: StrictBool = False
    max_high_severity_skipped: CountThreshold | None = None


def validate_gate_config_schema(data: object) -> GateConfigSchema:
    return GateConfigSchema.model_validate(data)
