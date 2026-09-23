"""Tool definitions: argument models, result models, and the registry.

Each tool has one Pydantic argument model. It produces the JSON schema sent
to the model and validates arguments before anything is executed. Result
models validate what the backend returned, so a malformed payload is caught
here and never reaches the model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .mock_backend import known_metrics, known_services

# --------------------------------------------------------------------------
# Argument models
# --------------------------------------------------------------------------


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ServiceField(ToolArgs):
    service: str = Field(description="Service name. Must be one of the known services.")

    @field_validator("service")
    @classmethod
    def _known(cls, value: str) -> str:
        name = value.strip().lower()
        if name not in known_services():
            raise ValueError(f"unknown service '{value}'. Known services: {', '.join(known_services())}")
        return name


class _WindowFields(_ServiceField):
    start_time: datetime = Field(description="Window start, ISO 8601 UTC, e.g. 2026-09-22T14:00:00Z.")
    end_time: datetime = Field(description="Window end, ISO 8601 UTC. Must be after start_time.")


class SearchLogsArgs(_WindowFields):
    query: str = Field(
        default="",
        max_length=200,
        description="Case-insensitive substring matched against each event's level and message. "
        "Empty returns every event in the window.",
    )

    @field_validator("query")
    @classmethod
    def _normalize(cls, value: str) -> str:
        # Matching is case-insensitive, so normalising here also makes two
        # differently-cased versions of the same search de-duplicate.
        return value.strip().lower()


class GetMetricsArgs(_WindowFields):
    metric: str = Field(description="Metric name. Must be one of the known metrics.")

    @field_validator("metric")
    @classmethod
    def _known(cls, value: str) -> str:
        name = value.strip().lower()
        if name not in known_metrics():
            raise ValueError(f"unknown metric '{value}'. Known metrics: {', '.join(known_metrics())}")
        return name


class GetDeploymentsArgs(_WindowFields):
    pass


class GetServiceDependenciesArgs(_ServiceField):
    pass


class ResolveTimeRangeArgs(ToolArgs):
    expression: str = Field(
        max_length=120,
        description="The user's time expression, copied from their message, e.g. 'yesterday afternoon'.",
    )


class CreateIncidentNoteArgs(ToolArgs):
    title: str = Field(max_length=120, description="Short title for the incident.")
    summary: str = Field(max_length=1200, description="What happened, in a few sentences.")
    evidence: list[str] = Field(
        max_length=20,
        description="Observation IDs supporting the note, e.g. ['obs_002', 'obs_004']. "
        "Only observations that succeeded can be cited.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list, max_length=10, description="Concrete next steps."
    )

    @field_validator("recommended_actions")
    @classmethod
    def _bounded(cls, value: list[str]) -> list[str]:
        for item in value:
            if len(item) > 300:
                raise ValueError("each recommended action must be at most 300 characters")
        return value


# --------------------------------------------------------------------------
# Result models (validate what the backend returned)
# --------------------------------------------------------------------------


class MetricPoint(BaseModel):
    timestamp: datetime
    value: float


class LogEvent(BaseModel):
    timestamp: datetime
    level: str
    message: str


class Deployment(BaseModel):
    service: str
    version: str
    deployed_at: datetime
    status: str
    author: str | None = None


class Dependencies(BaseModel):
    service: str
    upstream: list[str]
    downstream: list[str]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

Category = Literal["data", "internal", "action", "control"]


class ToolSpec(BaseModel):
    name: str
    description: str
    args_model: type[ToolArgs]
    category: Category

    def openai_schema(self) -> dict:
        """The Responses API tool shape: flat, not nested under "function"."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {"type": "function", "name": self.name, "description": self.description, "parameters": schema}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="resolve_time_range",
        description=(
            "Convert a time expression from the user's message into a UTC start and end time. "
            "Use this whenever the user describes a time in words or without a date "
            "('yesterday afternoon', 'last 2 hours', '2 PM to 4 PM'). Do not calculate dates yourself. "
            "If the user already gave full ISO 8601 timestamps, use those directly instead."
        ),
        args_model=ResolveTimeRangeArgs,
        category="internal",
    ),
    ToolSpec(
        name="get_metrics",
        description=(
            "Return a time series for one metric on one service, with a summary stating the baseline, "
            "whether a spike was detected, and when it started. Usually the first step when asked why "
            "something went wrong, because it establishes whether anything actually changed."
        ),
        args_model=GetMetricsArgs,
        category="data",
    ),
    ToolSpec(
        name="get_deployments",
        description="Return releases for a service in a window, with version, time and status.",
        args_model=GetDeploymentsArgs,
        category="data",
    ),
    ToolSpec(
        name="search_logs",
        description=(
            "Return log events for a service in a window. Narrow the window around a known spike and "
            "use a query term to find the specific errors behind it."
        ),
        args_model=SearchLogsArgs,
        category="data",
    ),
    ToolSpec(
        name="get_service_dependencies",
        description=(
            "Return the upstream and downstream services of one service. Use it when a service looks "
            "affected but nothing local explains it, then check the upstream service's metrics."
        ),
        args_model=GetServiceDependenciesArgs,
        category="data",
    ),
    ToolSpec(
        name="create_incident_note",
        description=(
            "Record a structured incident note. Use it only after an investigation that reached a "
            "finding. Do not use it for lookups, clarifying questions, or when nothing abnormal was found."
        ),
        args_model=CreateIncidentNoteArgs,
        category="action",
    ),
]

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
CATEGORY_BY_NAME: dict[str, str] = {spec.name: spec.category for spec in TOOL_SPECS}
CATEGORY_BY_NAME["submit_response"] = "control"
