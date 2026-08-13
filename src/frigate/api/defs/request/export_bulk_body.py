"""Request bodies for bulk export operations."""

from typing import Annotated

from pydantic import BaseModel, Field


class ExportBulkDeleteBody(BaseModel):
    """Request body for bulk deleting exports."""

    # List of export IDs with at least one element and each element with at least one char
    ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class ExportBulkReassignBody(BaseModel):
    """Request body for bulk reassigning exports to a case."""

    # List of export IDs with at least one element and each element with at least one char
    ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    export_case_id: str | None = Field(
        default=None,
        max_length=30,
        description="Case ID to assign to, or null to unassign from current case",
    )
