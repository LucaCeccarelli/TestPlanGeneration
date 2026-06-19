"""Pydantic data models for the entire test plan generation system.

Implements the ISO/IEC/IEEE 29119-3:2021 document hierarchy:

    TestPlan -> FeatureSet -> TestCondition -> TestCoverageItem -> TestCase

`RequirementChunk` (Axis A output) also lives here so that every module in the
project imports its models from a single place.

All models use the Pydantic v2 API and forbid extra fields to catch schema
drift early.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class RequirementChunk(BaseModel):
    """One semantically coherent chunk of the source norm (Axis A output)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description='Unique id: "<norm-slug>_<index:04d>"')
    text: str = Field(..., description="Raw chunk text from the norm")
    section: str = Field(..., description='Norm section number, e.g. "7.2.1"')
    page_start: int = Field(..., description="First PDF page the chunk appears on")
    is_normative: bool = Field(
        ..., description="True if the chunk contains SHALL/MUST/SHOULD/MAY"
    )
    modals: list[str] = Field(
        default_factory=list,
        description='Modal verbs found in the chunk, e.g. ["shall", "must not"]',
    )
    source_norm: str = Field(..., description='Source norm name, e.g. "ISO/IEC 18013-5"')
    # --- A1: two-pass chunking ---
    is_full_clause: bool = Field(
        default=False,
        description=(
            "True if the chunk survived the first pass intact (fits within CHUNK_SIZE "
            "without further splitting). Acts as a downstream confidence signal."
        ),
    )
    # --- A3: cross-reference resolution and defined-terms injection ---
    context_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Short excerpts (first 150 chars) of other norm chunks that this chunk "
            "cross-references via '§ / clause / section / table / annex' patterns."
        ),
    )
    defined_terms: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Subset of the norm's Terms and Definitions (§3) whose keys appear in "
            "this chunk's text. Injected to give agents grounded definitions."
        ),
    )


class TestInput(BaseModel):
    """ISO 29119-3 SS7.3.5 -- one numbered input + expected result pair.

    Maps to SS7.3.5.7 "Inputs" and SS7.3.5.8 "Expected results".
    """

    model_config = ConfigDict(extra="forbid")

    input_number: int = Field(..., description="Sequential number within the test case")
    action: str = Field(
        ..., description="SS7.3.5.7 -- action to bring the item to a testable state"
    )
    expected_result: str = Field(
        ..., description="SS7.3.5.8 -- observable output for this specific action"
    )


class TestCase(BaseModel):
    """ISO 29119-3 SS7.3.5 -- core test case.

    All fields marked SHALL in Annex B SSB.1.7.b unless noted.
    """

    model_config = ConfigDict(extra="forbid")

    tc_id: str = Field(
        ..., description='SS7.3.5.2 SHALL -- e.g. "TC-18013-5-<section>-<seq>"'
    )
    objective: str = Field(
        ..., description="SS7.3.5.3 SHOULD -- brief focus/title of the test case"
    )
    priority: Literal["High", "Medium", "Low"] = Field(
        ..., description="SS7.3.5.4 SHALL"
    )
    traceability: str = Field(
        ..., description="SS7.3.5.5 SHALL -- chunk_id + section ref from Axis A"
    )
    preconditions: list[str] = Field(
        ..., description="SS7.3.5.6 SHALL -- required environment state before execution"
    )
    inputs: list[TestInput] = Field(
        ..., description="SS7.3.5.7 SHALL -- minimum 2 inputs"
    )
    actual_results: str = Field(
        default="",
        description="SS7.3.5.9 SHALL -- placeholder filled during execution",
    )
    requirement_type: Literal["SHALL", "SHOULD", "MAY"] = Field(
        ..., description="Derived from the source norm modal verb"
    )
    coverage_item_id: str = Field(
        ..., description="Link to TestCoverageItem (traceability chain)"
    )
    feature_set: str = Field(
        ..., description='Functional area of the norm, e.g. "Authentication"'
    )
    notes: str = Field(default="")


class TestCoverageItem(BaseModel):
    """ISO 29119-3 SS7.3.4 -- intermediate layer between condition and case.

    SHALL per Annex B SSB.1.7.a.
    """

    model_config = ConfigDict(extra="forbid")

    tci_id: str = Field(..., description='e.g. "TCI-18013-5-007-001"')
    description: str = Field(
        ..., description="SS7.3.4.3 -- what is expected to be covered"
    )
    priority: Literal["High", "Medium", "Low"] = Field(
        ..., description="SS7.3.4.4"
    )
    traceability: str = Field(
        ..., description="SS7.3.4.5 -- ref to test condition / feature set"
    )


class TestCondition(BaseModel):
    """ISO 29119-3 SS7.2.5 -- one verifiable item from the norm."""

    model_config = ConfigDict(extra="forbid")

    tc_condition_id: str = Field(..., description='e.g. "TCOND-18013-5-007"')
    description: str = Field(..., description="SS7.2.5.3 -- what can be tested")
    priority: Literal["High", "Medium", "Low"]
    traceability: str = Field(
        ..., description="SS7.2.5.5 -- ref to feature set and source norm section"
    )
    coverage_items: list[TestCoverageItem] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(default_factory=list)


class FeatureSet(BaseModel):
    """ISO 29119-3 SS7.2.4 -- logical subset of the test item."""

    model_config = ConfigDict(extra="forbid")

    fs_id: str = Field(..., description='e.g. "FS-18013-5-007"')
    objective: str = Field(..., description="SS7.2.4.3")
    priority: Literal["High", "Medium", "Low"]
    traceability: str = Field(..., description="SS7.2.4.6 -- ref to norm sections")
    test_conditions: list[TestCondition] = Field(default_factory=list)


class TestPlan(BaseModel):
    """ISO 29119-3 SS6.2 -- top-level test plan document."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(..., description='e.g. "TP-ISO-IEC-18013-5-001"')
    title: str
    norm_reference: str = Field(..., description='e.g. "ISO/IEC 18013-5:2021"')
    generation_date: str
    test_scope: str = Field(..., description="SS6.2.4.3 -- features in/out of scope")
    assumptions: list[str] = Field(default_factory=list, description="SS6.2.4.4")
    feature_sets: list[FeatureSet] = Field(default_factory=list)
    coverage_sections: list[str] = Field(
        default_factory=list,
        description="Norm sections covered by at least one test case",
    )
