# What am I trying to do here?
# - I want to be able to extract the lines of therapy from some document context (in this case its a progress note)

# The reason i want to extract this data is so i can do the following types of queries
# - how many patients have at least one line of met therapy
# - group patients by primary diagnosis and lines of met therapy
# - how many patients have at least one line of met therapy and have at any point been treated with folfox


# to do this you need the following functionality

# provide some document context as text
# provide a system prompt
# provide the model to use
# provide the output model you want to use for the extraction
# provide a validator that will validate the output
# provide the max number of iterations you want to allow the model to run

# what will then happen is this data will be provided to an agent that can do the following
# - do the initial extraction
# - run the validation loop where it
# -- validates the output is correctly in line with the original system prompt
# -- provides feedback if the output is not correct
# -- uses that feedback to create search and replace tool calls to modify the ouptut
# -- does this in a loop until max iterations hit or the output is complete

# NOTE: because we're doing search and replace we might have to provide the output of the model back in as json and then allow the model to do a search and replace on that, then convert it back into the structured output model i originally wanted
# - so you need the ability to convert structured output to json which you can do with model dump json, then edit that json, then convert that json back into the original structured mode - shouldn't be too hard since you're using pydantic

# so how do you know this worked ?
# - well for us right now we'll be doing a unit test where we simply mock the output of the llm so that we can just make sure this setup is working
# - then we'll do an integration test where we run it to make sure it works with live data, once that's working you can start doing the evals i suppose

# now that i'm saying this though i think the smart move might actually be to start with the evals, cause there all you need is
# - document context
# - system prompt
# - model to use
# - validator
# - max iterations

# - expected output
# -- so given some input, what are you expecting the output to be, and this can be llm as judge which will likely be the move, but you can also validate some things deterministicly, for example number of lines of therapy you were expecting, number in metestatic setting, etc. - the thing you might need llm as judge for are the actual regimens and perhaps the ordering of them or something.

# keep in mind that you need to get this info out so that you can setup the infrastructure to enable you to ultimately provide document context, system prompt, and output model, and have that data extracted.

# once you can extract that data, then you can save it to a db and have that data queried so you can start doing some really good feasability analysis.

# so what's the simplest thing you can do right now
# - you need an output model for the resulting extraction for lines of therapy


# at a high level what i need is
# - regimen name
# - specific drugs
# - drug class


# a line of therapy can have multiple components for combo, so i feel like a line of therapy should be
# - regimen name
# - specific drugs each drug should have its own:
# -- drug class
# -- admin route
# - disease setting - non-met, locally-advanced, met
# - treamtment intent
# - start date
# - optional end date (optional because it might not be finished yet)
# - current status
# - reason for change
# - supporting evidence - list of strings (excerpts from document context)
# - confidence score
# - notes - optional


# the extraction then is just:
# - list of lines of therapy
# - optional list of extraction challenges (should be short and brief)


# TODO - need a deidentified progress note to use for this once you get it runing

# then you need to clean this up, there's a lot of shared stuff that you'll be doing, for example the whole validation loop and search and replace and the model being used is all going to be shared across this type of thing, so we can refactor later but just wanted to remind myself to do it


from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union
from datetime import date, datetime
from enum import Enum
from dataclasses import dataclass, field as dataclass_field, asdict
from pydantic_ai import Agent, RunContext, PromptedOutput, RunUsage
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from config.settings import settings
import pytest
import json


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for model with prompted output requirement"""

    model: Union[AnthropicModel, OpenAIResponsesModel]
    requires_prompted_output: bool = False


def wrap_output_type(output_type, requires_prompted_output: bool):
    """
    Conditionally wrap output type with PromptedOutput when required.

    Some model configurations (e.g., Anthropic with extended thinking enabled)
    don't support structured output tools and require PromptedOutput instead.
    This is an API restriction, not a capability difference.

    Args:
        output_type: The Pydantic model class for structured output
        requires_prompted_output: Whether the model requires PromptedOutput wrapper

    Returns:
        PromptedOutput(output_type) if requires_prompted_output, else output_type
    """
    if requires_prompted_output:
        return PromptedOutput(output_type)
    return output_type


# ============================================================================
# ENUMS
# ============================================================================


class DiseaseSettingEnum(str, Enum):
    """Classification of disease extent"""

    NON_METASTATIC = "non-metastatic"
    LOCALLY_ADVANCED = "locally-advanced"
    METASTATIC = "metastatic"


class TreatmentIntentEnum(str, Enum):
    """Intent of therapy"""

    NEOADJUVANT = "neoadjuvant"
    ADJUVANT = "adjuvant"
    PALLIATIVE = "palliative"
    CURATIVE = "curative"


class DrugClassEnum(str, Enum):
    """Type of anti-cancer drug"""

    CHEMOTHERAPY = "chemotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    IMMUNOTHERAPY = "immunotherapy"
    HORMONAL_THERAPY = "hormonal_therapy"
    RADIOPHARMACEUTICAL = "radiopharmaceutical"
    INVESTIGATIONAL = "investigational"


class AdministrationRouteEnum(str, Enum):
    """Route of administration"""

    ORAL = "oral"
    INTRAVENOUS = "intravenous"
    SUBCUTANEOUS = "subcutaneous"
    INTRAMUSCULAR = "intramuscular"


class CurrentStatusEnum(str, Enum):
    """Status of this line"""

    ONGOING = "ongoing"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"


class Drug(BaseModel):
    """Individual drug within a regimen"""

    name: str = Field(description="Drug name (generic preferred)", min_length=1)

    drug_class: Optional[DrugClassEnum] = Field(
        default=None,
        description="Type of anti-cancer drug (null if cannot be determined)",
    )

    administration_route: Optional[AdministrationRouteEnum] = Field(
        default=None,
        description="How this drug is administered (null if cannot be determined)",
    )


class LineOfTherapy(BaseModel):
    """Single line of systemic anti-cancer therapy"""

    regimen_name: str = Field(
        description="Name of the regimen (e.g., 'FOLFOX', 'palbociclib + letrozole')",
        min_length=1,
    )

    specific_drugs: List[Drug] = Field(
        description="Individual drugs in this regimen", min_length=1
    )

    disease_setting: Optional[DiseaseSettingEnum] = Field(
        default=None,
        description="Disease extent at time of treatment (null if cannot be determined)",
    )

    treatment_intent: Optional[TreatmentIntentEnum] = Field(
        default=None, description="Intent of the therapy (null if cannot be determined)"
    )

    start_date: date = Field(description="When treatment started")

    end_date: Optional[date] = Field(
        None, description="When treatment ended (null if ongoing)"
    )

    current_status: Optional[CurrentStatusEnum] = Field(
        default=None,
        description="Current status of this line (null if cannot be determined)",
    )

    reason_for_change: Optional[str] = Field(
        None,
        description="Why THIS line of therapy ended or was changed from its original plan (e.g., 'disease progression', 'unacceptable toxicity', 'completed planned course'). Leave null if therapy is ongoing or reason is unknown.",
    )

    concurrent_radiation: Optional[bool] = Field(
        default=None,
        description="Whether radiation therapy was given concurrently with this systemic therapy (null if not applicable or unknown)",
    )

    concurrent_radiation_details: Optional[str] = Field(
        default=None,
        description="Details of concurrent radiation (e.g., 'IMRT 60Gy in 15 fractions', 'SBRT 54Gy/3fx to liver', 'whole brain radiation 30Gy/10fx'). Only populate if concurrent_radiation is True.",
    )

    supporting_evidence: List[str] = Field(
        default_factory=list, description="Relevant excerpts from progress notes"
    )

    confidence_score: float = Field(
        description="Confidence in extraction (0-1)", ge=0.0, le=1.0
    )

    notes: Optional[str] = Field(None, description="Additional clarifications")

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v, info):
        """Ensure end_date is not before start_date"""
        if v is not None and "start_date" in info.data:
            if v < info.data["start_date"]:
                raise ValueError("end_date cannot be before start_date")
        return v

    @model_validator(mode="after")
    def validate_status_consistency(self):
        """Ensure status matches end_date"""
        # Only validate if current_status is not None
        if self.current_status is not None:
            if (
                self.current_status == CurrentStatusEnum.ONGOING
                and self.end_date is not None
            ):
                raise ValueError("Cannot be ongoing with an end_date")

            if self.current_status in [
                CurrentStatusEnum.COMPLETED,
                CurrentStatusEnum.DISCONTINUED,
            ]:
                if self.end_date is None:
                    raise ValueError(
                        f"Status {self.current_status.value} requires an end_date"
                    )

        return self


class LinesOfTherapyExtraction(BaseModel):
    """Complete extraction of lines of therapy"""

    lines_of_therapy: List[LineOfTherapy] = Field(
        description="All lines of therapy in chronological order"
    )

    extraction_challenges: Optional[List[str]] = Field(
        None, description="Brief notes on any extraction difficulties"
    )

    @field_validator("lines_of_therapy")
    @classmethod
    def validate_chronological(cls, v):
        """Ensure chronological order"""
        if len(v) < 2:
            return v

        for i in range(len(v) - 1):
            if v[i].start_date > v[i + 1].start_date:
                raise ValueError(f"Line {i+1} starts after Line {i+2}")

        return v


# ============================================================================
# ORCHESTRATION MODELS FOR MARKDOWN-BASED EXTRACTION
# ============================================================================


class MarkdownOutput(BaseModel):
    """Structured output for markdown generation"""

    markdown: str = Field(
        description="The generated markdown representation of lines of therapy"
    )


class ValidationResult(BaseModel):
    """Structured output for markdown validation"""

    is_valid: bool = Field(description="Whether the markdown meets all requirements")
    feedback: Optional[str] = Field(
        default=None,
        description="Detailed feedback on what needs to be fixed (only if invalid)",
    )
    specific_issues: List[str] = Field(
        default_factory=list, description="List of specific issues found"
    )


class SearchReplaceOperation(BaseModel):
    """Single search and replace operation"""

    old_string: str = Field(description="The exact text to find and replace")
    new_string: str = Field(description="The replacement text")
    reason: str = Field(description="Why this replacement is needed")


class ToolCallPlan(BaseModel):
    """Structured output for tool call generation"""

    operations: List[SearchReplaceOperation] = Field(
        description="List of search/replace operations to perform"
    )
    rationale: str = Field(description="Overall rationale for these changes")


@dataclass
class ValidationAttempt:
    """Single validation attempt with feedback"""

    iteration: int
    markdown_content: str
    is_valid: bool
    feedback: Optional[str] = None
    timestamp: datetime = dataclass_field(default_factory=datetime.now)


@dataclass
class ToolCallApplication:
    """Single tool call (search/replace) application"""

    iteration: int
    old_string: str
    new_string: str
    success: bool
    timestamp: datetime = dataclass_field(default_factory=datetime.now)


@dataclass
class ValidationState:
    """Comprehensive state tracking for the validation process"""

    # Core data
    initial_progress_note: str
    initial_markdown: str
    current_markdown: str

    # Process tracking
    current_iteration: int = 0
    max_iterations: int = 3
    is_complete: bool = False
    final_result: Optional[str] = None

    # History tracking
    validation_history: List[ValidationAttempt] = dataclass_field(default_factory=list)
    tool_call_history: List[ToolCallApplication] = dataclass_field(default_factory=list)

    # Usage tracking
    usage: RunUsage = dataclass_field(default_factory=RunUsage)

    def add_validation_attempt(self, is_valid: bool, feedback: Optional[str] = None):
        """Record a validation attempt"""
        attempt = ValidationAttempt(
            iteration=self.current_iteration,
            markdown_content=self.current_markdown,
            is_valid=is_valid,
            feedback=feedback,
        )
        self.validation_history.append(attempt)

        if is_valid:
            self.is_complete = True
            self.final_result = self.current_markdown

    def add_tool_call(self, old_string: str, new_string: str, success: bool):
        """Record a tool call application"""
        tool_call = ToolCallApplication(
            iteration=self.current_iteration,
            old_string=old_string,
            new_string=new_string,
            success=success,
        )
        self.tool_call_history.append(tool_call)

    def get_context_summary(self) -> str:
        """Generate a summary of current state for agents"""
        summary = f"""
        VALIDATION CONTEXT:
        - Current Iteration: {self.current_iteration}/{self.max_iterations}
        - Total Validation Attempts: {len(self.validation_history)}
        - Total Tool Calls Applied: {len(self.tool_call_history)}
        - Current Status: {'Complete' if self.is_complete else 'In Progress'}

        RECENT VALIDATION FEEDBACK:
        """

        # Include last 3 validation attempts
        for attempt in self.validation_history[-3:]:
            summary += f"\n  Iteration {attempt.iteration}: {'VALID' if attempt.is_valid else 'INVALID'}"
            if attempt.feedback:
                summary += f"\n    Feedback: {attempt.feedback}"

        # Include recent tool calls
        if self.tool_call_history:
            summary += f"\n\nRECENT TOOL CALLS:"
            for tool_call in self.tool_call_history[-3:]:
                summary += f"\n  Iteration {tool_call.iteration}: {'SUCCESS' if tool_call.success else 'FAILED'}"
                summary += f"\n    Changed: '{tool_call.old_string[:50]}...' -> '{tool_call.new_string[:50]}...'"

        return summary.strip()


class ExtractionResult(BaseModel):
    """Final result from the extraction process"""

    success: bool = Field(description="Whether the process completed successfully")
    extraction: Optional[LinesOfTherapyExtraction] = Field(
        default=None, description="The final extracted lines of therapy"
    )
    iterations_used: int = Field(description="Number of iterations required")
    total_issues_found: int = Field(description="Total number of issues encountered")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a world-class medical oncologist and expert medical data extraction specialist. Your task is to extract lines of systemic anti-cancer therapy from medical progress notes and represent them in a structured markdown format. Apply your deep clinical knowledge of oncology to make nuanced distinctions about disease staging, treatment intent, and therapy classifications.

## What Counts as a Line of Therapy

Include all systemic anti-cancer therapies:
- **Treatment must have been INITIATED** (first dose administered or first medication dispensed). Do not extract planned, recommended, or discussed future treatments that have not yet started.
- IV/oral chemotherapy (FOLFOX, carboplatin/paclitaxel, capecitabine, etc.)
- Targeted therapies any route (osimertinib, palbociclib, trastuzumab, bevacizumab)
- Immunotherapy (pembrolizumab, nivolumab, ipilimumab)
- Hormonal therapies (tamoxifen, letrozole, abiraterone, enzalutamide)
- Systemic radiopharmaceuticals (Lu-177 PSMA, I-131, Ra-223, Y-90)
- Concurrent chemoradiation (count the chemo component as one line)

Exclude these from lines of therapy (these are procedures/supportive care, not systemic therapy):
- "received brain radiation" -> exclude (standalone radiation)
- "underwent right lower lobectomy" -> exclude (surgery)
- "SBRT to liver lesion" -> exclude (radiation only)
- "radiofrequency ablation of liver met" -> exclude (local therapy only)
- "Zofran for nausea" -> exclude (supportive care)
- "Neulasta for neutropenia" -> exclude (growth factor support)
- "Zometa for bone health" -> exclude (supportive care, unless part of cancer regimen like in myeloma)

## When to Start a NEW Line

- Disease progression and treatment changed
- New disease setting (e.g., completed adjuvant therapy → years later developed metastases → started new therapy)
- Unplanned drug change due to progression, resistance, or severe intolerance
- Different regimen started after completing prior course

## When to CONTINUE the Same Line

- Planned sequential therapies (AC x4 then paclitaxel x4 = one line)
- Cycle progression (C1 → C2 → C3 of same regimen)
- Dose modifications (reduced osimertinib 80mg to 40mg)
- Brief treatment holds that resume (held for toxicity then restarted)
- Schedule adjustments (pembro Q3 weeks → Q6 weeks)
- Planned maintenance that was part of original treatment plan

## Treatment Pattern Recognition

Traditional IV chemotherapy: Progress notes say "Cycle 4 of FOLFOX" or "C3D1". Extract start date from first cycle, end date when explicitly stopped or completed.

Oral targeted therapies: Progress notes say "continues on osimertinib" or "tolerating erlotinib well". These run continuously until progression. Start when first mentioned, end only when explicitly stopped.

Immunotherapy: Progress notes say "pembrolizumab Q3 weeks" or "received dose #8". Treat like continuous therapy - regular dosing but not discrete "cycles". Start when first dose given, end when stopped.

Combination regimens: List all drugs. Each drug has its own drug_class and administration_route.

Concurrent chemoradiation/immunoradiation: Extract ONLY the systemic drugs (chemotherapy/immunotherapy). Use concurrent_radiation = True and populate concurrent_radiation_details with the radiation prescription (dose, fractionation, site). DO NOT list radiation as a drug. Disease_setting usually locally-advanced, treatment_intent usually curative or neoadjuvant.

Examples:
- "completed IMRT 60Gy in 15 fractions with pembrolizumab" → concurrent_radiation: True, concurrent_radiation_details: "IMRT 60Gy in 15 fractions"
- "SBRT 54Gy/3fx to liver with capecitabine" → concurrent_radiation: True, concurrent_radiation_details: "SBRT 54Gy in 3 fractions to liver"
- "whole brain radiation with concurrent temozolomide" → concurrent_radiation: True, concurrent_radiation_details: "Whole brain radiation"

## Treatment Initiation Language

Treatment STARTED (extract):
- "started on", "initiated", "received first dose", "C1D1", "began treatment"
- "continues on", "tolerating well", "dose #X received"
- "completed", "discontinued after X cycles"

Treatment PLANNED (do not extract):
- "will start", "plan to initiate", "recommend", "discussed starting"
- "pending biopsy confirmation", "scheduled to begin"
- "next line will be", "offered treatment with"

If treatment status is ambiguous, check: Has the document date passed the stated/implied start date? Has any actual administration been documented?

## Key Principles

Dates: Use YYYY-MM-DD format. If only month/year given, use first of month (June 2023 = 2023-06-01). Calculate relative dates from note date. Always provide best estimate.

End dates: Must be null if therapy is currently ongoing. Only provide when therapy explicitly stopped or changed.

Current status: ongoing (currently receiving), completed (finished planned course), discontinued (stopped early due to progression/toxicity/other reason), null (cannot determine)

Reason for change: Why this line ended (e.g., progression, toxicity, completed course), not why it started. Null if ongoing or unknown.

Disease setting:
- non-metastatic: Early stage, no distant mets
- locally-advanced: Unresectable local disease, no distant mets
- metastatic: Stage IV, distant metastases present
- null: Only use if truly cannot be determined from context

Treatment intent:
- neoadjuvant: Before surgery to shrink tumor
- adjuvant: After surgery to prevent recurrence
- curative: Intent to cure (some locally-advanced cases)
- palliative: For metastatic disease to extend life
- null: Only use if truly cannot be determined from context

Drug classification:
- chemotherapy: Traditional cytotoxics
- targeted_therapy: Drugs targeting specific mutations/pathways
- immunotherapy: Checkpoint inhibitors
- hormonal_therapy: Endocrine agents
- radiopharmaceutical: Radioactive therapeutic agents
- investigational: Drugs in clinical trials, experimental agents, or agents not yet FDA-approved (use for phase I/II trials or when mechanism/class is unclear)
- null: Only use if truly cannot be determined from context

## Clinical Trial Documentation

For investigational agents in clinical trials:
- Use the study drug code/name as it appears in documentation (e.g., "GDC-6036", "ABC-123", "study drug XYZ")
- Set drug_class to "investigational" unless the specific class is clearly stated (e.g., "investigational KRAS inhibitor" → targeted_therapy)
- Document trial details in notes field: trial identifier (NCT number), phase, dose level/cohort
- Regimen name can include "investigational" descriptor (e.g., "investigational KRAS G12C inhibitor", "GDC-6036")

## Drug Name Standardization

ALWAYS use generic drug names (lowercase), following RxNorm conventions. Convert brand names and abbreviations to generic equivalents:

Common corrections (brand/abbreviation -> generic):
- "Keytruda" -> "pembrolizumab"
- "Opdivo" -> "nivolumab"
- "Tecentriq" -> "atezolizumab"
- "Alimta" -> "pemetrexed"
- "Herceptin" -> "trastuzumab"
- "Avastin" -> "bevacizumab"
- "Tagrisso" -> "osimertinib"
- "Erbitux" -> "cetuximab"
- "5-FU" -> "fluorouracil"
- "Taxol" -> "paclitaxel"
- "Platinol" -> "cisplatin"
- "Paraplatin" -> "carboplatin"
- "Adriamycin" -> "doxorubicin"
- "Cytoxan" -> "cyclophosphamide"

## Regimen Name Standardization

Use standard abbreviations from HemOnc.org when available, otherwise use descriptive combination format with generic names (lowercase) and " + " between drugs.

Standard regimen abbreviations to corrections:
- "Folfox" -> "FOLFOX"
- "folfox" -> "FOLFOX"
- "folfirinox with oxaliplatin" -> "FOLFOX"
- "Folfiri" -> "FOLFIRI"
- "5FU/LV/irinotecan" -> "FOLFIRI"
- "Folfirinox" -> "FOLFIRINOX"
- "folfox with avastin" -> "FOLFOX + bevacizumab"
- "Folfox + Avastin" -> "FOLFOX + bevacizumab"
- "r-chop" -> "R-CHOP"
- "Rituxan-CHOP" -> "R-CHOP"

Non-standard combination corrections (use " + " separator, generic names, proper order):
- "Cisplatin/Pemetrexed/Pembrolizumab" -> "cisplatin + pemetrexed + pembrolizumab"
- "Keytruda" -> "pembrolizumab"
- "Pembro + Chemo" -> "carboplatin + pemetrexed + pembrolizumab"
- "palbociclib/letrozole" -> "palbociclib + letrozole"
- "Herceptin and Perjeta" -> "trastuzumab + pertuzumab"
- "Carboplatin-Paclitaxel-Bevacizumab" -> "carboplatin + paclitaxel + bevacizumab"

Supporting evidence: Include relevant excerpts from progress notes showing treatment start, changes, progression, and end.

Confidence score:
- 0.9-1.0 = all info clear and exact
- 0.7-0.9 = most info clear, minor inference needed
- 0.5-0.7 = moderate ambiguity
- 0.3-0.5 = significant uncertainty
- <0.3 = very uncertain

## Common Scenarios - Single Line vs Multiple Lines

Single line examples (keep as ONE line):
- "AC x4 cycles then weekly paclitaxel x12" -> ONE line (planned sequential therapy)
- "osimertinib 80mg daily, reduced to 40mg for rash" -> ONE line (dose adjustment)
- "pembrolizumab held for pneumonitis, resumed 2 weeks later" -> ONE line (brief hold with resumption)
- "picked up refill of erlotinib" -> continue SAME line (medication refill)
- "carboplatin/pemetrexed completed, continued pemetrexed maintenance" -> ONE line (planned maintenance)

Multiple line examples (create NEW lines):
- "FOLFOX for 6 months, then FOLFIRI after progression" -> TWO lines (progression, different regimen)
- "adjuvant capecitabine completed 2015, now on FOLFOX for metastatic disease 2024" -> TWO lines (different disease setting, years apart)
- "osimertinib for 2 years, then switched to chemo after T790M resistance" -> TWO lines (progression/resistance, different mechanism)
- "received palliative radiation to bone" -> ZERO lines (not systemic therapy)

## Systematic Extraction Process

Follow this structured approach to ensure complete and internally consistent extraction:

**Step 1: Initial Scan**
- Read through the entire document to understand the patient's clinical narrative
- Identify the document date - this is your reference point for determining what has been initiated vs. what is planned
- Note key clinical milestones: diagnosis, disease progression, treatment changes

**Step 2: Treatment Identification**
- Scan for all mentions of systemic anti-cancer therapies (chemotherapy, targeted therapy, immunotherapy, hormonal therapy, radiopharmaceuticals)
- Mark each mention as INITIATED (started/received/ongoing) vs. PLANNED (recommended/discussed/future)
- Only extract INITIATED treatments - do NOT extract planned or recommended future treatments

**Step 3: Line Grouping**
- Group related treatment mentions that represent the same line of therapy (e.g., "C1 of FOLFOX", "C4 of FOLFOX", "tolerating FOLFOX well")
- Determine boundaries between lines: disease progression, new disease setting, unplanned regimen change → NEW line
- Planned sequential therapies, dose adjustments, brief holds → SAME line

**Step 4: Detail Extraction for Each Line**
- Extract all required fields (regimen, drugs, dates, setting, intent, status)
- Gather supporting evidence from the progress note for each assertion
- Assign confidence based on clarity of documentation

**Step 5: Internal Consistency Check**
This step is critical - validate that your extracted fields are logically consistent with each other:

- **Status ↔ End Date**: ongoing must have null end_date; completed/discontinued must have end_date
- **Status ↔ Reason for Change**: 
  - "completed" reason should describe finishing planned course (e.g., "completed planned 6 cycles")
  - "discontinued" reason should describe unplanned stop (e.g., "progression", "toxicity", "resistance")
  - If reason mentions progression/toxicity/resistance → status MUST be "discontinued", NOT "completed"
- **Disease Setting ↔ Treatment Intent**: palliative intent typically goes with metastatic setting; neoadjuvant/adjuvant with non-metastatic
- **Dates**: start_date < end_date, lines in chronological order
- **Concurrent Radiation**: If documented, radiation should appear in concurrent_radiation_details, NOT as a drug

**Step 6: Final Review**
- Verify all INITIATED treatments are captured
- Verify NO planned/recommended treatments are included
- Check that lines are in chronological order
- Ensure markdown formatting is correct

## Output Format

Generate a markdown document with the following structure for each line of therapy:

```
## Line of Therapy 1
Regimen: [regimen name]
Disease Setting: [non-metastatic|locally-advanced|metastatic|null]
Treatment Intent: [neoadjuvant|adjuvant|curative|palliative|null]
Start Date: YYYY-MM-DD
End Date: YYYY-MM-DD or null
Status: [ongoing|completed|discontinued|null]
Reason for Change: [reason or null]
Concurrent Radiation: [true|false|null]
Concurrent Radiation Details: [e.g., "IMRT 60Gy in 15 fractions" or null]

Drugs:
  - Name: [drug name] | Class: [chemotherapy|targeted_therapy|immunotherapy|hormonal_therapy|radiopharmaceutical|null] | Route: [oral|intravenous|subcutaneous|intramuscular|null]
  - Name: [drug name] | Class: [...] | Route: [...]

Supporting Evidence:
  - "[excerpt from note]"
  - "[excerpt from note]"

Confidence: [0.0-1.0]
Notes: [optional notes]

---
```

Separate each line of therapy with `---` on its own line.
"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted lines of therapy data. Your job is to verify that the markdown representation accurately captures all relevant information from the original progress note.

## Validation Checks

1. **Completeness**: Are all lines of therapy mentioned in the progress note captured in the markdown?
2. **Accuracy**: Do the dates, drug names, disease settings, and intents match the source material?
3. **Proper Line Determination**: Are therapies correctly grouped or separated as lines?
4. **Format Compliance**: Does the markdown follow the expected structure?
5. **Supporting Evidence**: Are there relevant excerpts that justify the extraction?
6. **Confidence Appropriateness**: Does the confidence score reflect the clarity of information?
7. **Radiation Handling**: Is concurrent radiation documented in the "Concurrent Radiation" and "Concurrent Radiation Details" fields, NOT as a drug entry?
8. **Status Accuracy**: Is the status (ongoing/completed/discontinued) correct?
   - completed = finished as planned
   - discontinued = stopped early due to progression, toxicity, or other reason
9. **Reason for Change Logic**: Does "Reason for Change" explain why THIS line ENDED (not why it started)?
   - Should describe progression, toxicity, completion, etc.
   - Should NOT describe why the line was chosen or started

## What to Flag

- Missing lines of therapy
- Incorrect dates or drug names
- Wrong disease setting or treatment intent classifications
- Therapies that should be combined but are separated (or vice versa)
- Missing or weak supporting evidence
- Formatting issues
- **Radiation listed as a drug** (should use Concurrent Radiation fields instead)
- Missing concurrent radiation documentation when progress note indicates radiation was given with systemic therapy

## Important: Radiation Therapy Handling

Lines of therapy track SYSTEMIC anti-cancer treatments only. Radiation therapy should NEVER appear in the drugs list. When radiation is given concurrently with systemic therapy:
- Set "Concurrent Radiation: true"
- Populate "Concurrent Radiation Details" with dose/fractionation (e.g., "IMRT 60Gy in 15 fractions")
- List ONLY the systemic drugs (chemotherapy, immunotherapy, etc.) in the Drugs section

Provide specific, actionable feedback on what needs to be corrected."""

CORRECTION_SYSTEM_PROMPT = """You are a precise editor. Given validation feedback on extracted lines of therapy markdown, generate search-and-replace operations to fix the identified issues.

## Instructions

1. Read the validation feedback carefully
2. For each issue, create a SearchReplaceOperation that:
   - Provides the exact old_string to find (must be unique and present in current markdown)
   - Provides the corrected new_string
   - Explains the reason for the change

## Guidelines

- Be precise - old_string must match exactly
- Include enough context in old_string to make it unique
- Fix one issue at a time
- Keep changes minimal and targeted
- Preserve markdown formatting"""


# ============================================================================
# AGENT CREATION FUNCTIONS
# ============================================================================


def get_default_model() -> ModelConfig:
    """Get default model configuration (defaults to Anthropic with extended thinking)"""
    return get_anthropic_model()


def get_anthropic_model(enable_thinking: bool = False) -> ModelConfig:
    """
    Get Anthropic Claude model configuration

    Args:
        enable_thinking: Whether to enable Anthropic's extended thinking mode (default: True)
                        When enabled, PromptedOutput will be required instead of tool calls

    Returns:
        ModelConfig with model and requires_prompted_output flag
    """
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured in settings")

    if enable_thinking:
        model_settings = AnthropicModelSettings(
            max_tokens=25000,
            anthropic_thinking={"type": "enabled", "budget_tokens": 1024},
        )
    else:
        model_settings = AnthropicModelSettings(max_tokens=25000)

    model = AnthropicModel(
        "claude-sonnet-4-5",
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
        settings=model_settings,
    )

    # Anthropic with thinking enabled requires PromptedOutput
    return ModelConfig(model=model, requires_prompted_output=enable_thinking)


def get_openai_model() -> ModelConfig:
    """
    Get OpenAI model configuration

    Returns:
        ModelConfig with model (doesn't require PromptedOutput)
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured in settings")

    model_settings = OpenAIResponsesModelSettings(max_tokens=25000)
    model = OpenAIResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key=settings.openai_api_key),
        settings=model_settings,
    )

    return ModelConfig(model=model, requires_prompted_output=False)


def create_extraction_agent(model_config: ModelConfig) -> Agent:
    """Create extraction agent for generating markdown from progress notes"""
    return Agent(
        model=model_config.model,
        output_type=wrap_output_type(
            MarkdownOutput, model_config.requires_prompted_output
        ),
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )


def create_validation_agent(model_config: ModelConfig) -> Agent:
    """Create validation agent for checking markdown accuracy"""
    validator = Agent(
        model=model_config.model,
        output_type=wrap_output_type(
            ValidationResult, model_config.requires_prompted_output
        ),
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        deps_type=ValidationState,
    )

    @validator.instructions
    async def add_validation_context(ctx: RunContext[ValidationState]) -> str:
        """Add dynamic context about the validation history"""
        context = f"""
        VALIDATION CONTEXT:
        {ctx.deps.get_context_summary()}

        ORIGINAL PROGRESS NOTE LENGTH: {len(ctx.deps.initial_progress_note)} characters
        CURRENT MARKDOWN LENGTH: {len(ctx.deps.current_markdown)} characters

        INITIAL MARKDOWN:
        {ctx.deps.initial_markdown}

        CURRENT MARKDOWN:
        {ctx.deps.current_markdown}
        """
        return context

    return validator


def create_correction_agent(model_config: ModelConfig) -> Agent:
    """Create correction agent for generating search/replace operations"""
    correction_agent = Agent(
        model=model_config.model,
        output_type=wrap_output_type(
            ToolCallPlan, model_config.requires_prompted_output
        ),
        system_prompt=CORRECTION_SYSTEM_PROMPT,
        deps_type=ValidationState,
    )

    @correction_agent.instructions
    async def add_correction_context(ctx: RunContext[ValidationState]) -> str:
        """Add context for tool call generation"""
        return f"""
        CURRENT ITERATION: {ctx.deps.current_iteration}

        CURRENT MARKDOWN TO FIX:
        {ctx.deps.current_markdown}

        ORIGINAL PROGRESS NOTE:
        {ctx.deps.initial_progress_note}
        """

    return correction_agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def display_usage(usage: RunUsage, label: str = "Usage") -> None:
    """Display formatted usage information"""
    total = (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_write_tokens
        + usage.cache_read_tokens
    )
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"  Requests:              {usage.requests}")
    print(f"  Tool Calls:            {usage.tool_calls}")
    print(f"  Input Tokens:          {usage.input_tokens:,}")
    print(f"  Output Tokens:         {usage.output_tokens:,}")
    print(f"  Cache Write Tokens:    {usage.cache_write_tokens:,}")
    print(f"  Cache Read Tokens:     {usage.cache_read_tokens:,}")
    print(f"  Total Tokens:          {total:,}")
    print(f"{'='*60}")


def calculate_usage_delta(current: RunUsage, previous: RunUsage) -> RunUsage:
    """Calculate the difference between two RunUsage objects"""
    delta = RunUsage()
    delta.requests = current.requests - previous.requests
    delta.tool_calls = current.tool_calls - previous.tool_calls
    delta.input_tokens = current.input_tokens - previous.input_tokens
    delta.output_tokens = current.output_tokens - previous.output_tokens
    delta.cache_write_tokens = current.cache_write_tokens - previous.cache_write_tokens
    delta.cache_read_tokens = current.cache_read_tokens - previous.cache_read_tokens
    return delta


def apply_corrections(
    markdown: str, operations: List[SearchReplaceOperation]
) -> tuple[str, list[bool]]:
    """
    Apply search and replace operations to markdown

    Returns:
        Tuple of (modified_markdown, list_of_success_flags)
        where success_flags[i] indicates whether operations[i] succeeded
    """
    success_flags = []
    current_markdown = markdown

    for op in operations:
        if op.old_string in current_markdown:
            print(f"  ✓ Applying: {op.reason}")
            current_markdown = current_markdown.replace(op.old_string, op.new_string, 1)
            success_flags.append(True)
        else:
            print(f"  ✗ Could not find text to replace: {repr(op.old_string[:100])}")
            success_flags.append(False)

    return current_markdown, success_flags


async def extract_to_pydantic(
    markdown: str, model_config: ModelConfig, usage: RunUsage, previous_usage: RunUsage
) -> tuple[LinesOfTherapyExtraction, RunUsage]:
    """Convert validated markdown to Pydantic model

    Returns:
        Tuple of (extracted_lines_of_therapy, updated_previous_usage)
    """
    extraction_agent = Agent(
        model=model_config.model,
        output_type=wrap_output_type(
            LinesOfTherapyExtraction, model_config.requires_prompted_output
        ),
        system_prompt="You are a precise data parser. Convert the provided markdown representation of lines of therapy into the structured LinesOfTherapyExtraction model. Preserve all information accurately.",
    )

    prompt = f"""
    Convert the following markdown representation of lines of therapy into the LinesOfTherapyExtraction model:

    {markdown}
    """

    print("\n🔄 Converting markdown to Pydantic model...")
    async with extraction_agent.run_stream(prompt, usage=usage) as result:
        output = await result.get_output()

    # Display usage for this step (delta)
    step_usage = calculate_usage_delta(result.usage(), previous_usage)
    display_usage(step_usage, "Step 3: Pydantic Conversion Usage")

    return output, result.usage()


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================


async def extract_lines_of_therapy_async(
    progress_note: str, max_iterations: int = 3
) -> ExtractionResult:
    """
    Extract lines of therapy from a progress note using iterative validation

    Args:
        progress_note: Raw progress note text
        max_iterations: Maximum validation/correction iterations (default 3)

    Returns:
        ExtractionResult with extracted data or error

    Raises:
        ValueError: If validation fails after max_iterations
    """
    model_config = get_default_model()

    # Initialize usage tracking
    total_usage = RunUsage()

    print("=" * 80)
    print("🚀 Starting Lines of Therapy Extraction")
    print(f"Model: {model_config.model.__class__.__name__}")
    print(
        f"Output Type: {'PromptedOutput' if model_config.requires_prompted_output else 'Structured Tools'}"
    )
    print("=" * 80)

    # Step 1: Generate initial markdown
    print("\n📝 Step 1: Generating initial markdown from progress note...")
    print("-" * 60)

    extraction_agent = create_extraction_agent(model_config)
    previous_usage = RunUsage()  # Start with empty usage to calculate first delta

    async with extraction_agent.run_stream(
        f"Extract lines of therapy from this progress note:\n\n{progress_note}",
        usage=total_usage,
    ) as result:
        output = await result.get_output()

    # Display usage for initial extraction (delta)
    step_usage = calculate_usage_delta(result.usage(), previous_usage)
    display_usage(step_usage, "Step 1: Initial Extraction Usage")
    previous_usage = result.usage()  # Update for next delta calculation

    initial_markdown = output.markdown.strip()

    # Clean up any markdown formatting
    if "```markdown" in initial_markdown:
        initial_markdown = (
            initial_markdown.split("```markdown")[1].split("```")[0].strip()
        )
    elif "```" in initial_markdown:
        initial_markdown = initial_markdown.split("```")[1].strip()

    print(f"\n✅ Generated markdown ({len(initial_markdown.splitlines())} lines)")
    print("\n" + initial_markdown)
    print("-" * 60)

    # Step 2: Initialize validation state
    state = ValidationState(
        initial_progress_note=progress_note,
        initial_markdown=initial_markdown,
        current_markdown=initial_markdown,
        max_iterations=max_iterations,
        usage=total_usage,
    )

    print(f"\n🔄 Step 2: Starting validation loop (max {max_iterations} iterations)")
    print("-" * 60)

    validation_agent = create_validation_agent(model_config)
    correction_agent = create_correction_agent(model_config)

    # Main validation loop
    for iteration in range(max_iterations):
        state.current_iteration = iteration
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Validate current markdown
        validation_prompt = f"""
        Validate the following markdown extraction against the original progress note.
        
        ORIGINAL PROGRESS NOTE:
        {state.initial_progress_note}
        
        CURRENT MARKDOWN:
        {state.current_markdown}
        """

        print("🔍 Validating markdown...")
        async with validation_agent.run_stream(
            validation_prompt, deps=state, usage=total_usage
        ) as validation_result:
            validation_output = await validation_result.get_output()

        # Display usage for this validation step (delta)
        step_usage = calculate_usage_delta(validation_result.usage(), previous_usage)
        display_usage(step_usage, f"Iteration {iteration + 1}: Validation Usage")
        previous_usage = validation_result.usage()

        # Record validation attempt
        state.add_validation_attempt(
            is_valid=validation_output.is_valid, feedback=validation_output.feedback
        )

        if validation_output.is_valid:
            print("✅ Validation passed!")
            print(validation_output.model_dump_json(indent=2))
            break

        # Show validation feedback
        print("❌ Validation failed:")
        print(validation_output.model_dump_json(indent=2))

        # Generate corrections
        correction_prompt = f"""
        The markdown validation failed with this feedback:
        {validation_output.feedback}
        
        Specific issues:
        {chr(10).join(f"- {issue}" for issue in validation_output.specific_issues)}
        
        Generate search-and-replace operations to fix these issues.
        """

        print("\n🔧 Generating corrections...")
        async with correction_agent.run_stream(
            correction_prompt, deps=state, usage=total_usage
        ) as correction_result:
            correction_output = await correction_result.get_output()
        operations = correction_output.operations

        # Display usage for this correction step (delta)
        step_usage = calculate_usage_delta(correction_result.usage(), previous_usage)
        display_usage(step_usage, f"Iteration {iteration + 1}: Correction Usage")
        previous_usage = correction_result.usage()

        print(f"📝 Generated {len(operations)} correction operations")
        print(correction_output.model_dump_json(indent=2))

        # Apply corrections
        print("\n⚙️  Applying corrections...")
        state.current_markdown, success_flags = apply_corrections(
            state.current_markdown, operations
        )

        # Record tool calls with correct success tracking
        for op, success in zip(operations, success_flags):
            state.add_tool_call(op.old_string, op.new_string, success)

        successful_count = sum(success_flags)
        print(f"✅ Applied {successful_count}/{len(operations)} operations")

        print(f"\n📄 Markdown after iteration {iteration + 1}:")
        print("-" * 40)
        print(state.current_markdown)
        print("-" * 40)

    # Check if validation succeeded
    if not state.is_complete:
        error_msg = f"Validation failed after {max_iterations} iterations"
        print(f"\n❌ {error_msg}")
        print("\nValidation history:")
        print(
            json.dumps(
                [asdict(attempt) for attempt in state.validation_history],
                indent=2,
                default=str,
            )
        )

        raise ValueError(
            f"{error_msg}. Last feedback: {state.validation_history[-1].feedback if state.validation_history else 'None'}"
        )

    # Step 3: Final extraction to Pydantic model
    print("\n📦 Step 3: Converting to Pydantic model...")
    print("-" * 60)

    extraction, previous_usage = await extract_to_pydantic(
        state.current_markdown, model_config, total_usage, previous_usage
    )

    print(
        f"✅ Successfully extracted {len(extraction.lines_of_therapy)} lines of therapy"
    )

    # Display total workflow usage
    display_usage(total_usage, "🎯 TOTAL WORKFLOW USAGE")

    result = ExtractionResult(
        success=True,
        extraction=extraction,
        iterations_used=state.current_iteration + 1,
        total_issues_found=len([v for v in state.validation_history if not v.is_valid]),
        error_message=None,
    )

    print("\n" + "=" * 80)
    print("🎉 Extraction Complete!")
    print(result.model_dump_json(indent=2))
    print("=" * 80)

    return result


# ============================================================================
# SAMPLE DATA
# ============================================================================

SAMPLE_PROGRESS_NOTE_001 = """
UCSF Medical Center Oncology and Hematology, PLLC at The MCKAY-DEE Hospital 8934 Fargo Court, Smelterville, Il, 16853 Phone: 475-199-9015 Fax: 475-199-3790 Patient Name: Naaji Jr, Leif Patient Number: 0471662 Date: 6/27/2025 Date Of Birth: 7/06/1954 PROGRESS NOTE Chief Complaint Mr. Naaji is a 70 year old male referred by Dr Esperanza for lung cancer. Smoker Lives in Murdo-Mechanicstown 6/27/2025: About 2 weeks ago patient had complaints of fatigue, nausea, vomiting, low appetite and also low-grade fevers and overall not feeling well. His cortisol level was checked twice in am and was low at 4. So he was started on steroids with significant improvement duration of his symptoms S/p endocrine evaluation and is pending ACTH stimulation test No other new concerns or issues Active Problems Assessed \u00b7 C34.81 - Malignant neoplasm of overlapping sites of right bronchus and lung \u00b7 Z92.3 - Personal history of irradiation . Z51.12 - Encounter for antineoplastic immunotherapy . 125.10 - Atherosclerotic heart disease of native coronary artery without angina pectoris \u00b7 F10.20 - Alcohol dependence, uncomplicated History of Present Illness Mr. Naaji is a 70 year old male with PMH of DLD, CAD with PCI, Afib/PPM and AICD, extensive smoking 1PPD for >45 yrs now quit post Bx 2 weeks ago. He was having cough with blood in sputum which prompted further evaluation with CXR with concerns followed by CT chest showing Rt lung mass s.p Bronch with EBUS taht showed Lung AdenoCa. Bx of Station 7 and $R Ln was negative for cancer although 4R LN was acellular No c.o weight loss, LOA, Headaches, dizziness, No SOB. Has cough ECOG 1. Retired. Worked as a geophysicist in Welfare Assistant business Work up: PFTs: Minimal obstructive lung defect. Diffusion capacity within normal limits 8/10/2024: CT chest with IV contrast: 3.6 x 3.8 cm noncalcified right hilar mass with mild peripheral post obstructive changes and minimal right upper lobe bronchiolitis. PET/CT advised 8/26/2024: CT chest noncontrast: Right hilar mass measuring 3.6 x 3.8 cm. Mass abuts right mainstem and right upper lobe Naaji Jr, Leif DOB: 7/06/1954 (8224771) Page 1 of 6 bronchi. Stable mild right hilar lymphadenopathy. No significant mediastinal or axillary lymph nodes Bronchoscopy: Right lung lesion biopsied. Endoscopic bronchial ultrasound showed enlarged station 7 and station 4R lymph nodes and both well sampled 8/26/2024: Lung, right upper lobe needle biopsy: Non-small cell carcinoma consistent with adenocarcinoma with solid pattern. Station 7 biopsy: Negative for tumor Station 4R, biopsy: Acellular specimen Omniseq: KRAS G12C mutation+, TMB: 18.9 (high), MSI-Stable. PDL TPS 1% (No EGFR, ALK, ROS, HEr2, MRT. ROS, RET, NTRK mutation) 9/10/2024: PET scan: Hypermetabolic right hilar mass measures over 4 cm [4.2 x 3.8 cm] with maximum SUV of 27. Mass invades the hilum. No second FDG avid pulmonary lesion. No hypermetabolic mediastinal lymph nodes. No evidence of metastatic disease 9/23/2024: CT head with contrast: No acute intracranial abnormality TB discussion 9/22/24:Likely will need Pneumonectomy as SX plan after perioperative Chemo immunotherapy. Cardiac risks as well Plan : Chemo immunotherapy for 4 cycles .reassess with Scans and Present at Tumor board prior to Sx planning Interval History After 4 cycles of chemoimmunotherapy 12/27/2024: Decrease size and FDG uptake of the right hilar mass now measuring 2.2 cm with SUV uptake of 8.3 previously measured 4.2 cm with uptake of 27. New focal FDG uptake in the anterior cortex of the sternal manubrium without CT correlate, possible metastasis. 01/04/2025: CT chest with contrast: Right hilar mass with postobstructive atelectasis or scarring. Mass measures 2.2 cm. No mediastinal or left hilar adenopathy. Previously noted FDG uptake in the anterior cortex of the left manubrium without definite correlate on CT TB discussion 12/2024: Cant bx the Sternal area, would be blind and might not yield the needed info Past Medical History Narrative: DLD CAD with 2 PCI 1995 and another in 2010 Ischemic cardiomyopathy/ventricular tachycardia s/p AICD/pacemaker Afib s/p Ablation Surgical History No previous treatment history has been entered for this patient. Current Medications Continued medications: Accu-Chek Guide Glucose Meter, Accu-Chek Guide test strips, Accu-Chek Softclix Lancets, atorvastatin 40 mg tablet, Compazine 10 mg tablet, ezetimibe 10 mg tablet, folic acid 1 mg tablet, hydrocortisone 10 mg tablet, Imodium A-D 2 mg tablet, lidocaine-prilocaine 2.5 %-2.5 % topical cream, lidocaine-prilocaine 2.5 %-2.5 % topical cream, metoprolol succinate ER 100 mg tabletextended release 24 hr, Nitrostat 0.4 mg sublingual tablet, ondansetron HCl 8 mg tablet, Plavix 75 mg tablet, prednisone 50 mg tablet, Senna-S 8.6 mg-50 mg tablet. Allergies penicillin Family History of Cancer Half-sister had ?Lymphoma M. Uncle Lung cancer Social History: 1PPD cigarettes for 45 yrs, quit 20 days ago. 15 beers a week now (was drinking 6 beers a day prior to 1 year) Naaji Jr, Leif DOB: 7/06/1954 (8224771) Page 2 of 6 Alcohol Patient advises current alcohol user. beer. Marital Status Patient is married. Living Arrangement Patient lives with spouse. Occupation Retired. Worked as a geophysicist in Welfare Assistant business Tobacco Use Past tobacco smoker. Smokes tobacco daily. Tobacco Use Continued: Number of years of use: 45 years. Patient has a 45 pack year smoking history. (CQM) Patient Stated Pain: Pain Description: 0 - No pain. Review of Systems 14 point review of systems is negative except as mentioned in HPI. Vitals Vitals on 6/27/2025 1:13:00 PM: Height=67in, Weight=173.4lb, Temp=98.4f, Heart rate=80bpm, Respiration rate=16, SystolicBP=127, DiastolicBP=75, Pulse Ox=99% Physical Exam Physical Examination General: Well developed, Thin Built Neck: Neck is supple. Chest: Symmetrical. PPM/AICD+ Lungs are clear Cardiac: normal rate; regular rhythm. Abdomen: soft, non tender, non distended Back: No tenderness to palpation. Extremities: No edema. No cyanosis. Musculoskeletal: Normal range of motion. Neuro/Psych: Alert , oriented. Good understanding of our conversation today. Performance Scale ECOG 1: Restricted in physically strenuous activity but ambulatory and able to carry out work of a light or sedentary nature, e.g., light house work, office work Lab Results Lab results for 6/27/2025 Result Value Units Range Comment WBC 7.10 x10E3/uL 4.8-10.8 RBC 4.02 x10E6/uL 4.7-6.1 Hgb 12.5 g/dL 14-18 HCT 38.5 % 42-52 MCV 95.7 fL 80-94 Naaji Jr, Leif DOB: 7/06/1954 (8224771) Page 3 of 6 MCH 31.0 pg 27-31 MCHC 32.4 g/dL 33-37 RDW Ratio 14.0 % 11.5-14.5 Platelet # 276 x10E3/uL 130-400 MPV 9.6 fL 7.2-11.1 Neut # 5.770 \u00d710E3/uL 2.496-8.424 Neut % 81.2 % 50-70 Lymph% 13.3 % 20-44 MONO% 3.8 % 2-11 EOS% 1.4 % 0-5 BASO% 0.3 % 0-1 Lymph# 0.900 x10E3/uL 0.96-4.752 MONO# 0.270 x10E3/uL 0.096-1.188 EOS# 0.100 x10E3/uL 0-0.54 BASO# 0.020 x10E3/uL 0-0.108 Sodium 137 mmol/L 136-145 Potassium 4.4 mmol/L 3.5-5.1 Chloride 100 mmol/L 98-107 CO2 25 mmol/L 21-32 Glucose 186 mg/dL 70-110 BUN 10 mg/dL 7-18 Creat 0.81 mg/dL 0.7-1.3 BUN Creat Ratio 12 Ratio Calcium 8.7 mg/dL 8.5-10.1 Total Protein 6.9 g/dL 6.4-8.2 Albumin 3.4 g/dL 3.4-5 Globulin 3.5 g/dL 2.3-3.5 Total Bili 0.64 mg/dL 0-1 Alk Phos 111 U/L 46-116 ALT 38 U/L 12-96 AST 26 U/L 15-37 eGFR (African American) 114 mL/min/1.73m2 eGFR (Non-Afr. American) 94 mL/min/1.73m2 Creatine Kinase 28 U/L 39-308 Lab Results Lab results were reviewed and discussed with the patient. Radiology Naaji Jr, Leif DOB: 7/06/1954 (0471662) Page 4 of 6 Radiology results were reviewed and discussed with the patient. Impression 1. Rt Lung adenocarcinoma in a Chronic smoker Dx 08/2024: Mass measures 4.2cm on PET and SUV 27. Bronch showed enlarged station 7 and station 4R lymph nodes(acellular) s/p Bx: negative: cT2b NoM0: AJCC 8th edition Stage 2A Omniseq: KRAS G12C mutation+, TMB: 18.9 (high), MSIStable. PDL TPS 1% (No EGFR, ALK, ROS, HEr2, MRT. ROS, RET, NTRK mutation) Ct head with contrast(unable to do MRI due to PPM): no mets Treatments: Neoadjuvant Cis-Alimta-Keytruda 4cycles 10/08/24-12/10/24 Completed neoadjuvant chemoimmunotherapy. s/p PET with significant decrease in size and uptake of the Rt UL lung mass. No adenopathy Noted was SUV uptake in sternum, s/p CT chest: No sternal lesion was noted Unable to undergo lobectomy vs pneumonectomy Surgery due to cardiac issues that needs stent placement Dr Shaughnessy Completed IMRT to Lung: 60Gy in 15Fx completed 02/4 Dr Dunham c/w Keytruda q3weeks (safety based on ISABR trial) for a total of 1yr: Proceed today dose 8 today Did not add any chemo with radiation as he completed 4 doses Cis-Alimta and also ongoing CAD issues Ordered for repeat PET scan 06/2025 2. CAD with PCI in 1995 and 2010: Dr Marco Svoboda retired. EP is Dr Lo. Going to see Cardio same office. He is not on any diuretics. No known CHF No latest Echo.Follows with Dr Lindsay s/p Stress test Noted CAD with restenosis with s/p PCI 02/10/25 Also Low EF on Echo Notes requested 3. Treatment related adverse effects --- Symptomatic Adrenal insufficiency : 06/2025 : Low cortisol, normal ACTH : On Hydrocortisone 10mg bid. s/p Endocrine Dr Hino : Planned for ACTH stim test On hydrocortisone 50 mg a.m., 5 mg p.m. 4. Alcohol use: Drinks 15 beers a week now. Discussed cessation. 5. Former Smoker >45 pack yrs. Quit August 2024 Plan Completed IMRT to Lung: 60Gy in 15Fx completed 02/4 C/w Keytruda q3weeks : Dose 8 today Adrenal insuff: On hydrocortisone 15 mg a.m., 5 mg p.m. PET scan ordered for 06/2025. CT chest canceled RTC in 3 weeks post PET scan to discuss Total Time Spent on Date of Encounter I spent 42 minutes in reviewing the record, seeing the patient and documenting in the medical record. Fax to: Bart Shaughnessy~(475)119-2823;Jack M. Dreschner~(475)516-1265;Robert Dunham~(475)199-4655;Kowalska Esperanza Beiler~(475)652- 1440;Jeanie Hino~(475)119-4471; Naaji Jr, Leif DOB: 7/06/1954 (8224771) Page 5 of 6 Signed Ivette Bauman on 6/27/2025 at 3:47 PM Gooden Jr, Leif DOB: 7/06/1954 (8224771) Page 6 of 6
"""

SAMPLE_DOCUMENT_002 = """
Scripps Memorial Hospital Oncology and Hematology, PLLC at The LUCILLE Packard 9242 Lansing Parkway, Marshfield, Ia, 38879 Phone: 605-811-5770 Fax: 605-811-9650 Patient Name: Lindner Jr, Tory Patient Number: 3312677 Date: 6/7/2025 Date Of Birth: 5/26/1954 PROGRESS NOTE Chief Complaint Mr. Lindner is a 70 year old male referred by Dr Candy for lung cancer. Smoker Lives in Farragut-Boykins 6/7/2025: Unfortunately his recent PET scan has shown metastatic disease He is here to discuss discuss results and next steps in management. He had a CT head today pending results Has complaints of left hip pain No complaints of chest pain, trouble breathing, abdominal pain etc. His weight is stable Active Problems Assessed \u00b7 C34.81 - Malignant neoplasm of overlapping sites of right bronchus and lung History of Present Illness Mr. Lindner is a 70 year old male with PMH of DLD, CAD with PCI, Afib/PPM and AICD, extensive smoking 1PPD for >45 yrs now quit post Bx 2 weeks ago. He was having cough with blood in sputum which prompted further evaluation with CXR with concerns followed by CT chest showing Rt lung mass s.p Bronch with EBUS taht showed Lung AdenoCa. Bx of Station 7 and $R Ln was negative for cancer although 4R LN was acellular No c.o weight loss, LOA, Headaches, dizziness, No SOB. Has cough ECOG 1. Retired. Worked as a maths teacher in Travel Agent business Work up: PFTs: Minimal obstructive lung defect. Diffusion capacity within normal limits 6/30/2024: CT chest with IV contrast: 3.6 x 3.8 cm noncalcified right hilar mass with mild peripheral post obstructive changes and minimal right upper lobe bronchiolitis. PET/CT advised 7/16/2024: CT chest noncontrast: Right hilar mass measuring 3.6 x 3.8 cm. Mass abuts right mainstem and right upper lobe bronchi. Stable mild right hilar lymphadenopathy. No significant mediastinal or axillary lymph nodes Bronchoscopy: Right lung lesion biopsied. Endoscopic bronchial ultrasound showed enlarged station 7 and station 4R lymph nodes and both well sampled 7/16/2024: Lung, right upper lobe needle biopsy: Non-small cell carcinoma consistent with adenocarcinoma with solid pattern. Lindner Jr, Tory DOB: 5/26/1954 (6584937) Page 1 of 7 Station 7 biopsy: Negative for tumor Station 4R, biopsy: Acellular specimen Omniseq: KRAS G12C mutation+, TMB: 18.9 (high), MSI-Stable. PDL TPS 1% (No EGFR, ALK, ROS, HEr2, MRT. ROS, RET, NTRK mutation) 7/31/2024: PET scan: Hypermetabolic right hilar mass measures over 4 cm [4.2 x 3.8 cm] with maximum SUV of 27. Mass invades the hilum. No second FDG avid pulmonary lesion. No hypermetabolic mediastinal lymph nodes. No evidence of metastatic disease 8/13/2024: CT head with contrast: No acute intracranial abnormality TB discussion 8/12/24:Likely will need Pneumonectomy as SX plan after perioperative Chemo immunotherapy. Cardiac risks as well Plan : Chemo immunotherapy for 4 cycles .reassess with Scans and Present at Tumor board prior to Sx planning Interval History After 4 cycles of chemoimmunotherapy 11/16/2024: Decrease size and FDG uptake of the right hilar mass now measuring 2.2 cm with SUV uptake of 8.3 previously measured 4.2 cm with uptake of 27. New focal FDG uptake in the anterior cortex of the sternal manubrium without CT correlate, possible metastasis. 11/24/2024: CT chest with contrast: Right hilar mass with postobstructive atelectasis or scarring. Mass measures 2.2 cm. No mediastinal or left hilar adenopathy. Previously noted FDG uptake in the anterior cortex of the left manubrium without definite correlate on CT TB discussion 10/2024: Cant bx the Sternal area, would be blind and might not yield the needed info Past Medical History Narrative: DLD CAD with 2 PCI 1995 and another in 2010 Ischemic cardiomyopathy/ventricular tachycardia s/p AICD/pacemaker Afib s/p Ablation Surgical History No previous treatment history has been entered for this patient. Current Medications Added medication: prednisone 2 mg tabletdelayed release. Continued medications: Accu-Chek Guide Glucose Meter, Accu-Chek Guide test strips, Accu-Chek Softclix Lancets, atorvastatin 40 mg tablet, Compazine 10 mg tablet, ezetimibe 10 mg tablet, folic acid 1 mg tablet, Imodium A-D 2 mg tablet, lidocaine-prilocaine 2.5 %-2.5 % topical cream, lidocaine-prilocaine 2.5 %-2.5 % topical cream, metformin 500 mg tablet, metoprolol succinate ER 100 mg tabletextended release 24 hr, Nitrostat 0.4 mg sublingual tablet, ondansetron HCI 8 mg tablet, Plavix 75 mg tablet, Senna-S 8.6 mg-50 mg tablet. Discontinued medications: hydrocortisone 10 mg tablet, prednisone 50 mg tablet. Allergies penicillin Family History of Cancer Half-sister had ?Lymphoma M. Uncle Lung cancer Social History: 1PPD cigarettes for 45 yrs, quit 20 days ago. 15 beers a week now (was drinking 6 beers a day prior to 1 year) Alcohol Patient advises current alcohol user. dedrick. Lindner Jr, Tory DOB: 5/26/1954 (6584937) Page 2 of 7 Marital Status Patient is married. Living Arrangement Patient lives with spouse. Occupation Retired. Worked as a maths teacher in Share dealer Tobacco Use Past tobacco smoker. Smokes tobacco daily. Tobacco Use Continued: Number of years of use: 45 years. Patient has a 45 pack year smoking history. (CQM) Patient Stated Pain: Pain Description: 5 - Very distressing. Left buttocks Review of Systems 14 point review of systems is negative except as mentioned in HPI. Vitals Vitals on 6/7/2025 1:02:00 PM: Height=67in, Weight=171lb, Temp=98.4f, Heart rate=79bpm, Respiration rate=16, SystolicBP=127, DiastolicBP=72, Pulse Ox=99% Physical Exam Physical Examination General: Well developed, Thin Built Neck: Neck is supple. Chest: Symmetrical. PPM/AICD+ Lungs are clear Cardiac: normal rate; regular rhythm. Abdomen: soft, non tender, non distended Back: No tenderness to palpation. Extremities: No edema. No cyanosis. Musculoskeletal: Normal range of motion. Neuro/Psych: Alert , oriented. Good understanding of our conversation today. Performance Scale ECOG 1: Restricted in physically strenuous activity but ambulatory and able to carry out work of a light or sedentary nature, e.g., light house work, office work Lab Results Lab results for 5/17/2025 Result Value Units Range Comment WBC 7.10 x10E3/uL 4.8- 10.8 RBC 4.02 x10E6/uL 4.7-6.1 Hgb 12.5 g/dL 14-18 HCT 38.5 % 42-52 MCV 95.7 fL 80-94 MCH 31.0 pg 27-31 MCHC 32.4 g/dL 33-37 Lindner Jr, Tory DOB: 5/26/1954 (3312677) Page 3 of 7 RDW Ratio 14.0 % 11.5- 14.5 Platelet # 276 x10E3/uL 130- 400 MPV 9.6 fL 7.2- 11.1 Neut # 5.770 x10E3/uL 2.496- 8.424 Neut % 81.2 % 50-70 Lymph% 13.3 % 20-44 MONO% 3.8 % 2-11 EOS% 1.4 % 0-5 BASO% 0.3 % 0-1 Lymph# 0.900 x10E3/uL 0.96- 4.752 MONO# 0.270 x10E3/uL 0.096- 1.188 EOS# 0.100 x10E3/uL 0-0.54 BASO# 0.020 x10E3/uL 0-0.108 Sodium 137 mmol/L 136- 145 Potassium 4.4 mmol/L 3.5-5.1 Chloride 100 mmol/L 98-107 CO2 25 mmol/L 21-32 Glucose 186 mg/dL 70-110 BUN 10 mg/dL 7-18 Creat 0.81 mg/dL 0.7-1.3 BUN Creat Ratio 12 Ratio Calcium 8.7 mg/dL 8.5- 10.1 Total Protein 6.9 g/dL 6.4-8.2 Albumin 3.4 g/dL 3.4-5 Globulin 3.5 g/dL 2.3-3.5 Total Bili 0.64 mg/dL 0-1 Alk Phos 111 U/L 46-116 ALT 38 U/L 12-78 AST 26 U/L 15-37 eGFR (African American) 114 mL/min/1.73m2 eGFR (Non-Afr. 94 mL/min/1.73m2 Lindner Jr, Tory DOB: 5/26/1954 (3312677) Page 4 of 7 American) T4 Free 1.26 ng/dL 0.89- 1.76 This test was performed using the Siemens Centaur XPT competitive immunoassay using direct chemiluminescent technology. Values obtained from different assay methods cannot be used interchangeably. FT4 levels, regardless of value, should not be interpreted as absolute evidence of the presence or absence of disease. TSH 1.396 mIU/L 0.35- 5.5 This test was performed using the Siemens Centaur XPT third- generation assay that employs anti-FITC monoclonal antibody covalently bound to paramagnetic particles, an FITC-labeled anti-TSH capture monoclonal antibody, and a tracer concisting of a proprietary acridinium ester and anti-TSH mAb antibody conjugated to bovine serum albumin for chemiluminsecent detection. Vales obtained from different assay methods cannot be used interchangeably. TSH levels, regardless of value, should not be interpreted as absolute evidence of the presence or absence of disease. Creatine Kinase 28 U/L 39-308 ACTH 6 pg/mL 6-50 (Note)Reference range applies only to specimens collected between 7am-10am.MDFmed 1594 Midcrest Place, Suite Mozelle IA 98248512-399-2729 Ricky K. Aitken, MD, PhD Cortisol, Random 19.40 ug/dL 5.27- 22.45 Please Note: The reference interval and flagging forthis test is for an AM collection. If this is a PM collection please use: Cortisol PM: 3.44 - 16.76 ug/dLThe ADVIA Centaur Cortisol (COR) assay is a competitive immunoassay using direct chemiluminescent technology. Cortisol in the patient sample competes with acridinium ester-labeled cortisol in the Lite Reagent for binding to polyclonal rabbit anti-cortisol antibody in the Solid Phase. The polyclonal rabbit anti-cortisol antibody is bound to monoclonal mouse anti-rabbit antibody, which is covalently coupled to paramagnetic particles in the Solid Phase. Radiology Radiology results were reviewed and discussed with the patient. Print? Date of Doc. Name MD Interpretation Comment 5/25/2025 PET/CT Impression 1. Rt Lung adenocarcinoma in a Chronic smoker Dx 06/2024: Mass measures 4.2cm on PET and SUV 27. Bronch showed enlarged station 7 and station 4R lymph nodes(acellular) s/p Bx: negative: cT2b NoM0: AJCC 8th edition Stage 2A Omniseq: KRAS G12C mutation+, TMB: 18.9 (high), MSIStable. PDL TPS 1% (No EGFR, ALK, ROS, HEr2, MRT. ROS, RET, NTRK mutation) Ct head with contrast(unable to do MRI due to PPM): no mets Lindner Jr, Tory DOB: 5/26/1954 (6584937) Page 5 of 7 Treatments: -- Neoadjuvant Cis-Alimta-Keytruda 4cycles 08/28/24-10/30/24 -- Completed neoadjuvant chemoimmunotherapy. s/p PET with significant decrease in size and uptake of the Rt UL lung mass. No adenopathy -- Noted was SUV uptake in sternum, s/p CT chest: No sternal lesion was noted --- Unable to undergo lobectomy vs pneumonectomy Surgery due to cardiac issues that needs stent placement Dr Lynton --- Completed IMRT to Lung: 60Gy in 15Fx completed 02/2 Dr Qurbanov with ongoing Keytruda until 5/17/25 (ISABR trial safety of immunotherapy) -- Noted new RT adrenal mets. Lt gluteal met on PET 05/2025 Discussed that this is concerning for metastatic disease and treatment is not curable. Ordered for CT head with contrast : pending Ordered for Bx of rt adrenal nodule/Lt gluteal met whichever is easily accessible : Scheduled for 6/30/25 Will DC keytruda. Noted KRAS G12c mutation: recommend next line with Adagrasib 600mg bid (Median PFS 6.5mon, ORR 43%, OS 12.6mon, intracranial response rate 33%) Monitor LFTS, Qtc Will order Signatera on the new Metastatic bx 2. CAD with PCI in 1995 and 2010: Dr Briggs Toledano retired. EP is Dr Vos. Going to see Cardio same office. He is not on any diuretics. No known CHF No latest Echo.Follows with Dr Saarinen s/p Stress test Noted CAD with restenosis with s/p PCI 12/31/24 Also Low EF on Echo Notes requested 3. Treatment related adverse effects --- Symptomatic Adrenal insufficiency : 04/2025 : Low cortisol, normal ACTH : On Hydrocortisone 10mg bid. s/p Endocrine Dr Harley : s/p ACTH stim test : Will get latest notes Now on prednisone 4 mg daily, managed by endocrine 4. Alcohol use: Drinks 15 beers a week now. Discussed cessation. 5. Former Smoker >45 pack yrs. Quit June 2024 Discussed side effects of Adagrasib including but not limited to GI symptoms such as nausea vomiting, diarrhea, hepatotoxicity, renal toxicity, cardiac complications such as elevated QTc, risk of infections , risk of pneumonitis etc. Reading material provided Plan Unfortunately noted to have metastatic disease with right 3 cm adrenal met and left gluteal met Pending biopsy to confirm metastatic disease S/p CT head with contrast: Pending results Recommend next line with Adagrasib 600mg bid. Qtc today and every visit for now. Monitor LFTs DC Keytruda Signatera on new metastatic biopsy RTC in 4 weeks with labs, Qtc and Bx results Total Time Spent on Date of Encounter I spent 42 minutes in reviewing the record, seeing the patient and documenting in the medical record. Fax to: Ray Lynton~(605)727-5084;Carmine S. Rennell~(605)197-8280;Kehlani Harley~(605)727-3255;Paul Qurbanov~(605)811- 3986;Hepburn Candy Knudsen~(605)767-8384; Lindner Jr, Tory DOB: 5/26/1954 (6584937) Page 6 of 7 Signed Kendall Devyn Morin on 6/7/2025 at 1:53 PM Lindner Jr, Tory DOB: 5/26/1954 (6584937) Page 7 of 7
"""


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


@pytest.mark.skip
@pytest.mark.integration
async def test_extract_lines_of_therapy_sample_001():
    """Integration test for lines of therapy extraction with real API - Sample 001"""
    # Run extraction
    result = await extract_lines_of_therapy_async(
        SAMPLE_PROGRESS_NOTE_001, max_iterations=3
    )

    # Verify result
    assert result.success is True
    assert result.extraction is not None
    assert len(result.extraction.lines_of_therapy) == 2

    # Check first line (Cisplatin + Pemetrexed + Pembrolizumab)
    line1 = result.extraction.lines_of_therapy[0]
    assert (
        "Cisplatin" in line1.regimen_name or "cisplatin" in line1.regimen_name.lower()
    )
    assert (
        "Pemetrexed" in line1.regimen_name
        or "pemetrexed" in line1.regimen_name.lower()
        or "Alimta" in line1.regimen_name
    )
    assert (
        "Pembrolizumab" in line1.regimen_name
        or "pembrolizumab" in line1.regimen_name.lower()
        or "Keytruda" in line1.regimen_name
    )
    assert line1.disease_setting == DiseaseSettingEnum.LOCALLY_ADVANCED
    assert line1.treatment_intent == TreatmentIntentEnum.NEOADJUVANT
    assert line1.current_status == CurrentStatusEnum.COMPLETED
    assert line1.end_date is not None
    assert line1.start_date == date(2024, 10, 8)
    assert line1.end_date == date(2024, 12, 10)
    assert len(line1.specific_drugs) == 3  # Cisplatin, Pemetrexed, Pembrolizumab

    # Check second line (Pembrolizumab maintenance/consolidation with IMRT)
    line2 = result.extraction.lines_of_therapy[1]
    assert (
        "Pembrolizumab" in line2.regimen_name
        or "pembrolizumab" in line2.regimen_name.lower()
        or "Keytruda" in line2.regimen_name
    )
    assert line2.disease_setting == DiseaseSettingEnum.LOCALLY_ADVANCED
    assert line2.treatment_intent == TreatmentIntentEnum.CURATIVE
    assert line2.current_status == CurrentStatusEnum.ONGOING
    assert line2.end_date is None  # Should be ongoing
    assert line2.start_date == date(2025, 1, 31)
    assert len(line2.specific_drugs) == 1  # Pembrolizumab only

    # Check concurrent radiation is properly documented (IMRT 60Gy/15fx)
    assert (
        line2.concurrent_radiation is True
    ), "Line 2 should have concurrent_radiation=True (IMRT given with pembrolizumab)"
    assert (
        line2.concurrent_radiation_details is not None
    ), "Line 2 should have radiation details populated"
    assert any(
        keyword in line2.concurrent_radiation_details.lower()
        for keyword in ["60gy", "15", "imrt", "radiation"]
    ), f"Radiation details should mention IMRT 60Gy/15fx: {line2.concurrent_radiation_details}"

    print("\n" + "=" * 80)
    print("TEST PASSED - SAMPLE 001")
    print("=" * 80)
    print(f"\nExtracted {len(result.extraction.lines_of_therapy)} lines of therapy:")
    print(result.extraction.model_dump_json(indent=2))


@pytest.mark.integration
async def test_extract_lines_of_therapy_sample_002():
    """
    Integration test for lines of therapy extraction with real API - Sample 002

    Clinical scenario: Mr. Lindner with right lung adenocarcinoma who received
    neoadjuvant chemoimmunotherapy followed by consolidation radiation and
    immunotherapy, then progressed to metastatic disease.

    Key treatment timeline:
    - Line 1: Neoadjuvant cisplatin + pemetrexed + pembrolizumab (Cis-Alimta-Keytruda)
      * 4 cycles from 08/28/24 to 10/30/24
      * Non-metastatic disease setting
      * Neoadjuvant intent (planned for surgery before cardiac issues)
      * Status: Completed

    - Line 2: Pembrolizumab consolidation (with concurrent IMRT)
      * Started after completion of neoadjuvant therapy
      * Continued until progression detected 5/25/25
      * Completed IMRT to Lung: 60Gy in 15Fx completed 02/2/25
      * Based on ISABR trial (safety of immunotherapy with SBRT)
      * Status: Discontinued due to progression

    - Planned Line 3: Adagrasib 600mg bid for KRAS G12C mutation
      * Not yet started as of document date 6/7/2025
      * Will start after biopsy confirmation
      * Should NOT be extracted as a line yet (not started)

    Document date: 6/7/2025
    """
    # Run extraction
    result = await extract_lines_of_therapy_async(SAMPLE_DOCUMENT_002, max_iterations=3)

    # Verify result
    assert result.success is True
    assert result.extraction is not None

    # Should have 2 lines (neoadjuvant chemo-immuno, then consolidation immuno)
    # Adagrasib is planned but not yet started, so should NOT be included
    assert (
        len(result.extraction.lines_of_therapy) == 2
    ), f"Expected 2 lines of therapy (completed treatments), got {len(result.extraction.lines_of_therapy)}"

    # Check first line (Neoadjuvant Cisplatin + Pemetrexed + Pembrolizumab)
    line1 = result.extraction.lines_of_therapy[0]

    # Regimen should include all three drugs
    regimen_lower = line1.regimen_name.lower()
    assert any(
        drug in regimen_lower for drug in ["cisplatin", "cis"]
    ), f"Regimen should include cisplatin: {line1.regimen_name}"
    assert any(
        drug in regimen_lower for drug in ["pemetrexed", "alimta"]
    ), f"Regimen should include pemetrexed/alimta: {line1.regimen_name}"
    assert any(
        drug in regimen_lower for drug in ["pembrolizumab", "keytruda"]
    ), f"Regimen should include pembrolizumab/keytruda: {line1.regimen_name}"

    assert (
        line1.disease_setting == DiseaseSettingEnum.NON_METASTATIC
    ), "Line 1 disease setting should be non-metastatic (Stage 2A resectable disease, surgery planned but cancelled due to cardiac issues)"
    assert (
        line1.treatment_intent == TreatmentIntentEnum.NEOADJUVANT
    ), "Line 1 intent should be neoadjuvant"
    assert (
        line1.current_status == CurrentStatusEnum.COMPLETED
    ), "Line 1 should be completed"

    # Dates for line 1
    assert line1.start_date == date(
        2024, 8, 28
    ), f"Line 1 start date should be 08/28/24, got {line1.start_date}"
    assert line1.end_date == date(
        2024, 10, 30
    ), f"Line 1 end date should be 10/30/24, got {line1.end_date}"

    # Should have 3 drugs
    assert (
        len(line1.specific_drugs) == 3
    ), f"Line 1 should have 3 drugs (cisplatin, pemetrexed, pembrolizumab), got {len(line1.specific_drugs)}"

    # Check drug classes for line 1
    drug_classes = [drug.drug_class for drug in line1.specific_drugs if drug.drug_class]
    assert (
        DrugClassEnum.CHEMOTHERAPY in drug_classes
    ), "Should include chemotherapy drugs"
    assert DrugClassEnum.IMMUNOTHERAPY in drug_classes, "Should include immunotherapy"

    # Check second line (Pembrolizumab consolidation)
    line2 = result.extraction.lines_of_therapy[1]

    regimen_lower2 = line2.regimen_name.lower()
    assert any(
        drug in regimen_lower2 for drug in ["pembrolizumab", "keytruda"]
    ), f"Line 2 regimen should include pembrolizumab/keytruda: {line2.regimen_name}"

    # Disease setting should be locally-advanced during consolidation
    # (large hilar mass treated with definitive chemoradiation; metastatic disease discovered later on 5/25/25 PET)
    assert (
        line2.disease_setting == DiseaseSettingEnum.LOCALLY_ADVANCED
    ), "Line 2 disease setting should be locally-advanced (definitive chemoradiation for extensive hilar disease)"

    # Treatment intent could be curative or consolidative
    assert line2.treatment_intent in [
        TreatmentIntentEnum.CURATIVE,
        TreatmentIntentEnum.ADJUVANT,
    ], f"Line 2 intent should be curative or adjuvant, got {line2.treatment_intent}"

    # Status should be discontinued (stopped due to progression)
    assert (
        line2.current_status == CurrentStatusEnum.DISCONTINUED
    ), "Line 2 should be discontinued (DC'd Keytruda due to progression)"

    # Line 2 should have start date after line 1 completion and end date when progression noted
    assert (
        line2.start_date > line1.end_date
    ), "Line 2 should start after line 1 completed"
    assert (
        line2.end_date is not None
    ), "Line 2 should have end date (discontinued due to progression)"

    # Should have only pembrolizumab (NOT radiation as a drug)
    assert (
        len(line2.specific_drugs) == 1
    ), f"Line 2 should have 1 drug (pembrolizumab monotherapy), got {len(line2.specific_drugs)}"
    assert (
        line2.specific_drugs[0].drug_class == DrugClassEnum.IMMUNOTHERAPY
    ), "Line 2 drug should be immunotherapy"

    # Check concurrent radiation is properly documented
    assert (
        line2.concurrent_radiation is True
    ), "Line 2 should have concurrent_radiation=True (IMRT given with pembrolizumab)"
    assert (
        line2.concurrent_radiation_details is not None
    ), "Line 2 should have radiation details populated"
    assert any(
        keyword in line2.concurrent_radiation_details.lower()
        for keyword in ["60gy", "15", "imrt", "radiation"]
    ), f"Radiation details should mention IMRT 60Gy/15fx: {line2.concurrent_radiation_details}"

    # Check reason for change on line 2
    if line2.reason_for_change:
        reason_lower = line2.reason_for_change.lower()
        assert any(
            keyword in reason_lower
            for keyword in ["progression", "metastatic", "metastasis"]
        ), "Reason for stopping line 2 should mention progression or metastatic disease"

    # Verify supporting evidence exists
    assert len(line1.supporting_evidence) > 0, "Line 1 should have supporting evidence"
    assert len(line2.supporting_evidence) > 0, "Line 2 should have supporting evidence"

    print("\n" + "=" * 80)
    print("TEST PASSED - SAMPLE 002 (PROGRESSION TO METASTATIC)")
    print("=" * 80)
    print(f"\nExtracted {len(result.extraction.lines_of_therapy)} lines of therapy:")
    print(result.extraction.model_dump_json(indent=2))
    print("\n" + "=" * 80)
    print("TREATMENT TIMELINE SUMMARY")
    print("=" * 80)
    for i, line in enumerate(result.extraction.lines_of_therapy, 1):
        print(f"\nLine {i}: {line.regimen_name}")
        print(
            f"  Setting: {line.disease_setting.value if line.disease_setting else 'Not documented'}"
        )
        print(
            f"  Intent: {line.treatment_intent.value if line.treatment_intent else 'Not documented'}"
        )
        print(
            f"  Dates: {line.start_date} to {line.end_date if line.end_date else 'ongoing'}"
        )
        print(
            f"  Status: {line.current_status.value if line.current_status else 'Not documented'}"
        )
        print(f"  Drugs: {', '.join([d.name for d in line.specific_drugs])}")
    print("=" * 80)
