"""
Diagnosis Data Extraction Module

Extracts structured diagnosis information from medical documents using an iterative
validation approach with markdown-based extraction.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
from datetime import date, datetime
from enum import Enum
from dataclasses import dataclass, field as dataclass_field, asdict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from config.settings import settings
import pytest
import json


# ============================================================================
# ENUMS
# ============================================================================


class DiagnosisType(str, Enum):
    """Type of cancer diagnosis"""

    SOLID_TUMOR = "solid_tumor"
    HEMATOLOGIC = "hematologic"
    NEUROENDOCRINE = "neuroendocrine"


class DiagnosisStatus(str, Enum):
    """Current status of the cancer diagnosis"""

    ACTIVE = "active"
    COMPLETE_REMISSION = "complete_remission"
    PARTIAL_REMISSION = "partial_remission"
    STABLE = "stable"
    PROGRESSION = "progression"
    RECURRENCE = "recurrence"


class AnatomicalSite(str, Enum):
    """Anatomical sites - used for both primary sites and metastatic sites"""

    # Head & Neck
    ORAL_CAVITY = "oral_cavity"
    OROPHARYNX = "oropharynx"
    NASOPHARYNX = "nasopharynx"
    HYPOPHARYNX = "hypopharynx"
    LARYNX = "larynx"
    SALIVARY_GLAND = "salivary_gland"
    THYROID = "thyroid"
    NASAL_CAVITY_PARANASAL_SINUS = "nasal_cavity_paranasal_sinus"

    # Thoracic
    LUNG = "lung"
    PLEURA = "pleura"
    MEDIASTINUM = "mediastinum"
    THYMUS = "thymus"

    # Gastrointestinal
    ESOPHAGUS = "esophagus"
    STOMACH = "stomach"
    SMALL_INTESTINE = "small_intestine"
    COLON = "colon"
    RECTUM = "rectum"
    ANUS = "anus"
    LIVER = "liver"
    INTRAHEPATIC_BILE_DUCT = "intrahepatic_bile_duct"
    EXTRAHEPATIC_BILE_DUCT = "extrahepatic_bile_duct"
    GALLBLADDER = "gallbladder"
    AMPULLA_OF_VATER = "ampulla_of_vater"
    PANCREAS = "pancreas"

    # Genitourinary
    KIDNEY = "kidney"
    RENAL_PELVIS = "renal_pelvis"
    URETER = "ureter"
    BLADDER = "bladder"
    URETHRA = "urethra"
    PROSTATE = "prostate"
    TESTIS = "testis"
    PENIS = "penis"

    # Gynecologic
    OVARY = "ovary"
    FALLOPIAN_TUBE = "fallopian_tube"
    UTERUS = "uterus"
    CERVIX = "cervix"
    VAGINA = "vagina"
    VULVA = "vulva"

    # Breast
    BREAST = "breast"

    # Skin
    SKIN_MELANOMA = "skin_melanoma"
    SKIN_NON_MELANOMA = "skin_non_melanoma"
    SKIN_MERKEL_CELL = "skin_merkel_cell"

    # Musculoskeletal
    BONE = "bone"
    SOFT_TISSUE = "soft_tissue"

    # Central Nervous System
    BRAIN = "brain"
    SPINAL_CORD = "spinal_cord"
    MENINGES = "meninges"
    LEPTOMENINGEAL = "leptomeningeal"

    # Lymphatic
    LYMPH_NODES = "lymph_nodes"

    # Other
    PERITONEUM = "peritoneum"
    ADRENAL = "adrenal"
    SKIN = "skin"
    UNKNOWN = "unknown"
    OTHER = "other"


class HematologicSubtype(str, Enum):
    """Hematologic malignancy subtypes"""

    AML = "aml"
    ALL = "all"
    CLL = "cll"
    CML = "cml"
    MULTIPLE_MYELOMA = "multiple_myeloma"
    NHL = "nhl"
    HODGKIN = "hodgkin"
    MDS = "mds"
    OTHER = "other"


class NeuroendocrineGrade(str, Enum):
    """Neuroendocrine tumor grading based on Ki-67 proliferation index"""

    G1 = "g1"  # Ki-67 <3%
    G2 = "g2"  # Ki-67 3-20%
    G3 = "g3"  # Ki-67 >20%
    NEC = "nec"  # Neuroendocrine carcinoma (high grade, aggressive)


# ============================================================================
# DISEASE-SPECIFIC BURDEN MODELS
# ============================================================================


class LeukemiaBurden(BaseModel):
    """Disease burden for leukemias"""

    blast_percentage: float = Field(
        description="Percentage of blasts in bone marrow or peripheral blood",
        ge=0.0,
        le=100.0,
    )


class LymphomaBurden(BaseModel):
    """Disease burden for lymphomas"""

    bulky_disease: bool = Field(
        description="Presence of bulky disease (>10cm or >1/3 chest diameter)"
    )
    number_of_sites: int = Field(description="Number of involved nodal regions", ge=0)


class MyelomaBurden(BaseModel):
    """Disease burden for multiple myeloma"""

    m_protein: float = Field(description="M-protein level in g/dL", ge=0.0)


# ============================================================================
# DISEASE-SPECIFIC MODELS
# ============================================================================


class LargestLesion(BaseModel):
    """Largest measurable lesion - value always in millimeters"""

    value: float = Field(
        description="Size of largest lesion in millimeters (mm)", gt=0.0
    )


class SolidTumor(BaseModel):
    """Solid tumor specific data"""

    primary_site: AnatomicalSite = Field(
        description="Primary anatomical site of the tumor"
    )

    largest_lesion: Optional[LargestLesion] = Field(
        default=None, description="Largest measurable lesion (always in millimeters)"
    )

    metastatic_sites: Optional[List[AnatomicalSite]] = Field(
        default=None, description="Sites of metastatic disease"
    )


class Hematologic(BaseModel):
    """Hematologic malignancy specific data"""

    disease_subtype: HematologicSubtype = Field(
        description="Specific subtype of hematologic malignancy"
    )

    disease_burden: Optional[Union[LeukemiaBurden, LymphomaBurden, MyelomaBurden]] = (
        Field(
            default=None,
            description="Disease burden metrics based on subtype. Use null if not found.",
        )
    )


class Neuroendocrine(BaseModel):
    """Neuroendocrine tumor specific data"""

    primary_site: AnatomicalSite = Field(
        description="Primary site of neuroendocrine tumor"
    )

    grade: Optional[NeuroendocrineGrade] = Field(
        default=None,
        description="WHO grade based on Ki-67 proliferation index. Use null if not found.",
    )

    functional_status: Optional[bool] = Field(
        default=None,
        description="Whether tumor is hormonally active/functional. Use null if not found.",
    )

    chromogranin_a: Optional[float] = Field(
        default=None, description="Chromogranin A level in ng/mL", ge=0.0
    )

    metastatic_sites: Optional[List[AnatomicalSite]] = Field(
        default=None, description="Sites of metastatic disease"
    )


# ============================================================================
# MAIN DIAGNOSIS MODEL
# ============================================================================


class BaseDiagnosis(BaseModel):
    """Complete diagnosis information"""

    type: DiagnosisType = Field(description="Type of cancer diagnosis")

    histology: Optional[str] = Field(
        default=None,
        description="Histologic classification using WHO standard terminology (lowercase, full terms, no abbreviations). Use null if not found.",
        min_length=1,
    )

    diagnosis_date: Optional[date] = Field(
        default=None,
        description="Date of diagnosis. Use null if not found.",
    )

    status: Optional[DiagnosisStatus] = Field(
        default=None,
        description="Current status of the cancer. Use null if not found.",
    )

    disease_data: Union[SolidTumor, Hematologic, Neuroendocrine] = Field(
        description="Disease-specific data based on diagnosis type"
    )

    supporting_evidence: List[str] = Field(
        default_factory=list, description="Relevant excerpts from medical documents"
    )

    confidence_score: float = Field(
        description="Confidence in extraction (0-1)", ge=0.0, le=1.0
    )

    notes: Optional[str] = Field(default=None, description="Additional clarifications")


class DiagnosisExtraction(BaseModel):
    """Complete extraction of diagnosis data"""

    diagnoses: List[BaseDiagnosis] = Field(
        default_factory=list,
        description="List of all cancer diagnoses found. Empty list if no diagnosis data in document.",
    )

    extraction_challenges: Optional[List[str]] = Field(
        default=None, description="Brief notes on any extraction difficulties"
    )


# ============================================================================
# ORCHESTRATION MODELS FOR MARKDOWN-BASED EXTRACTION
# ============================================================================


class MarkdownOutput(BaseModel):
    """Structured output for markdown generation"""

    markdown: str = Field(
        description="The generated markdown representation of diagnosis data"
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
    initial_document: str
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
    extraction: Optional[DiagnosisExtraction] = Field(
        default=None, description="The final extracted diagnosis data"
    )
    iterations_used: int = Field(description="Number of iterations required")
    total_issues_found: int = Field(description="Total number of issues encountered")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert medical data extraction specialist. Your task is to extract structured diagnosis information from medical documents and represent it in a structured markdown format.

## Multiple Primaries and Empty Cases

**Multiple Primary Cancers**: A patient may have multiple distinct primary cancers (either synchronous - diagnosed at same time, or metachronous - diagnosed at different times). Extract each distinct primary cancer as a separate diagnosis.

**Key Distinctions**:
- **Metastases vs Second Primary**: Liver metastases from lung adenocarcinoma = ONE diagnosis with metastatic_sites=[liver]. Lung adenocarcinoma + separate prostate adenocarcinoma = TWO diagnoses.
- **Recurrence**: If a cancer recurs after remission, it's still the same primary - use status="recurrence"

**No Diagnosis Data**: If the document contains no cancer diagnosis information (e.g., screening visit, benign findings, non-oncology consult, purely administrative note), return an empty diagnosis list. This is a valid outcome.

## Diagnosis Type Classification

Classify each diagnosis into one of three types:

**solid_tumor**: Any solid mass forming cancer including common cancers like lung, breast, colon, prostate, melanoma, sarcomas, etc.

**hematologic**: Blood cancers including leukemias (AML, ALL, CLL, CML), lymphomas (NHL, Hodgkin), multiple myeloma, myelodysplastic syndromes (MDS)

**neuroendocrine**: Neuroendocrine tumors (NETs) arising from neuroendocrine cells - includes carcinoid tumors, pancreatic NETs, small bowel NETs, bronchial carcinoids

## Histology Standards

Use WHO 5th Edition Classification of Tumours terminology:
- Always lowercase
- Use full terms, never abbreviations
- Be specific when subtype information is available

Common examples:
- "adenocarcinoma" (not "ADC" or "Adenocarcinoma")
- "squamous cell carcinoma" (not "SCC" or "squamous")
- "non-small cell lung cancer" only if no further subtyping available
- "invasive ductal carcinoma" for breast
- "clear cell renal cell carcinoma" for kidney
- "serous carcinoma" for ovarian
- "diffuse large b-cell lymphoma" for NHL subtype
- "acute myeloid leukemia" for AML

If only generic diagnosis without histology: use the disease name in lowercase (e.g., "lung cancer", "breast cancer")

**If histology is not explicitly stated in the document, use null. Do not infer.**

## Diagnosis Date

Extract the date when the cancer diagnosis was first made. This is usually:
- Date of biopsy confirming malignancy
- Date pathology report was finalized
- Explicitly stated "diagnosed in [month/year]"

Use YYYY-MM-DD format. If only month/year given, use first of month (e.g., "June 2023" = "2023-06-01").

**If diagnosis date is not explicitly stated in the document, use null. Do not infer.**

## Diagnosis Status

**active**: Currently has cancer, receiving or planning treatment
**complete_remission**: No evidence of disease (NED), cancer undetectable
**partial_remission**: Cancer has responded to treatment but still detectable
**stable**: Disease present but not growing (stable disease)
**progression**: Cancer is growing or spreading despite treatment
**recurrence**: Cancer returned after previous complete remission

**If current diagnosis status is not explicitly stated in the document, use null. Do not infer.**

## Solid Tumor Specific Fields

**Primary Site**: Use the anatomical site enum value. Be as specific as possible:
- "lung" for lung cancer
- "colon" vs "rectum" (not just "colorectal")
- "breast" for breast cancer
- Use "unknown" only for cancer of unknown primary (CUP)

**Largest Lesion**: Extract the size of the largest measurable lesion.
- CRITICAL: Always convert to millimeters (mm)
- Common conversions: 1 cm = 10 mm, 2.5 cm = 25 mm, 0.5 cm = 5 mm
- If given as "4.2 cm" → extract as 42 mm
- If given as "25 mm" → extract as 25 mm
- Only extract if explicitly stated in imaging reports or physical exam

**Metastatic Sites**: List all sites where metastatic disease is documented.
- Extract from imaging reports, clinical notes, pathology
- Use general anatomical categories (liver, lung, bone, brain, lymph_nodes)
- If no metastases documented, field should be null

## Hematologic Specific Fields

**Disease Subtype**: Classify into the appropriate hematologic subtype
- aml, all, cll, cml, multiple_myeloma, nhl, hodgkin, mds, other

**Disease Burden** (varies by subtype):

For Leukemias (AML, ALL, CLL, CML):
- blast_percentage: Extract % blasts from bone marrow biopsy or peripheral blood
- Look for phrases like "30% blasts", "blast count of 45%"

For Lymphomas (NHL, Hodgkin):
- bulky_disease: true if any mass >10cm or mediastinal mass >1/3 chest width
- number_of_sites: Count distinct involved nodal regions (cervical, axillary, mediastinal, etc.)

For Multiple Myeloma:
- m_protein: Extract M-spike value in g/dL from serum protein electrophoresis
- Look for "M-protein 2.5 g/dL", "M-spike of 3.2"

**If disease burden metrics are not explicitly stated in the document, use null. Do not infer.**

## Neuroendocrine Specific Fields

**Primary Site**: Use anatomical site enum (pancreas, small_intestine, lung, stomach, rectum, appendix, etc.)

**Grade**: Based on Ki-67 proliferation index
- g1: Ki-67 <3% (well-differentiated, indolent)
- g2: Ki-67 3-20% (intermediate)
- g3: Ki-67 >20% (poorly differentiated)
- nec: Neuroendocrine carcinoma (high grade, aggressive, often called "small cell" or "large cell NEC")

**If grade is not explicitly stated in the document, use null. Do not infer.**

**Functional Status**: 
- true: Tumor secretes hormones causing symptoms (carcinoid syndrome, insulinoma, gastrinoma, etc.)
- false: Non-functional, no hormone-related symptoms
- Look for mentions of flushing, diarrhea, hypoglycemia, or specific hormone elevations

**If functional status is not explicitly stated in the document, use null. Do not infer.**

**Chromogranin A**: Tumor marker for NETs
- Extract value in ng/mL if reported
- Common reference range is 0-100 ng/mL

**Metastatic Sites**: Same as solid tumors (liver is most common for NETs)

## Supporting Evidence

Include relevant excerpts demonstrating:
- Diagnosis and histology ("Biopsy showed adenocarcinoma")
- Staging information ("PET scan shows 4.2 cm mass with liver metastases")
- Disease status ("Patient in complete remission")
- Disease burden metrics ("Bone marrow with 35% blasts")

## Confidence Score

- 0.9-1.0: All information explicit and clear
- 0.7-0.9: Most information clear, minor inference needed
- 0.5-0.7: Moderate ambiguity requiring inference
- 0.3-0.5: Significant uncertainty in extraction
- <0.3: Very uncertain, major information gaps

## Output Format

**For documents with NO diagnosis data:**
```
No cancer diagnosis information found in document.
```

**For documents with one or more diagnoses:**

Generate a markdown document with numbered diagnosis sections. For a single diagnosis, use "## Diagnosis 1". For multiple primaries, use "## Diagnosis 1", "## Diagnosis 2", etc.

```
## Diagnosis 1

Type: [solid_tumor|hematologic|neuroendocrine]
Histology: [WHO standard terminology, lowercase] or null
Diagnosis Date: YYYY-MM-DD or null
Status: [active|complete_remission|partial_remission|stable|progression|recurrence] or null

### Disease-Specific Data

[For Solid Tumor:]
Primary Site: [anatomical_site]
Largest Lesion: [value in mm] mm or null
Metastatic Sites: [site1, site2, site3] or null

[For Hematologic:]
Disease Subtype: [aml|all|cll|cml|multiple_myeloma|nhl|hodgkin|mds|other]
Disease Burden: null or:
  - Blast Percentage: [value]% (for leukemias)
  OR
  - Bulky Disease: [true|false]
  - Number of Sites: [count] (for lymphomas)
  OR
  - M-Protein: [value] g/dL (for myeloma)

[For Neuroendocrine:]
Primary Site: [anatomical_site]
Grade: [g1|g2|g3|nec] or null
Functional Status: [true|false] or null
Chromogranin A: [value] ng/mL or null
Metastatic Sites: [site1, site2] or null

Supporting Evidence:
  - "[excerpt from document]"
  - "[excerpt from document]"

Confidence: [0.0-1.0]
Notes: [optional clarifications]

## Diagnosis 2
[... repeat structure if additional primaries exist ...]
```
"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted diagnosis data. Your job is to verify that the markdown representation accurately captures all relevant information from the original medical document.

## Multiple Diagnoses and Empty Cases

1. **Multiple Primaries**: If the patient has multiple primary cancers, ensure each is captured as a separate numbered diagnosis section
2. **Empty Documents**: If there's truly no cancer diagnosis information, confirm the markdown says "No cancer diagnosis information found in document."
3. **Metastases vs Second Primary**: Verify metastases are listed within the primary diagnosis's metastatic_sites, NOT as separate diagnoses

## Validation Checks (per diagnosis)

1. **Diagnosis Type**: Is the classification (solid_tumor/hematologic/neuroendocrine) correct?
2. **Histology**: Does it follow WHO terminology? Lowercase? Full terms not abbreviations?
3. **Dates**: Is the diagnosis date accurate and in YYYY-MM-DD format?
4. **Status**: Does the status accurately reflect the current disease state?
5. **Disease-Specific Data**: Are all required fields present and accurate?
   - Solid tumors: primary_site, lesion size in MM (critical!), metastatic sites
   - Hematologic: subtype, appropriate disease burden metrics
   - Neuroendocrine: primary_site, grade, functional status
6. **Unit Consistency**: Are all lesion measurements in millimeters (mm)?
7. **Supporting Evidence**: Are there relevant excerpts justifying the extraction?
8. **Confidence**: Does the score reflect extraction certainty?

## What to Flag

- Wrong diagnosis type classification
- Incorrect or non-standard histology terminology
- Lesion sizes not in millimeters (must convert cm to mm)
- Missing required fields for the diagnosis type
- Inaccurate dates or status
- Missing or insufficient supporting evidence
- Confidence score doesn't match data quality
- Metastases incorrectly represented as separate primary diagnoses
- Multiple primaries in the document but only one captured

Provide specific, actionable feedback on what needs to be corrected."""

CORRECTION_SYSTEM_PROMPT = """You are a precise editor. Given validation feedback on extracted diagnosis markdown, generate search-and-replace operations to fix the identified issues.

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
- Preserve markdown formatting
- For unit conversions (cm to mm), be very careful with the math"""


# ============================================================================
# AGENT CREATION FUNCTIONS
# ============================================================================


def get_default_model():
    """Get default model configuration (defaults to OpenAI)"""
    return get_anthropic_model()


def get_anthropic_model():
    """Get Anthropic Claude model configuration"""
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured in settings")

    model_settings = AnthropicModelSettings(
        max_tokens=25000,
    )
    return AnthropicModel(
        "claude-sonnet-4-5",
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
        settings=model_settings,
    )


def get_openai_model():
    """Get OpenAI model configuration"""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured in settings")

    model_settings = OpenAIResponsesModelSettings(max_tokens=25000)
    return OpenAIResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key=settings.openai_api_key),
        settings=model_settings,
    )


def create_extraction_agent(model) -> Agent:
    """Create extraction agent for generating markdown from medical documents"""
    return Agent(
        model=model,
        output_type=MarkdownOutput,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )


def create_validation_agent(model) -> Agent:
    """Create validation agent for checking markdown accuracy"""
    validator = Agent(
        model=model,
        output_type=ValidationResult,
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        deps_type=ValidationState,
    )

    @validator.instructions
    async def add_validation_context(ctx: RunContext[ValidationState]) -> str:
        """Add dynamic context about the validation history"""
        context = f"""
        VALIDATION CONTEXT:
        {ctx.deps.get_context_summary()}
        
        ORIGINAL DOCUMENT LENGTH: {len(ctx.deps.initial_document)} characters
        CURRENT MARKDOWN LENGTH: {len(ctx.deps.current_markdown)} characters
        
        INITIAL MARKDOWN:
        {ctx.deps.initial_markdown}
        
        CURRENT MARKDOWN:
        {ctx.deps.current_markdown}
        """
        return context

    return validator


def create_correction_agent(model) -> Agent:
    """Create correction agent for generating search/replace operations"""
    correction_agent = Agent(
        model=model,
        output_type=ToolCallPlan,
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
        
        ORIGINAL DOCUMENT:
        {ctx.deps.initial_document}
        """

    return correction_agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


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


async def extract_to_pydantic(markdown: str, model) -> DiagnosisExtraction:
    """Convert validated markdown to Pydantic model"""
    extraction_agent = Agent(
        model=model,
        output_type=DiagnosisExtraction,
        system_prompt="You are a precise data parser. Convert the provided markdown representation of diagnosis data into the structured DiagnosisExtraction model. Preserve all information accurately. If the markdown says 'No cancer diagnosis information found', return an empty diagnoses list.",
    )

    prompt = f"""
    Convert the following markdown representation of diagnosis data into the DiagnosisExtraction model:
    
    {markdown}
    
    Remember:
    - If markdown shows "No cancer diagnosis information found", set diagnoses to empty list []
    - Each "## Diagnosis N" section becomes one BaseDiagnosis object in the diagnoses list
    - Preserve all information accurately
    """

    print("\n🔄 Converting markdown to Pydantic model...")
    async with extraction_agent.run_stream(prompt) as result:
        output = await result.get_output()
    return output


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================


async def extract_diagnosis_async(
    document: str, max_iterations: int = 3
) -> ExtractionResult:
    """
    Extract diagnosis information from a medical document using iterative validation

    Args:
        document: Raw medical document text (progress note, pathology report, etc.)
        max_iterations: Maximum validation/correction iterations (default 3)

    Returns:
        ExtractionResult with extracted data or error

    Raises:
        ValueError: If validation fails after max_iterations
    """
    model = get_default_model()

    print("=" * 80)
    print("🚀 Starting Diagnosis Data Extraction")
    print("=" * 80)

    # Step 1: Generate initial markdown
    print("\n📝 Step 1: Generating initial markdown from document...")
    print("-" * 60)

    extraction_agent = create_extraction_agent(model)
    async with extraction_agent.run_stream(
        f"Extract diagnosis information from this medical document:\n\n{document}"
    ) as result:
        output = await result.get_output()
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
        initial_document=document,
        initial_markdown=initial_markdown,
        current_markdown=initial_markdown,
        max_iterations=max_iterations,
    )

    print(f"\n🔄 Step 2: Starting validation loop (max {max_iterations} iterations)")
    print("-" * 60)

    validation_agent = create_validation_agent(model)
    correction_agent = create_correction_agent(model)

    # Main validation loop
    for iteration in range(max_iterations):
        state.current_iteration = iteration
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Validate current markdown
        validation_prompt = f"""
        Validate the following markdown extraction against the original medical document.
        
        ORIGINAL DOCUMENT:
        {state.initial_document}
        
        CURRENT MARKDOWN:
        {state.current_markdown}
        """

        print("🔍 Validating markdown...")
        async with validation_agent.run_stream(
            validation_prompt, deps=state
        ) as validation_result:
            validation_output = await validation_result.get_output()

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
            correction_prompt, deps=state
        ) as correction_result:
            correction_output = await correction_result.get_output()
        operations = correction_output.operations

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

    extraction = await extract_to_pydantic(state.current_markdown, model)

    print("✅ Successfully extracted diagnosis data")

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

SAMPLE_DOCUMENT = """
UCSF Medical Center Oncology and Hematology
Patient Name: Naaji Jr, Leif
Patient Number: 0471662
Date: 6/27/2025
Date Of Birth: 7/06/1954

PROGRESS NOTE

Chief Complaint
Mr. Naaji is a 70 year old male referred by Dr Esperanza for lung cancer.

Active Problems Assessed
· C34.81 - Malignant neoplasm of overlapping sites of right bronchus and lung

History of Present Illness
Mr. Naaji is a 70 year old male with extensive smoking 1PPD for >45 yrs now quit post Bx 2 weeks ago. 
He was having cough with blood in sputum which prompted further evaluation.

Work up:
8/26/2024: Lung, right upper lobe needle biopsy: Non-small cell carcinoma consistent with 
adenocarcinoma with solid pattern.
Station 7 biopsy: Negative for tumor
Station 4R, biopsy: Acellular specimen

Omniseq: KRAS G12C mutation+, TMB: 18.9 (high), MSI-Stable. PDL TPS 1%

9/10/2024: PET scan: Hypermetabolic right hilar mass measures over 4 cm [4.2 x 3.8 cm] with 
maximum SUV of 27. Mass invades the hilum. No second FDG avid pulmonary lesion. No hypermetabolic 
mediastinal lymph nodes. No evidence of metastatic disease

9/23/2024: CT head with contrast: No acute intracranial abnormality

Impression
1. Rt Lung adenocarcinoma in a Chronic smoker
   Dx 08/2024: Mass measures 4.2cm on PET and SUV 27.
   cT2b N0 M0: AJCC 8th edition Stage 2A
   Omniseq: KRAS G12C mutation+, TMB: 18.9 (high), MSI-Stable. PDL TPS 1%
   CT head with contrast: no mets
   
   Treatments:
   Neoadjuvant Cis-Alimta-Keytruda 4 cycles 10/08/24-12/10/24
   Completed neoadjuvant chemoimmunotherapy.
   s/p PET with significant decrease in size and uptake of the Rt UL lung mass.
   Completed IMRT to Lung: 60Gy in 15Fx completed 02/4
   C/w Keytruda q3weeks for a total of 1yr: Dose 8 today

Plan
Completed IMRT to Lung: 60Gy in 15Fx completed 02/4
C/w Keytruda q3weeks : Dose 8 today
PET scan ordered for 06/2025
RTC in 3 weeks post PET scan to discuss

Signed Ivette Bauman on 6/27/2025 at 3:47 PM
"""


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


@pytest.mark.integration
async def test_extract_diagnosis():
    """Integration test for diagnosis extraction with real API"""
    # Run extraction
    result = await extract_diagnosis_async(SAMPLE_DOCUMENT, max_iterations=3)

    # Verify result
    assert result.success is True
    assert result.extraction is not None

    # Verify we have exactly one diagnosis for this sample
    diagnoses = result.extraction.diagnoses
    assert len(diagnoses) == 1

    diagnosis = diagnoses[0]

    # Check diagnosis fields
    assert diagnosis.type == DiagnosisType.SOLID_TUMOR
    assert "adenocarcinoma" in diagnosis.histology.lower()
    assert diagnosis.diagnosis_date == date(2024, 8, 26)
    assert diagnosis.status == DiagnosisStatus.ACTIVE

    # Check solid tumor specific data
    assert isinstance(diagnosis.disease_data, SolidTumor)
    solid_tumor = diagnosis.disease_data
    assert solid_tumor.primary_site == AnatomicalSite.LUNG

    # Check largest lesion (should be in mm)
    assert solid_tumor.largest_lesion is not None
    assert solid_tumor.largest_lesion.value == 42.0  # 4.2 cm = 42 mm

    # Check no metastases
    assert (
        solid_tumor.metastatic_sites is None or len(solid_tumor.metastatic_sites) == 0
    )

    # Check supporting evidence exists
    assert len(diagnosis.supporting_evidence) > 0

    # Check confidence
    assert 0.0 <= diagnosis.confidence_score <= 1.0

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)
    print(f"\nExtracted diagnoses:")
    print(result.extraction.model_dump_json(indent=2))
