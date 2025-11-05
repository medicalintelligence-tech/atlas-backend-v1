from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime
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


class CancerType(str, Enum):
    """Type of cancer"""

    SOLID = "solid"
    BLOOD = "blood"
    NEUROENDOCRINE = "neuroendocrine"


class DiseaseStatus(str, Enum):
    """Current disease status"""

    ACTIVE = "active"
    STABLE = "stable"
    RESPONDING = "responding"
    PROGRESSING = "progressing"
    NO_EVIDENCE = "no_evidence_of_disease"
    REMISSION = "remission"


# ============================================================================
# MOLECULAR PROFILE - Comprehensive for Phase I matching
# ============================================================================


class MolecularProfile(BaseModel):
    """Comprehensive molecular profile for any tumor site"""

    # Genomic alterations
    mutations: Optional[list[str]] = Field(
        None, description="Mutations found: 'EGFR L858R', 'KRAS G12C', 'BRAF V600E'"
    )
    amplifications: Optional[list[str]] = Field(
        None, description="Gene amplifications: 'HER2', 'MET', 'FGFR1'"
    )
    deletions: Optional[list[str]] = Field(
        None, description="Gene deletions: 'PTEN loss', 'CDKN2A'"
    )
    fusions: Optional[list[str]] = Field(
        None, description="Gene fusions: 'ALK-EML4', 'NTRK', 'RET'"
    )

    # Expression markers (critical for immunotherapy trials)
    pdl1_expression: Optional[str] = Field(
        None, description="PD-L1: 'TPS 80%', 'CPS 10', 'negative'"
    )
    her2_expression: Optional[str] = Field(
        None, description="HER2: 'IHC 3+', 'IHC 2+ FISH+', 'negative'"
    )
    hormone_receptors: Optional[list[str]] = Field(
        None,
        description="Hormone receptors: 'ER positive 90%', 'PR negative', 'AR positive'",
    )

    # Genomic signatures
    msi_status: Optional[str] = Field(
        None, description="MSI: 'MSI-high', 'MSS', 'MSI-low'"
    )
    tmb: Optional[str] = Field(
        None, description="Tumor mutational burden: 'TMB-high 15 mut/Mb', 'TMB-low'"
    )
    hrd_status: Optional[str] = Field(
        None, description="HRD: 'HRD positive', 'HRD negative'"
    )


# ============================================================================
# TUMOR MODEL - Recursive structure
# ============================================================================


class Tumor(BaseModel):
    """Individual tumor site that can have its own metastases"""

    site_id: str = Field(
        description="Unique identifier: 'primary_lung', 'liver_met_1', 'bone_met_2'"
    )
    anatomic_location: str = Field(
        description="Precise location: 'right upper lobe lung', 'liver segment 6', 'L3 vertebra'"
    )
    is_primary: bool = Field(description="True if primary tumor, False if metastasis")

    # Size and status
    size_mm: Optional[int] = Field(None, description="Size in millimeters if measured")
    status: Optional[DiseaseStatus] = Field(
        None, description="Current status of this specific tumor"
    )

    # This tumor's specific molecular profile (can differ from primary)
    molecular_profile: Optional[MolecularProfile] = Field(
        None, description="This tumor's specific molecular profile if tested"
    )

    # Metastases from THIS tumor (recursive structure)
    metastases: Optional[list["Tumor"]] = Field(
        None, description="Metastatic tumors originating from this site"
    )


# ============================================================================
# CANCER TYPE MODELS
# ============================================================================


class SolidCancer(BaseModel):
    """Solid tumor cancer"""

    cancer_type: Literal[CancerType.SOLID] = CancerType.SOLID
    histology: str = Field(
        description="Histology: 'adenocarcinoma', 'squamous cell', 'sarcoma'"
    )
    primary_tumor: Tumor = Field(
        description="Primary tumor (which may have metastases)"
    )

    # Treatment history
    prior_therapy_lines: Optional[int] = Field(
        None, description="Number of prior systemic therapy lines"
    )
    prior_therapies: Optional[list[str]] = Field(
        None,
        description="Prior treatments with specific drug names: 'erlotinib', 'pembrolizumab', 'carboplatin/pemetrexed'",
    )

    # Overall disease status
    disease_status: DiseaseStatus = Field(description="Overall disease status")


class BloodCancer(BaseModel):
    """Hematologic malignancy"""

    cancer_type: Literal[CancerType.BLOOD] = CancerType.BLOOD
    subtype: str = Field(
        description="Specific subtype: 'AML', 'DLBCL', 'CLL', 'multiple myeloma'"
    )

    # Disease characteristics
    molecular_profile: Optional[MolecularProfile] = Field(
        None, description="Current molecular profile"
    )
    disease_burden: Optional[dict[str, str]] = Field(
        None, description="Burden metrics: {'blasts': '45%', 'M-protein': '3.2 g/dL'}"
    )

    # Treatment history
    prior_therapy_lines: Optional[int] = Field(
        None, description="Number of prior therapies"
    )
    prior_therapies: Optional[list[str]] = Field(
        None,
        description="Prior treatments with specific drug names: 'venetoclax', 'ibrutinib', 'CAR-T (tisagenlecleucel)'",
    )
    prior_transplant: Optional[str] = Field(
        None,
        description="Transplant history: 'autologous 2022', 'allogeneic matched related'",
    )

    # Status
    disease_status: DiseaseStatus = Field(description="Current disease status")


class NeuroendocrineTumor(BaseModel):
    """Neuroendocrine tumor"""

    cancer_type: Literal[CancerType.NEUROENDOCRINE] = CancerType.NEUROENDOCRINE
    grade: str = Field(description="Grade: 'G1', 'G2', 'G3', 'Ki-67 15%'")
    primary_tumor: Tumor = Field(
        description="Primary tumor (which may have metastases)"
    )

    # NET-specific features
    functional: Optional[bool] = Field(
        None, description="Whether tumor produces hormones"
    )
    sstr_expression: Optional[str] = Field(
        None, description="Somatostatin receptor: 'SSTR positive', 'Dotatate avid'"
    )

    # Treatment history
    prior_therapy_lines: Optional[int] = Field(
        None, description="Number of prior therapies"
    )
    prior_therapies: Optional[list[str]] = Field(
        None,
        description="Prior treatments with specific drug names: 'octreotide', 'lutetium-177 PRRT', 'everolimus'",
    )

    disease_status: DiseaseStatus = Field(description="Current disease status")


# ============================================================================
# EXTRACTION RESULT
# ============================================================================


class PatientCancerExtraction(BaseModel):
    """Complete cancer profile for a patient"""

    cancers: Optional[list[SolidCancer | BloodCancer | NeuroendocrineTumor]] = Field(
        None,
        description="All cancers for this patient (can have multiple, can be different types)",
    )
    extraction_challenges: Optional[list[str]] = Field(
        None, description="Brief notes on any extraction difficulties"
    )


# ============================================================================
# PROMPTS
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert medical data extraction specialist focusing on cancer characteristics for Phase I clinical trial matching. Extract cancer information from medical documents and represent in structured markdown.

## Cancer Types

Identify the cancer type and use the appropriate structure:

1. **SOLID** - Solid tumors (lung, breast, colon, etc.)
2. **BLOOD** - Hematologic malignancies (leukemias, lymphomas, myelomas)
3. **NEUROENDOCRINE** - NETs (pancreatic NET, carcinoid, etc.)

## Key Extraction Principles

### Tumor Hierarchy (for Solid/NET)

- Primary tumor is the original site
- Metastases are secondary sites that spread FROM the primary
- Each metastasis can theoretically spawn further metastases (rare but possible)
- Track each tumor site separately as they may have different molecular profiles

### Molecular Profiles

- Can vary by site (primary vs mets, or between different mets)
- Extract the most recent profile for each site if available

### Disease Status

- Track overall disease status AND individual tumor status where mentioned
- Options: active, stable, responding, progressing, no_evidence_of_disease, remission

### Treatment History

- Capture specific drug names when mentioned (e.g., "erlotinib", "pembrolizumab", "carboplatin/pemetrexed")
- Include regimen combinations when applicable
- Preserve drug name formatting as documented

## Output Format

For SOLID or NEUROENDOCRINE cancers:

```
# Cancer Type: SOLID
Histology: [histology]
Disease Status: [status]
Prior Therapy Lines: [number or null]
Prior Therapies: [drug1, drug2, drug3, ...] or null

## Primary Tumor
Site ID: primary_[location]
Location: [anatomic location]
Is Primary: true
Size: [size in mm or null]
Status: [status or null]

### Molecular Profile
Mutations: [list or null]
Amplifications: [list or null]
Deletions: [list or null]
Fusions: [list or null]
PD-L1: [expression or null]
HER2: [expression or null]
Hormone Receptors: [list or null]
MSI: [status or null]
TMB: [value or null]
HRD: [status or null]

### Metastases

#### Metastasis 1
Site ID: [unique id]
Location: [anatomic location]
Is Primary: false
Size: [size or null]
Status: [status or null]

##### Molecular Profile (if different from primary)
[Include only if this met was separately profiled]

---
```

For BLOOD cancers:

```
# Cancer Type: BLOOD
Subtype: [specific subtype]
Disease Status: [status]
Prior Therapy Lines: [number or null]
Prior Therapies: [drug1, drug2, drug3, ...] or null
Prior Transplant: [details or null]

## Molecular Profile
[Same fields as above]

## Disease Burden
[Key: value pairs like blasts: 45%]

---
```

## Critical Rules

1. **Site IDs must be unique** - Use descriptive IDs like "primary_lung", "liver_met_1", "bone_met_2"
2. **Recursive structure** - If a met has spawned further mets, nest them appropriately
3. **Omit null fields** - Don't include fields with no information
4. **Specific drug names** - Capture actual drug names as documented
5. **Focus on actionable info** - Prioritize molecular findings relevant to Phase I trials

"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted cancer data for Phase I trial matching purposes.

## Validation Checks

1. **Cancer Type Classification**: Is it correctly classified as SOLID, BLOOD, or NEUROENDOCRINE?

2. **Tumor Hierarchy** (for Solid/NET):
   - Is there exactly one primary tumor?
   - Are metastases properly nested under the tumor they originated from?
   - Are all site IDs unique?

3. **Molecular Profiles**:
   - Are mutations in standard format (e.g., "EGFR L858R")?
   - Is PD-L1 expression format preserved (e.g., "TPS 50%")?
   - Are profiles correctly associated with specific tumor sites?

4. **Treatment History**:
   - Are specific drug names captured as documented?
   - Is the line count consistent with the therapies listed?
   - Are combination regimens properly formatted?

5. **Disease Status**:
   - Is overall disease status present?
   - Are individual tumor statuses consistent with overall status?

6. **Format Compliance**:
   - Proper markdown structure?
   - Correct nesting of metastases?

## What to Flag

- Missing cancer type or incorrect classification
- Multiple primary tumors (should be one)
- Non-unique site IDs
- Metastases not properly nested
- Non-standard molecular nomenclature
- Missing critical Phase I relevant data (molecular profile, prior therapies)
- Inconsistent disease status
"""

CORRECTION_SYSTEM_PROMPT = """You are a precise editor fixing cancer extraction markdown based on validation feedback.

## Guidelines

1. Generate minimal search-and-replace operations
2. Each operation must have:
   - Exact old_string (must be unique in document)
   - Corrected new_string
   - Clear reason for change

## Common Corrections

- Fix cancer type classification
- Correct tumor hierarchy (ensure proper nesting)
- Standardize molecular nomenclature (e.g., "Her-2" → "HER2")
- Fix unique IDs (e.g., "liver_met_1" appearing twice)
- Correct status values to enum options
- Preserve specific drug name formatting
"""


# ============================================================================
# ORCHESTRATION MODELS FOR MARKDOWN-BASED EXTRACTION
# ============================================================================


class MarkdownOutput(BaseModel):
    """Structured output for markdown generation"""

    markdown: str = Field(
        description="The generated markdown representation of cancer data"
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
    extraction: Optional[PatientCancerExtraction] = Field(
        default=None, description="The final extracted cancer data"
    )
    iterations_used: int = Field(description="Number of iterations required")
    total_issues_found: int = Field(description="Total number of issues encountered")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


# ============================================================================
# AGENT CREATION FUNCTIONS
# ============================================================================


def get_default_model():
    """Get default model configuration (defaults to OpenAI)"""
    return get_openai_model()


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
) -> tuple[str, int]:
    """
    Apply search and replace operations to markdown

    Returns:
        Tuple of (modified_markdown, successful_operations_count)
    """
    successful_operations = 0
    current_markdown = markdown

    for op in operations:
        if op.old_string in current_markdown:
            print(f"  ✓ Applying: {op.reason}")
            current_markdown = current_markdown.replace(op.old_string, op.new_string, 1)
            successful_operations += 1
        else:
            print(f"  ✗ Could not find text to replace: {repr(op.old_string[:100])}")

    return current_markdown, successful_operations


async def extract_to_pydantic(markdown: str, model) -> PatientCancerExtraction:
    """Convert validated markdown to Pydantic model"""
    extraction_agent = Agent(
        model=model,
        output_type=PatientCancerExtraction,
        system_prompt="You are a precise data parser. Convert the provided markdown representation of cancer data into the structured PatientCancerExtraction model. Preserve all information accurately.",
    )

    prompt = f"""
    Convert the following markdown representation of cancer data into the PatientCancerExtraction model:

    {markdown}
    """

    print("\n🔄 Converting markdown to Pydantic model...")
    result = await extraction_agent.run(prompt)
    return result.output


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================


async def extract_patient_cancer_async(
    document_text: str, max_iterations: int = 3
) -> ExtractionResult:
    """
    Extract cancer information from a medical document using iterative validation

    Args:
        document_text: Raw medical document text (progress notes, pathology, etc.)
        max_iterations: Maximum validation/correction iterations (default 3)

    Returns:
        ExtractionResult with extracted data or error

    Raises:
        ValueError: If validation fails after max_iterations
    """
    model = get_default_model()

    print("=" * 80)
    print("🚀 Starting Patient Cancer Extraction")
    print("=" * 80)

    # Step 1: Generate initial markdown
    print("\n📝 Step 1: Generating initial markdown from document...")
    print("-" * 60)

    extraction_agent = create_extraction_agent(model)
    result = await extraction_agent.run(
        f"Extract cancer information from this medical document:\n\n{document_text}"
    )
    initial_markdown = result.output.markdown.strip()

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
        initial_document=document_text,
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
        Validate the following markdown extraction against the original document.
        
        ORIGINAL DOCUMENT:
        {state.initial_document}
        
        CURRENT MARKDOWN:
        {state.current_markdown}
        """

        print("🔍 Validating markdown...")
        validation_result = await validation_agent.run(validation_prompt, deps=state)
        validation_output = validation_result.output

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
        correction_result = await correction_agent.run(correction_prompt, deps=state)
        operations = correction_result.output.operations

        print(f"📝 Generated {len(operations)} correction operations")
        print(correction_result.output.model_dump_json(indent=2))

        # Apply corrections
        print("\n⚙️  Applying corrections...")
        state.current_markdown, successful_ops = apply_corrections(
            state.current_markdown, operations
        )

        # Record tool calls
        for op in operations:
            success = op.old_string in initial_markdown or successful_ops > 0
            state.add_tool_call(op.old_string, op.new_string, success)

        print(f"✅ Applied {successful_ops}/{len(operations)} operations")

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

    print(f"✅ Successfully extracted cancer data")

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
UCSF Medical Center Oncology Progress Note
Patient: Smith, John
DOB: 3/15/1965
Date: 11/5/2024

DIAGNOSIS: Stage IV lung adenocarcinoma with brain and bone metastases

HISTORY:
Mr. Smith is a 59-year-old man with newly diagnosed lung adenocarcinoma. He presented 
in August 2024 with persistent cough and was found to have a 4.5 cm right upper lobe 
mass. Biopsy confirmed adenocarcinoma.

IMAGING:
- PET/CT (8/20/2024): 4.5 cm right upper lobe mass, SUV 12.8. Multiple brain metastases 
  (largest 2.1 cm left frontal). Lytic bone metastases in T8 and L3 vertebrae.
- Brain MRI (8/22/2024): 4 enhancing lesions (2.1 cm left frontal, 1.3 cm right parietal, 
  0.8 cm left cerebellum, 0.6 cm right occipital)

MOLECULAR TESTING (Omniseq 8/26/2024):
- EGFR: Exon 19 deletion (c.2235_2249del15, p.E746_A750del)
- KRAS: Wild type
- ALK: Negative by IHC
- ROS1: Negative
- PD-L1 TPS: 5%
- TMB: 6.2 mutations/Mb (low)
- MSI: Stable

TREATMENT HISTORY:
Started osimertinib 80 mg daily on 9/10/2024 after completing whole brain radiation 
therapy (30 Gy in 10 fractions, completed 9/5/2024).

CURRENT STATUS:
Patient reports improved cough, no headaches. ECOG PS 1. Tolerating osimertinib well 
with mild rash.

ASSESSMENT: 
Stage IV lung adenocarcinoma (EGFR exon 19 deletion) with brain and bone metastases, 
currently stable on targeted therapy.

PLAN:
- Continue osimertinib 80 mg daily
- Restaging CT chest/abdomen/pelvis and brain MRI in 6 weeks
- Supportive care with denosumab for bone health
"""


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


@pytest.mark.integration
async def test_extract_patient_cancer():
    """Integration test for patient cancer extraction with real API"""
    # Run extraction
    result = await extract_patient_cancer_async(SAMPLE_DOCUMENT, max_iterations=3)

    # Verify result
    assert result.success is True
    assert result.extraction is not None
    assert result.extraction.cancers is not None
    assert len(result.extraction.cancers) >= 1

    # Check cancer type
    cancer = result.extraction.cancers[0]
    assert isinstance(cancer, SolidCancer)
    assert cancer.cancer_type == CancerType.SOLID
    assert "adenocarcinoma" in cancer.histology.lower()

    # Check primary tumor
    primary = cancer.primary_tumor
    assert primary.is_primary is True
    assert "lung" in primary.anatomic_location.lower()
    assert primary.site_id.startswith("primary_")

    # Check metastases
    assert primary.metastases is not None
    assert len(primary.metastases) >= 2  # Brain and bone mets

    # Check for brain metastasis
    brain_mets = [
        m for m in primary.metastases if "brain" in m.anatomic_location.lower()
    ]
    assert len(brain_mets) >= 1

    # Check for bone metastasis
    bone_mets = [
        m
        for m in primary.metastases
        if "bone" in m.anatomic_location.lower()
        or "vertebra" in m.anatomic_location.lower()
    ]
    assert len(bone_mets) >= 1

    # Check molecular profile
    assert primary.molecular_profile is not None
    profile = primary.molecular_profile

    # Should have EGFR mutation
    if profile.mutations:
        assert any("EGFR" in m for m in profile.mutations)

    # Should have PD-L1 expression
    assert profile.pdl1_expression is not None
    assert "5" in profile.pdl1_expression or "TPS" in profile.pdl1_expression

    # Check treatment history
    assert cancer.prior_therapies is not None
    assert len(cancer.prior_therapies) >= 1
    assert any("osimertinib" in t.lower() for t in cancer.prior_therapies)

    # Check disease status
    assert cancer.disease_status in [
        DiseaseStatus.ACTIVE,
        DiseaseStatus.STABLE,
        DiseaseStatus.PROGRESSING,
        DiseaseStatus.RESPONDING,
    ]

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)
    print(f"\nExtracted {len(result.extraction.cancers)} cancer(s):")
    print(result.extraction.model_dump_json(indent=2))
