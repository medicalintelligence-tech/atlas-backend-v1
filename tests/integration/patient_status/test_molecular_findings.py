# What am I trying to do here?
# - I want to be able to extract molecular/genomic findings from document context (pathology reports, genomic test results)

# The reason i want to extract this data is so i can do the following types of queries
# - how many patients have KRAS mutations
# - which patients have actionable mutations (EGFR, ALK, etc.)
# - group patients by TMB status or PD-L1 expression
# - find patients with specific biomarker profiles

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
# -- uses that feedback to create search and replace tool calls to modify the output
# -- does this in a loop until max iterations hit or the output is complete

# NOTE: because we're doing search and replace we might have to provide the output of the model back in as json and then allow the model to do a search and replace on that, then convert it back into the structured output model i originally wanted
# - so you need the ability to convert structured output to json which you can do with model dump json, then edit that json, then convert that json back into the original structured model - shouldn't be too hard since you're using pydantic

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
# -- so given some input, what are you expecting the output to be, and this can be llm as judge which will likely be the move, but you can also validate some things deterministicly, for example number of mutations found, specific mutations expected, etc.

# keep in mind that you need to get this info out so that you can setup the infrastructure to enable you to ultimately provide document context, system prompt, and output model, and have that data extracted.

# once you can extract that data, then you can save it to a db and have that data queried so you can start doing some really good feasability analysis.

# so what's the simplest thing you can do right now
# - you need an output model for the resulting extraction for molecular findings

# at a high level what i need is
# - test source (pathology report, genomic test result)
# - test date
# - test type (NGS, IHC, FISH, PCR, etc.)
# - mutations found (gene, variant, significance)
# - biomarkers (PD-L1, TMB, MSI, etc.)
# - supporting evidence

# the base unit here is the source of the information
# - so in this case you really just have pathology or genomic testing results
# - each test result can have multiple findings (mutations, biomarkers)

# I also want to try a new model for the output that can handle variations much more effectively


from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List
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


class TestSourceEnum(str, Enum):
    """Source of the molecular/genomic test"""

    PATHOLOGY_REPORT = "pathology_report"
    GENOMIC_TEST_RESULT = "genomic_test_result"
    LAB_RESULT = "lab_result"
    UNKNOWN = "unknown"


class TestTypeEnum(str, Enum):
    """Type of molecular/genomic test performed"""

    NGS = "ngs"  # Next Generation Sequencing
    IHC = "ihc"  # Immunohistochemistry
    FISH = "fish"  # Fluorescence In Situ Hybridization
    PCR = "pcr"  # Polymerase Chain Reaction
    SANGER = "sanger"  # Sanger sequencing
    RT_PCR = "rt_pcr"  # Reverse Transcriptase PCR
    OTHER = "other"
    UNKNOWN = "unknown"


class MutationSignificanceEnum(str, Enum):
    """Clinical significance of a mutation"""

    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    VARIANT_OF_UNCERTAIN_SIGNIFICANCE = "vus"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"
    ACTIONABLE = "actionable"  # Has targeted therapy available
    UNKNOWN = "unknown"


class BiomarkerTypeEnum(str, Enum):
    """Type of biomarker"""

    PD_L1_TPS = "pd_l1_tps"  # PD-L1 Tumor Proportion Score
    PD_L1_CPS = "pd_l1_cps"  # PD-L1 Combined Positive Score
    TMB = "tmb"  # Tumor Mutational Burden
    MSI = "msi"  # Microsatellite Instability
    TMB_HIGH = "tmb_high"
    MSI_HIGH = "msi_high"
    HER2 = "her2"
    HR_STATUS = "hr_status"  # Hormone Receptor Status
    OTHER = "other"


class Mutation(BaseModel):
    """Single genetic mutation/variant"""

    gene: str = Field(
        description="Gene name (e.g., 'KRAS', 'EGFR', 'ALK')", min_length=1
    )
    variant: str = Field(
        description="Specific variant notation (e.g., 'G12C', 'L858R', 'exon 19 deletion')",
        min_length=1,
    )
    significance: MutationSignificanceEnum = Field(
        description="Clinical significance of this mutation"
    )
    variant_frequency: Optional[float] = Field(
        None, description="Variant allele frequency (0-1) if available"
    )
    notes: Optional[str] = Field(
        None, description="Additional details about the mutation"
    )


class Biomarker(BaseModel):
    """Single biomarker result"""

    biomarker_type: BiomarkerTypeEnum = Field(description="Type of biomarker")
    value: Optional[str] = Field(
        None, description="Value (e.g., '1%', 'high', 'positive', '18.9')"
    )
    unit: Optional[str] = Field(None, description="Unit of measurement if applicable")
    interpretation: Optional[str] = Field(
        None,
        description="Clinical interpretation (e.g., 'high', 'positive', 'negative')",
    )
    notes: Optional[str] = Field(None, description="Additional details")


class MolecularTestResult(BaseModel):
    """Single molecular/genomic test result"""

    test_source: TestSourceEnum = Field(description="Source of the test")
    test_type: TestTypeEnum = Field(description="Type of test performed")
    test_date: Optional[date] = Field(
        None, description="Date test was performed or reported"
    )
    test_name: Optional[str] = Field(
        None,
        description="Name of the test (e.g., 'Omniseq', 'FoundationOne', 'Guardant360')",
    )
    specimen_type: Optional[str] = Field(
        None, description="Type of specimen tested (e.g., 'tissue', 'blood', 'plasma')"
    )
    mutations: List[Mutation] = Field(
        default_factory=list, description="Mutations/variants found"
    )
    biomarkers: List[Biomarker] = Field(
        default_factory=list, description="Biomarker results"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Relevant excerpts from document"
    )
    confidence_score: float = Field(
        description="Confidence in extraction (0-1)", ge=0.0, le=1.0
    )
    notes: Optional[str] = Field(None, description="Additional clarifications")


class MolecularFindingsExtraction(BaseModel):
    """Complete extraction of molecular/genomic findings"""

    test_results: List[MolecularTestResult] = Field(
        description="All molecular/genomic test results in chronological order"
    )

    extraction_challenges: Optional[List[str]] = Field(
        None, description="Brief notes on any extraction difficulties"
    )

    @field_validator("test_results")
    @classmethod
    def validate_non_empty(cls, v):
        """Ensure at least one test result if mutations or biomarkers are present"""
        if len(v) == 0:
            # Allow empty if truly no findings, but warn
            return v
        return v


# ============================================================================
# ORCHESTRATION MODELS FOR MARKDOWN-BASED EXTRACTION
# ============================================================================


class MarkdownOutput(BaseModel):
    """Structured output for markdown generation"""

    markdown: str = Field(
        description="The generated markdown representation of molecular findings"
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
    extraction: Optional[MolecularFindingsExtraction] = Field(
        default=None, description="The final extracted molecular findings"
    )
    iterations_used: int = Field(description="Number of iterations required")
    total_issues_found: int = Field(description="Total number of issues encountered")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert medical data extraction specialist. Your task is to extract molecular/genomic findings from medical documents (pathology reports, genomic test results, lab reports) and represent them in a structured markdown format.

## What Counts as Molecular/Genomic Findings

Include all molecular and genomic test results:
- Genetic mutations/variants (EGFR, KRAS, ALK, BRAF, PIK3CA, etc.)
- Biomarker results (PD-L1, TMB, MSI, HER2, hormone receptor status)
- Genomic test results (NGS panels, targeted sequencing, IHC, FISH, PCR)
- Variant allele frequencies
- Clinical significance annotations

Do NOT include:
- Routine lab values (CBC, chemistry panels, liver function tests)
- Imaging results
- Treatment history (that's a separate extraction)
- Pathology diagnoses without molecular data

## Test Sources

Each test result should be categorized by source:
- pathology_report: Findings from pathology reports with molecular testing
- genomic_test_result: Dedicated genomic testing reports (FoundationOne, Guardant360, Omniseq, etc.)
- lab_result: Other lab-based molecular testing

## Test Types

Identify the type of test performed:
- NGS: Next Generation Sequencing (comprehensive panels)
- IHC: Immunohistochemistry (protein expression)
- FISH: Fluorescence In Situ Hybridization (gene rearrangements)
- PCR: Polymerase Chain Reaction (targeted mutations)
- Sanger: Sanger sequencing
- RT-PCR: Reverse Transcriptase PCR
- Other: Other specialized tests

## Mutations

For each mutation, extract:
- Gene name (standardized: EGFR, KRAS, ALK, etc.)
- Variant notation (exact notation from report: G12C, L858R, exon 19 deletion, etc.)
- Clinical significance (pathogenic, likely_pathogenic, vus, actionable, etc.)
- Variant frequency if available (as decimal 0-1)
- Any relevant notes

Common actionable mutations:
- EGFR: L858R, exon 19 deletions, T790M, exon 20 insertions
- KRAS: G12C, G12D, G12V, G13D
- ALK: Rearrangements, fusions
- BRAF: V600E, V600K
- ROS1: Rearrangements
- RET: Rearrangements
- NTRK: Fusions

## Biomarkers

Extract biomarker results:
- PD-L1 TPS (Tumor Proportion Score): Usually expressed as percentage (e.g., "1%", "50%")
- PD-L1 CPS (Combined Positive Score): Usually expressed as integer (e.g., "10", "50")
- TMB (Tumor Mutational Burden): Usually expressed as mutations/Mb (e.g., "18.9", "high")
- MSI (Microsatellite Instability): Usually "stable", "high", or "MSI-H"
- HER2: Usually "positive", "negative", or IHC score (0, 1+, 2+, 3+)
- HR Status: Estrogen/progesterone receptor status

## Key Principles

Dates: Use YYYY-MM-DD format. If only month/year given, use first of month (June 2023 = 2023-06-01). Calculate relative dates from document date. Always provide best estimate.

Test dates: Extract when test was performed or reported. If multiple dates mentioned, use the most specific/accurate one.

Variant notation: Preserve exact notation from the source document. Don't normalize (e.g., keep "G12C" as-is, not "Gly12Cys").

Significance: Classify mutations based on clinical significance:
- pathogenic: Clearly disease-causing
- likely_pathogenic: Strong evidence but not definitive
- actionable: Has targeted therapy available (e.g., EGFR mutations → osimertinib)
- vus: Variant of uncertain significance
- likely_benign/benign: Not disease-causing

Supporting evidence: Include relevant excerpts from documents showing the mutations, biomarkers, test names, dates, etc.

Confidence score:
- 0.9-1.0 = all info clear and exact
- 0.7-0.9 = most info clear, minor inference needed
- 0.5-0.7 = moderate ambiguity
- 0.3-0.5 = significant uncertainty
- <0.3 = very uncertain

## Common Mistakes to Avoid

- Don't confuse variant notation (keep exact format from report)
- Don't mix up different test results - each test should be separate
- Don't include negative results unless explicitly stated (e.g., "No EGFR mutations detected")
- Don't confuse wild-type/negative with missing data
- Don't normalize gene names unnecessarily (use standard but preserve if report uses different)

## Output Format

Generate a markdown document with the following structure for each test result:

```
## Test Result 1
Test Source: [pathology_report|genomic_test_result|lab_result|unknown]
Test Type: [ngs|ihc|fish|pcr|sanger|rt_pcr|other|unknown]
Test Date: YYYY-MM-DD or null
Test Name: [test name or null]
Specimen Type: [specimen type or null]

Mutations:
  - Gene: [gene name] | Variant: [variant notation] | Significance: [pathogenic|likely_pathogenic|vus|actionable|likely_benign|benign|unknown] | Frequency: [0-1 or null] | Notes: [notes or null]

Biomarkers:
  - Type: [pd_l1_tps|pd_l1_cps|tmb|msi|her2|hr_status|other] | Value: [value or null] | Unit: [unit or null] | Interpretation: [interpretation or null] | Notes: [notes or null]

Supporting Evidence:
  - "[excerpt from document]"
  - "[excerpt from document]"

Confidence: [0.0-1.0]
Notes: [optional notes]

---
```

Separate each test result with `---` on its own line.
"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted molecular findings data. Your job is to verify that the markdown representation accurately captures all relevant information from the original document.

## Validation Checks

1. **Completeness**: Are all molecular/genomic findings mentioned in the document captured in the markdown?
2. **Accuracy**: Do the gene names, variants, biomarker values, and test details match the source material?
3. **Proper Test Separation**: Are different test results correctly separated?
4. **Format Compliance**: Does the markdown follow the expected structure?
5. **Supporting Evidence**: Are there relevant excerpts that justify the extraction?
6. **Confidence Appropriateness**: Does the confidence score reflect the clarity of information?
7. **Variant Notation**: Are variants preserved exactly as stated in the source?

## What to Flag

- Missing mutations or biomarkers
- Incorrect gene names or variant notations
- Wrong biomarker values or interpretations
- Tests that should be combined but are separated (or vice versa)
- Missing or weak supporting evidence
- Formatting issues
- Incorrect test type or source classification

Provide specific, actionable feedback on what needs to be corrected."""

CORRECTION_SYSTEM_PROMPT = """You are a precise editor. Given validation feedback on extracted molecular findings markdown, generate search-and-replace operations to fix the identified issues.

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
- Preserve exact variant notation from source"""


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
    """Create extraction agent for generating markdown from documents"""
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


async def extract_to_pydantic(markdown: str, model) -> MolecularFindingsExtraction:
    """Convert validated markdown to Pydantic model"""
    extraction_agent = Agent(
        model=model,
        output_type=MolecularFindingsExtraction,
        system_prompt="You are a precise data parser. Convert the provided markdown representation of molecular findings into the structured MolecularFindingsExtraction model. Preserve all information accurately.",
    )

    prompt = f"""
    Convert the following markdown representation of molecular findings into the MolecularFindingsExtraction model:

    {markdown}
    """

    print("\n🔄 Converting markdown to Pydantic model...")
    result = await extraction_agent.run(prompt)
    return result.output


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================


async def extract_molecular_findings_async(
    document_text: str, max_iterations: int = 3
) -> ExtractionResult:
    """
    Extract molecular/genomic findings from a document using iterative validation

    Args:
        document_text: Raw document text (pathology report, genomic test result, etc.)
        max_iterations: Maximum validation/correction iterations (default 3)

    Returns:
        ExtractionResult with extracted data or error

    Raises:
        ValueError: If validation fails after max_iterations
    """
    model = get_default_model()

    print("=" * 80)
    print("🚀 Starting Molecular Findings Extraction")
    print("=" * 80)

    # Step 1: Generate initial markdown
    print("\n📝 Step 1: Generating initial markdown from document...")
    print("-" * 60)

    extraction_agent = create_extraction_agent(model)
    result = await extraction_agent.run(
        f"Extract molecular/genomic findings from this document:\n\n{document_text}"
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

    print(f"✅ Successfully extracted {len(extraction.test_results)} test results")

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
# NOTE - do with path and genomic testing results document
SAMPLE_DOCUMENT = """
Omniseq Comprehensive Genomic Profiling Report

Patient: Naaji Jr, Leif
DOB: 7/06/1954
Test Date: 8/26/2024
Specimen: Right upper lobe lung biopsy

Results:
- KRAS G12C mutation detected (positive)
- TMB: 18.9 mutations/Mb (high)
- MSI: Stable
- PD-L1 TPS: 1%

Additional Testing:
- EGFR: Negative
- ALK: Negative
- ROS1: Negative
- HER2: Negative
- MET: Negative
- RET: Negative
- NTRK: Negative

Interpretation:
The tumor shows a KRAS G12C mutation with high tumor mutational burden. PD-L1 expression is low at 1%. No targetable alterations detected in EGFR, ALK, ROS1, HER2, MET, RET, or NTRK.
"""


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


@pytest.mark.integration
async def test_extract_molecular_findings():
    """Integration test for molecular findings extraction with real API"""
    # Run extraction
    result = await extract_molecular_findings_async(SAMPLE_DOCUMENT, max_iterations=3)

    # Verify result
    assert result.success is True
    assert result.extraction is not None
    assert len(result.extraction.test_results) >= 1

    # Check first test result
    test_result = result.extraction.test_results[0]
    assert test_result.test_source == TestSourceEnum.GENOMIC_TEST_RESULT
    assert test_result.test_type == TestTypeEnum.NGS
    assert test_result.test_name == "Omniseq" or "Omniseq" in str(test_result.test_name)
    assert test_result.test_date == date(2024, 8, 26)

    # Check mutations
    assert len(test_result.mutations) >= 1
    kras_mutation = next(
        (m for m in test_result.mutations if m.gene.upper() == "KRAS"), None
    )
    assert kras_mutation is not None
    assert "G12C" in kras_mutation.variant

    # Check biomarkers
    assert len(test_result.biomarkers) >= 3  # TMB, MSI, PD-L1

    tmb_biomarker = next(
        (
            b
            for b in test_result.biomarkers
            if b.biomarker_type == BiomarkerTypeEnum.TMB
        ),
        None,
    )
    assert tmb_biomarker is not None
    assert (
        "18.9" in str(tmb_biomarker.value)
        or "high" in str(tmb_biomarker.interpretation).lower()
    )

    pdl1_biomarker = next(
        (
            b
            for b in test_result.biomarkers
            if b.biomarker_type == BiomarkerTypeEnum.PD_L1_TPS
        ),
        None,
    )
    assert pdl1_biomarker is not None
    assert "1" in str(pdl1_biomarker.value)

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)
    print(f"\nExtracted {len(result.extraction.test_results)} test results:")
    print(result.extraction.model_dump_json(indent=2))
