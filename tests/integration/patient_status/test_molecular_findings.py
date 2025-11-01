# What am I trying to do here?
# - I want to be able to extract molecular/genomic findings from document context (pathology reports, genomic test results)

# The reason i want to extract this data is so i can do the following types of queries
# - patient has KRAS G12D mutation (somatic or germline)
# - patient has HER2 3+
# - patient has MTAP deletion
# - group patients by TMB status or MSI status
# - find patients with specific biomarker profiles

# Key insight: Use discriminated union pattern
# - Base class with shared fields
# - Four specific finding types: VariantFinding, CNAFinding, ExpressionFinding, SignatureFinding
# - This makes it structurally impossible to put wrong data in wrong fields
# - Makes queries precise and extraction instructions clear

# Normalization strategy:
# - Enums for truly limited options (Origin, FindingType)
# - Strings with standardization for variable fields (canonical_variant, gene, etc.)
# - Tell the model to normalize to canonical forms

# Mutation constraints:
# - Only extract pathogenic or likely_pathogenic mutations (skip VUS)
# - Max 15 mutations per report (prioritize most clinically relevant)


from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union, Literal
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
# ENUMS - Only for truly limited options
# ============================================================================


class Origin(str, Enum):
    """Biological origin of the finding"""

    SOMATIC = "somatic"
    GERMLINE = "germline"
    UNKNOWN = "unknown"


class FindingType(str, Enum):
    """Type of molecular finding"""

    VARIANT = "variant"
    CNA = "cna"
    EXPRESSION = "expression"
    SIGNATURE = "signature"


# ============================================================================
# BASE MODEL - Shared fields across all finding types
# ============================================================================


class MolecularFindingBase(BaseModel):
    """Base class for all molecular findings"""

    finding_type: FindingType = Field(description="Type of finding (discriminator)")
    origin: Origin = Field(description="Somatic vs germline vs unknown")
    test_method: str = Field(
        description="Test method (NGS, IHC, FISH, PCR, etc.)", min_length=1
    )
    test_date: Optional[date] = Field(
        None, description="Date test was performed or reported"
    )
    test_name: Optional[str] = Field(
        None, description="Name of test (e.g., 'Omniseq', 'FoundationOne')"
    )
    raw_text: str = Field(
        description="Raw text from report showing this finding", min_length=1
    )
    clinical_significance: Optional[str] = Field(
        None, description="Clinical significance or interpretation"
    )


# ============================================================================
# SPECIFIC FINDING TYPES
# ============================================================================


class VariantFinding(MolecularFindingBase):
    """Genetic mutation/variant (SNV, indel, etc.)"""

    finding_type: Literal[FindingType.VARIANT] = FindingType.VARIANT

    gene: str = Field(
        description="Gene name using HUGO nomenclature (e.g., 'KRAS', 'EGFR')",
        min_length=1,
    )
    canonical_variant: str = Field(
        description="Standardized variant notation (e.g., 'G12D', 'L858R', 'exon 19 deletion')",
        min_length=1,
    )
    protein_change: Optional[str] = Field(
        None, description="Protein change notation if available (e.g., 'p.G12D')"
    )
    cdna_change: Optional[str] = Field(
        None, description="cDNA change notation if available (e.g., 'c.35G>A')"
    )
    variant_frequency: Optional[float] = Field(
        None, description="Variant allele frequency (0-1) if available", ge=0.0, le=1.0
    )
    notes: Optional[str] = Field(None, description="Additional details")


class CNAFinding(MolecularFindingBase):
    """Copy number alteration"""

    finding_type: Literal[FindingType.CNA] = FindingType.CNA

    gene: str = Field(
        description="Gene name using HUGO nomenclature (e.g., 'MTAP', 'MET')",
        min_length=1,
    )
    alteration_direction: str = Field(
        description="Direction of alteration: 'amplification', 'deletion', 'gain', 'loss', 'loss_of_heterozygosity'",
        min_length=1,
    )
    copy_number: Optional[float] = Field(None, description="Copy number if quantified")
    notes: Optional[str] = Field(None, description="Additional details")


class ExpressionFinding(MolecularFindingBase):
    """Biomarker expression level"""

    finding_type: Literal[FindingType.EXPRESSION] = FindingType.EXPRESSION

    biomarker: str = Field(
        description="Biomarker name (e.g., 'HER2', 'PD-L1', 'ER', 'PR')", min_length=1
    )
    intensity_score: str = Field(
        description="Expression level (e.g., '3+', '2+', '50%', 'positive', 'negative')",
        min_length=1,
    )
    score_scale: Optional[str] = Field(
        None, description="Scale used (e.g., 'IHC 0-3+', 'TPS', 'CPS')"
    )
    quantitative_value: Optional[float] = Field(
        None, description="Numeric value if applicable (e.g., 50 for 50% TPS)"
    )
    notes: Optional[str] = Field(None, description="Additional details")


class SignatureFinding(MolecularFindingBase):
    """Genomic signature or global biomarker"""

    finding_type: Literal[FindingType.SIGNATURE] = FindingType.SIGNATURE

    signature_type: str = Field(
        description="Type of signature (e.g., 'MSI', 'TMB', 'dMMR', 'HRD', 'COSMIC-SBS1')",
        min_length=1,
    )
    status: str = Field(
        description="Status or interpretation (e.g., 'MSI-High', 'stable', 'deficient')",
        min_length=1,
    )
    quantitative_value: Optional[float] = Field(
        None, description="Numeric value if applicable (e.g., 18.9 for TMB)"
    )
    unit: Optional[str] = Field(
        None, description="Unit of measurement (e.g., 'mutations/Mb')"
    )
    notes: Optional[str] = Field(None, description="Additional details")


# Union type for discriminated parsing
MolecularFinding = Union[
    VariantFinding, CNAFinding, ExpressionFinding, SignatureFinding
]


# ============================================================================
# EXTRACTION RESULT MODEL
# ============================================================================


class MolecularFindingsExtraction(BaseModel):
    """Complete extraction of molecular/genomic findings"""

    findings: List[MolecularFinding] = Field(
        description="All molecular/genomic findings extracted from the document"
    )

    extraction_challenges: Optional[List[str]] = Field(
        None, description="Brief notes on any extraction difficulties"
    )

    @field_validator("findings")
    @classmethod
    def validate_variant_limit(cls, v):
        """Ensure max 15 variants"""
        variants = [f for f in v if isinstance(f, VariantFinding)]
        if len(variants) > 15:
            raise ValueError(
                f"Too many variants extracted ({len(variants)}). Maximum is 15. Prioritize most clinically relevant."
            )
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

EXTRACTION_SYSTEM_PROMPT = """You are an expert medical data extraction specialist. Your task is to extract molecular/genomic findings from medical documents (pathology reports, genomic test results) and represent them in a structured markdown format.

## Finding Types and Structure

You must classify each finding into ONE of four types:

1. **VARIANT** - Genetic mutations/variants
2. **CNA** - Copy number alterations
3. **EXPRESSION** - Biomarker expression levels
4. **SIGNATURE** - Genomic signatures (MSI, TMB, etc.)

Each finding type has DIFFERENT fields - you cannot put intensity_score in a variant or canonical_variant in an expression finding.

## VARIANT Findings (Mutations)

**CRITICAL CONSTRAINTS**:
- ONLY extract pathogenic or likely pathogenic mutations
- SKIP variants of uncertain significance (VUS)
- SKIP benign or likely benign variants
- MAXIMUM 15 variants per document
- If more than 15, prioritize by clinical actionability and significance

Extract:
- Gene (HUGO nomenclature): "KRAS", "EGFR", "BRAF"
- Canonical variant (protein-level): "G12D", "L858R", "V600E", "exon 19 deletion"
- Protein change (optional): "p.G12D"
- cDNA change (optional): "c.35G>A"
- Variant frequency (optional): 0.42 (as decimal)
- Origin: somatic/germline/unknown

**Normalization**:
- Standardize gene names to HUGO symbols
- Use protein-level notation for canonical_variant ("G12D" not "Gly12Asp")
- If report says "mutation detected" without frequency, that's fine - leave null

**Common actionable mutations to prioritize**:
- EGFR: L858R, exon 19 deletions, T790M, C797S, exon 20 insertions
- KRAS: G12C, G12D, G12V, G13D
- BRAF: V600E, V600K
- ALK: rearrangements, fusions (extract as "ALK rearrangement")
- ROS1: rearrangements
- RET: rearrangements
- NTRK: fusions
- MET: exon 14 skipping
- PIK3CA: H1047R, E545K
- BRCA1/BRCA2: pathogenic mutations (usually germline)

## CNA Findings (Copy Number Alterations)

Extract:
- Gene: "MTAP", "MET", "EGFR", "FGFR1"
- Alteration direction: "amplification", "deletion", "gain", "loss", "loss_of_heterozygosity"
- Copy number (optional): 8.2
- Origin: somatic/germline/unknown

**Normalization**:
- Use canonical terms: "amplification" (not "amp"), "deletion" (not "del")
- Common CNAs: EGFR amplification, MET amplification, MTAP deletion, CDKN2A deletion

## EXPRESSION Findings (Biomarker Expression)

Extract:
- Biomarker: "HER2", "PD-L1", "ER", "PR", "ALK", "ROS1"
- Intensity score: "3+", "2+", "1+", "0", "50%", "positive", "negative"
- Score scale (optional): "IHC 0-3+", "TPS", "CPS"
- Quantitative value (optional): 50.0 (for "50%")
- Origin: somatic/germline/unknown (usually somatic for expression)

**Normalization**:
- Standardize biomarker names: "HER2" (not "Her-2" or "HER-2")
- Preserve exact score as stated: "3+", "50%", "positive"
- For PD-L1, note if it's TPS (Tumor Proportion Score) or CPS (Combined Positive Score) in score_scale

## SIGNATURE Findings (Genomic Signatures)

Extract:
- Signature type: "MSI", "TMB", "dMMR", "HRD", "COSMIC-SBS1", etc.
- Status: "MSI-High", "MSI-Low", "stable", "high", "deficient", "proficient"
- Quantitative value (optional): 18.9
- Unit (optional): "mutations/Mb"
- Origin: somatic/germline/unknown (usually somatic)

**Normalization**:
- Use standard abbreviations: "MSI" (not "microsatellite instability")
- Preserve status as stated: "MSI-High", "stable", "high"
- For TMB, extract numeric value if given

## Shared Fields (All Finding Types)

Every finding must have:
- **finding_type**: variant/cna/expression/signature (determines which fields are valid)
- **origin**: somatic/germline/unknown
- **test_method**: "NGS", "IHC", "FISH", "PCR", "Sanger", "RT-PCR", "other"
- **test_date**: YYYY-MM-DD (or null if not stated)
- **test_name**: "Omniseq", "FoundationOne", "Guardant360" (or null)
- **raw_text**: Exact excerpt from document showing this finding
- **clinical_significance**: "pathogenic", "actionable", "resistance", etc. (or null)

## Negative Results

If report explicitly states "No mutations detected" or "EGFR: negative" or "ALK: negative", you can SKIP these. Only extract POSITIVE findings.

Exception: If explicitly asked to extract negative results, do so, but prioritize positive findings if constrained by variant limit.

## Output Format

Generate markdown with this structure for each finding:

```
## Finding 1: [VARIANT|CNA|EXPRESSION|SIGNATURE]
Finding Type: [variant|cna|expression|signature]
Origin: [somatic|germline|unknown]
Test Method: [test method]
Test Date: YYYY-MM-DD or null
Test Name: [test name or null]

[TYPE-SPECIFIC FIELDS]

For VARIANT:
  Gene: [gene]
  Canonical Variant: [variant]
  Protein Change: [protein change or null]
  cDNA Change: [cdna change or null]
  Variant Frequency: [0-1 or null]

For CNA:
  Gene: [gene]
  Alteration Direction: [alteration direction]
  Copy Number: [copy number or null]

For EXPRESSION:
  Biomarker: [biomarker]
  Intensity Score: [score]
  Score Scale: [scale or null]
  Quantitative Value: [numeric value or null]

For SIGNATURE:
  Signature Type: [signature type]
  Status: [status]
  Quantitative Value: [value or null]
  Unit: [unit or null]

Raw Text: "[excerpt from document]"
Clinical Significance: [significance or null]
Notes: [notes or null]

---
```

Separate each finding with `---` on its own line.

## Key Principles

1. **Discriminate correctly** - Each finding is ONE type only
2. **Normalize to canonical forms** - HUGO genes, protein-level variants, standard abbreviations
3. **Prioritize clinical relevance** - For variants, actionable > pathogenic > likely pathogenic
4. **Respect the 15 variant limit** - If genomic report has 50 mutations, extract top 15 most relevant
5. **Skip VUS** - Do not extract variants of uncertain significance
6. **Preserve exact values** - Don't round or modify scores/percentages
7. **Extract from raw text** - Pull exact excerpts showing each finding
"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted molecular findings data. Your job is to verify that the markdown representation accurately captures all relevant information from the original document.

## Validation Checks

1. **Completeness**: Are all clinically significant findings captured?
2. **Correct Classification**: Is each finding in the right category (variant/cna/expression/signature)?
3. **Variant Constraints**:
   - Are there more than 15 variants? (FAIL if yes)
   - Are VUS mutations included? (FAIL if yes)
   - Are only pathogenic/likely pathogenic variants included? (PASS if yes)
4. **Normalization**: Are genes, variants, biomarkers using standardized nomenclature?
5. **Field Accuracy**: Do the type-specific fields match the finding type?
6. **Format Compliance**: Does the markdown follow the expected structure?
7. **Supporting Evidence**: Is raw_text present and relevant?

## What to Flag

- VUS mutations included (should be excluded)
- More than 15 variants
- Benign variants included
- Wrong finding type classification (e.g., HER2 IHC classified as variant instead of expression)
- Non-standard gene names (e.g., "Her-2" instead of "HER2")
- Missing finding_type or origin
- Missing raw_text
- Fields that don't belong to that finding type

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
- When removing VUS mutations, remove the entire finding block including the `---` separator

## Common Corrections

- Remove VUS mutations: Find the entire "## Finding X: VARIANT" block through the `---` and replace with empty string
- Fix gene names: "Her-2" → "HER2"
- Fix finding type: "Finding Type: variant" → "Finding Type: expression" (for IHC results)
- Remove excess variants: Remove lowest priority variants to get to 15 max
"""


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
        system_prompt="You are a precise data parser. Convert the provided markdown representation of molecular findings into the structured MolecularFindingsExtraction model. Use the finding_type field to determine which specific model to use (VariantFinding, CNAFinding, ExpressionFinding, or SignatureFinding). Preserve all information accurately.",
    )

    prompt = f"""
    Convert the following markdown representation of molecular findings into the MolecularFindingsExtraction model.
    
    Each finding will be one of four types based on the finding_type field:
    - variant → VariantFinding (with gene, canonical_variant, etc.)
    - cna → CNAFinding (with gene, alteration_direction, etc.)
    - expression → ExpressionFinding (with biomarker, intensity_score, etc.)
    - signature → SignatureFinding (with signature_type, status, etc.)
    
    Use the discriminated union pattern to parse each finding into the correct type.

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

    print(f"✅ Successfully extracted {len(extraction.findings)} findings")

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
Omniseq Comprehensive Genomic Profiling Report

Patient: Naaji Jr, Leif
DOB: 7/06/1954
Test Date: 8/26/2024
Specimen: Right upper lobe lung biopsy (tissue)

RESULTS:

Somatic Variants Detected:
- KRAS c.35G>A (p.G12D) - Pathogenic, VAF: 42%
- TP53 c.524G>A (p.R175H) - Pathogenic, VAF: 38%

Somatic Copy Number Alterations:
- MTAP deletion detected
- CDKN2A homozygous deletion

Tumor Mutational Burden (TMB):
- 18.9 mutations/Mb (High)

Microsatellite Instability (MSI):
- MSI-Stable

PD-L1 Expression (IHC, 22C3 antibody):
- Tumor Proportion Score (TPS): 1%
- Interpretation: Low expression

Additional Testing:
- EGFR: No mutations detected
- ALK: Negative by IHC
- ROS1: Negative by IHC
- HER2: Not amplified
- MET: No exon 14 skipping mutations
- RET: No rearrangements detected
- NTRK: No fusions detected

INTERPRETATION:
The tumor harbors a KRAS G12D mutation (actionable with KRAS G12C inhibitors in appropriate context) and TP53 R175H mutation. High tumor mutational burden (18.9 mutations/Mb) suggests potential benefit from immunotherapy. PD-L1 expression is low at 1% TPS. MTAP and CDKN2A deletions noted. No targetable alterations detected in EGFR, ALK, ROS1, HER2, MET, RET, or NTRK.
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
    assert len(result.extraction.findings) >= 5

    # Separate findings by type
    variants = [f for f in result.extraction.findings if isinstance(f, VariantFinding)]
    cnas = [f for f in result.extraction.findings if isinstance(f, CNAFinding)]
    expressions = [
        f for f in result.extraction.findings if isinstance(f, ExpressionFinding)
    ]
    signatures = [
        f for f in result.extraction.findings if isinstance(f, SignatureFinding)
    ]

    # Check variant constraint
    assert len(variants) <= 15, "Should have max 15 variants"

    # Check KRAS variant
    kras = next((v for v in variants if v.gene.upper() == "KRAS"), None)
    assert kras is not None
    assert "G12D" in kras.canonical_variant or "G12C" in kras.canonical_variant
    assert kras.origin == Origin.SOMATIC

    # Check TP53 variant
    tp53 = next((v for v in variants if v.gene.upper() == "TP53"), None)
    assert tp53 is not None

    # Check CNAs
    assert len(cnas) >= 2  # MTAP deletion, CDKN2A deletion
    mtap = next((c for c in cnas if c.gene.upper() == "MTAP"), None)
    assert mtap is not None
    assert "deletion" in mtap.alteration_direction.lower()

    # Check PD-L1 expression
    assert len(expressions) >= 1
    pdl1 = next(
        (
            e
            for e in expressions
            if "PD-L1" in e.biomarker.upper() or "PDL1" in e.biomarker.upper()
        ),
        None,
    )
    assert pdl1 is not None
    assert "1" in pdl1.intensity_score or "low" in pdl1.intensity_score.lower()

    # Check signatures
    assert len(signatures) >= 2  # TMB and MSI
    tmb = next((s for s in signatures if "TMB" in s.signature_type.upper()), None)
    assert tmb is not None
    assert "high" in tmb.status.lower() or tmb.quantitative_value is not None

    msi = next((s for s in signatures if "MSI" in s.signature_type.upper()), None)
    assert msi is not None

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)
    print(f"\nExtracted {len(result.extraction.findings)} findings:")
    print(f"  - {len(variants)} variants")
    print(f"  - {len(cnas)} CNAs")
    print(f"  - {len(expressions)} expression findings")
    print(f"  - {len(signatures)} signature findings")
    print("\nFull extraction:")
    print(result.extraction.model_dump_json(indent=2))
