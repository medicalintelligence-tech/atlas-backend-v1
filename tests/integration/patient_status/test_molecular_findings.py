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


from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Optional, List, Union, Literal
from datetime import date, datetime
from enum import Enum
from dataclasses import dataclass, field as dataclass_field, asdict
from pydantic_ai import Agent, RunContext, PromptedOutput
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
    FUSION = "fusion"
    IHC = "ihc"
    FISH = "fish"
    SIGNATURE = "signature"
    WILDTYPE = "wildtype"


class CNADirection(str, Enum):
    """Direction of copy number alteration"""

    AMPLIFICATION = "amplification"
    DELETION = "deletion"
    GAIN = "gain"
    LOSS = "loss"
    LOSS_OF_HETEROZYGOSITY = "loss_of_heterozygosity"


class FusionType(str, Enum):
    """Type of gene fusion"""

    IN_FRAME = "in_frame"
    OUT_OF_FRAME = "out_of_frame"
    REARRANGEMENT = "rearrangement"  # When frame unknown
    UNKNOWN = "unknown"


class StainingIntensity(str, Enum):
    """Staining intensity for IHC"""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class PDL1ScoreType(str, Enum):
    """PD-L1 scoring systems"""

    TPS = "TPS"  # Tumor Proportion Score
    CPS = "CPS"  # Combined Positive Score
    IC = "IC"  # Immune Cell score
    TC = "TC"  # Tumor Cell score


class SignatureType(str, Enum):
    """Types of genomic signatures"""

    MSI = "MSI"
    TMB = "TMB"
    MMR = "MMR"
    DMMR = "dMMR"
    HRD = "HRD"


class MSIResult(str, Enum):
    """MSI status results"""

    MSI_HIGH = "MSI-High"
    MSI_LOW = "MSI-Low"
    MSS = "MSS"  # Microsatellite stable


class MMRResult(str, Enum):
    """MMR status results"""

    PROFICIENT = "proficient"
    DEFICIENT = "deficient"
    INTACT = "intact"
    LOST = "lost"


class AlterationTypeTested(str, Enum):
    """Types of alterations tested for wildtype status"""

    MUTATIONS = "mutations"
    AMPLIFICATION = "amplification"
    FUSION = "fusion"
    ANY_ALTERATION = "any alteration"


class Zygosity(str, Enum):
    """Zygosity of copy number alterations"""

    HOMOZYGOUS = "homozygous"
    HETEROZYGOUS = "heterozygous"
    HEMIZYGOUS = "hemizygous"


# ============================================================================
# BASE MODEL - Shared fields across all finding types
# ============================================================================


class MolecularFindingBase(BaseModel):
    """Base class for all molecular findings"""

    model_config = ConfigDict(extra="forbid")

    finding_type: FindingType = Field(description="Type of finding (discriminator)")
    raw_text: str = Field(
        description="Raw text from report showing this finding", min_length=1
    )
    notes: Optional[str] = Field(
        None,
        description="Additional context or caveats",
    )


# ============================================================================
# SPECIFIC FINDING TYPES
# ============================================================================


class VariantFinding(MolecularFindingBase):
    """Genetic mutation/variant (SNV, indel, splice variant)"""

    finding_type: Literal[FindingType.VARIANT] = FindingType.VARIANT

    gene: str = Field(
        description="Gene name using HUGO nomenclature (e.g., 'KRAS', 'EGFR', 'TP53')",
        min_length=1,
    )
    canonical_variant: str = Field(
        description="Standardized variant notation (e.g., 'G12D', 'L858R', 'E746_A750del', 'exon 19 deletion', 'exon 14 skipping')",
        min_length=1,
    )
    origin: Origin = Field(
        default=Origin.UNKNOWN, description="Somatic vs germline vs unknown"
    )
    protein_change: Optional[str] = Field(
        None, description="Protein change notation if available (e.g., 'p.G12D')"
    )
    cdna_change: Optional[str] = Field(
        None, description="cDNA change notation if available (e.g., 'c.35G>A')"
    )
    exon: Optional[str] = Field(
        None, description="Exon number or region if specified (e.g., '19', '14')"
    )
    variant_frequency: Optional[float] = Field(
        None, description="Variant allele frequency (0-1) if available", ge=0.0, le=1.0
    )
    is_hotspot: Optional[bool] = Field(
        None, description="Whether this is a known hotspot mutation"
    )


class CNAFinding(MolecularFindingBase):
    """Copy number alteration"""

    finding_type: Literal[FindingType.CNA] = FindingType.CNA

    gene: str = Field(
        description="Gene name using HUGO nomenclature (e.g., 'ERBB2', 'MET', 'CDKN2A')",
        min_length=1,
    )
    alteration_direction: CNADirection = Field(
        description="Direction of copy number change"
    )
    origin: Origin = Field(
        default=Origin.UNKNOWN, description="Somatic vs germline vs unknown"
    )
    copy_number: Optional[float] = Field(
        None, description="Absolute copy number if quantified (e.g., 8.2)"
    )
    fold_change: Optional[float] = Field(
        None, description="Fold change or ratio if reported (e.g., 3.5x)"
    )
    is_focal: Optional[bool] = Field(
        None,
        description="Whether this is focal (gene-level) vs broad (chromosomal region)",
    )
    zygosity: Optional[Zygosity] = Field(
        None,
        description="Zygosity of the alteration: homozygous (both alleles affected), heterozygous (one allele affected), hemizygous (single copy, typically X chromosome in males)",
    )


class FusionFinding(MolecularFindingBase):
    """Gene fusion or structural rearrangement"""

    finding_type: Literal[FindingType.FUSION] = FindingType.FUSION

    gene_5prime: str = Field(
        description="5' fusion partner gene (e.g., 'EML4' in EML4-ALK)",
        min_length=1,
    )
    gene_3prime: str = Field(
        description="3' fusion partner gene (e.g., 'ALK' in EML4-ALK)",
        min_length=1,
    )
    origin: Origin = Field(
        default=Origin.SOMATIC,
        description="Almost always somatic; germline fusions are extremely rare",
    )
    fusion_type: FusionType = Field(
        default=FusionType.UNKNOWN, description="Whether fusion is in-frame"
    )
    exon_5prime: Optional[str] = Field(
        None, description="Exon from 5' partner if specified (e.g., 'exon 6')"
    )
    exon_3prime: Optional[str] = Field(
        None, description="Exon from 3' partner if specified (e.g., 'exon 20')"
    )
    variant_frequency: Optional[float] = Field(
        None,
        description="Variant allele frequency or fusion read percentage if available",
        ge=0.0,
        le=1.0,
    )


class IHCFinding(MolecularFindingBase):
    """Immunohistochemistry protein expression"""

    finding_type: Literal[FindingType.IHC] = FindingType.IHC

    biomarker: str = Field(
        description="Biomarker/protein tested (e.g., 'HER2', 'ER', 'PR', 'PD-L1', 'MLH1', 'MSH2', 'AR')",
        min_length=1,
    )

    # For HER2-style scoring (0, 1+, 2+, 3+)
    intensity_score: Optional[int] = Field(
        None,
        description="Intensity score (0-3, where 0=negative, 1=1+, 2=2+, 3=3+)",
        ge=0,
        le=3,
    )

    # For percentage-based results (ER, PR, PD-L1)
    percentage_positive: Optional[float] = Field(
        None,
        description="Percentage of positive cells (0-100)",
        ge=0.0,
        le=100.0,
    )

    # For binary results (MMR proteins, etc.)
    is_positive: Optional[bool] = Field(
        None,
        description="Binary positive/negative or intact/lost",
    )

    # Staining intensity for ER/PR
    staining_intensity: Optional[StainingIntensity] = Field(
        None,
        description="Staining intensity if reported",
    )

    # Composite scores
    h_score: Optional[int] = Field(
        None,
        description="H-score if reported (0-300)",
        ge=0,
        le=300,
    )
    allred_score: Optional[int] = Field(
        None,
        description="Allred score if reported (0-8)",
        ge=0,
        le=8,
    )

    # PD-L1 specific
    score_type: Optional[PDL1ScoreType] = Field(
        None,
        description="Scoring system for PD-L1",
    )

    # Assay details
    assay: Optional[str] = Field(
        None,
        description="Specific assay/antibody used (e.g., '22C3', 'SP263', 'HercepTest', '4B5')",
    )


class FISHFinding(MolecularFindingBase):
    """Fluorescence in situ hybridization"""

    finding_type: Literal[FindingType.FISH] = FindingType.FISH

    target: str = Field(
        description="Target gene/locus (e.g., 'HER2', 'ALK', 'EGFR', 'MET')",
        min_length=1,
    )

    # For amplification detection (HER2, MET, etc.)
    ratio: Optional[float] = Field(
        None,
        description="Signal ratio if reported (e.g., HER2/CEP17 = 2.5)",
    )
    copy_number: Optional[float] = Field(
        None,
        description="Average copy number per cell if reported",
    )
    is_amplified: Optional[bool] = Field(
        None,
        description="Whether amplification detected",
    )

    # For rearrangement detection (ALK, ROS1, etc.)
    is_rearranged: Optional[bool] = Field(
        None,
        description="Whether rearrangement/break-apart detected",
    )
    percentage_positive: Optional[float] = Field(
        None,
        description="Percentage of cells with positive signal (0-100)",
        ge=0.0,
        le=100.0,
    )

    # Reference probe
    reference_probe: Optional[str] = Field(
        None,
        description="Reference probe used (e.g., 'CEP17' for HER2 FISH)",
    )


class SignatureFinding(MolecularFindingBase):
    """Genomic signature or global biomarker"""

    finding_type: Literal[FindingType.SIGNATURE] = FindingType.SIGNATURE

    signature_type: SignatureType = Field(description="Type of signature")
    result: str = Field(
        description="Result as reported (e.g., 'MSI-High', 'MSS', 'stable', 'deficient', 'proficient')",
        min_length=1,
    )

    # Standardized results based on signature type
    msi_result: Optional[MSIResult] = Field(
        None, description="Standardized MSI result if signature_type is MSI"
    )
    mmr_result: Optional[MMRResult] = Field(
        None, description="Standardized MMR result if signature_type is MMR or dMMR"
    )

    quantitative_value: Optional[float] = Field(
        None, description="Numeric value if applicable (e.g., 18.9 for TMB)"
    )
    unit: Optional[str] = Field(
        None, description="Unit of measurement (e.g., 'mutations/Mb', 'muts/Mb')"
    )


class WildtypeFinding(MolecularFindingBase):
    """Wildtype/not detected status - when absence of alteration is clinically relevant"""

    finding_type: Literal[FindingType.WILDTYPE] = FindingType.WILDTYPE

    gene: str = Field(
        description="Gene confirmed as wildtype (e.g., 'KRAS', 'BRAF', 'EGFR')",
        min_length=1,
    )
    origin: Origin = Field(
        default=Origin.SOMATIC,
        description="Usually checking for somatic alterations",
    )
    alteration_type_tested: Optional[AlterationTypeTested] = Field(
        None,
        description="What was tested for",
    )


# ============================================================================
# UNION TYPE
# ============================================================================

MolecularFinding = Union[
    VariantFinding,
    CNAFinding,
    FusionFinding,
    IHCFinding,
    FISHFinding,
    SignatureFinding,
    WildtypeFinding,
]


# ============================================================================
# TEST REPORT MODEL
# ============================================================================


class TestReport(BaseModel):
    """A molecular/genomic test report with its findings"""

    model_config = ConfigDict(extra="forbid")

    test_name: Optional[str] = Field(
        None,
        description="Name of test (e.g., 'Omniseq', 'FoundationOne CDx', 'Guardant360', 'Tempus xT')",
    )
    test_methods: List[str] = Field(
        description="Test methods used in this report (e.g., ['NGS', 'IHC', 'FISH', 'PCR'])",
        min_length=1,
    )
    specimen_type: Optional[str] = Field(
        None, description="Type of specimen tested (e.g., 'tissue', 'blood', 'plasma')"
    )
    specimen_site: Optional[str] = Field(
        None,
        description="Anatomical site/source of specimen (e.g., 'lung', 'liver metastasis', 'primary tumor', 'lymph node')",
    )
    tumor_content: Optional[float] = Field(
        None,
        description="Tumor content/cellularity as percentage (0-100) if reported",
        ge=0.0,
        le=100.0,
    )
    findings: List[MolecularFinding] = Field(
        description="All molecular findings from this report"
    )

    @field_validator("findings")
    @classmethod
    def validate_variant_limit(cls, v):
        """Ensure max 15 variants per report"""
        variants = [f for f in v if isinstance(f, VariantFinding)]
        if len(variants) > 15:
            raise ValueError(
                f"Too many variants in this report ({len(variants)}). Maximum is 15. Prioritize most clinically relevant."
            )
        return v


# ============================================================================
# EXTRACTION RESULT MODEL
# ============================================================================


class MolecularFindingsExtraction(BaseModel):
    """Complete extraction of molecular/genomic findings from a document"""

    model_config = ConfigDict(extra="forbid")

    reports: List[TestReport] = Field(
        description="All test reports extracted from the document (usually 1, but can be multiple if document references multiple historical tests)"
    )
    extraction_challenges: Optional[List[str]] = Field(
        None, description="Brief notes on any extraction difficulties"
    )


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

You must classify each finding into ONE of seven types:

1. **VARIANT** - Genetic mutations/variants (SNV, indel, splice variants)
2. **CNA** - Copy number alterations (amplifications, deletions, gains, losses)
3. **FUSION** - Gene fusions or structural rearrangements
4. **IHC** - Immunohistochemistry protein expression
5. **FISH** - Fluorescence in situ hybridization
6. **SIGNATURE** - Genomic signatures (MSI, TMB, MMR, HRD, etc.)
7. **WILDTYPE** - Confirmed wildtype/not detected status when clinically relevant

Each finding type has DIFFERENT fields - you cannot put intensity_score in a variant or canonical_variant in an IHC finding.

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
- Gene: Gene name or chromosomal region
  - If specific gene mentioned: Use HUGO nomenclature ("MTAP", "MET", "EGFR")
  - If only chromosomal region mentioned: Use band notation ("11q13-q23", "17p", "5q")
  - Add to notes: "Region-level CNA" or "Gene-specific CNA" or "From cytogenetics"
- Alteration direction: "amplification", "deletion", "gain", "loss", "loss_of_heterozygosity"
- Copy number (optional): 8.2
- Fold change (optional): 3.5
- Is focal (optional): true/false
- Zygosity (optional): "homozygous", "heterozygous", "hemizygous"
- Origin: somatic/germline/unknown

**Zygosity extraction** (only when explicitly stated):
- "homozygous deletion" / "biallelic deletion" → zygosity="homozygous"
- "heterozygous deletion" / "heterozygous loss" → zygosity="heterozygous"
- "hemizygous deletion" → zygosity="hemizygous" (typically X chromosome in males)
- If not mentioned, leave as null

**Examples**:
- Cytogenetics: "del(11)(q13q23)" → gene="11q13-q23", alteration_direction="deletion", notes="Region-level CNA from cytogenetics"
- FISH: "KMT2A deletion" → gene="KMT2A", alteration_direction="deletion", notes="Gene-specific CNA by FISH"
- NGS: "MTAP homozygous deletion" → gene="MTAP", alteration_direction="deletion", zygosity="homozygous", notes="Gene-specific CNA"

**Normalization**:
- Use canonical terms: "amplification" (not "amp"), "deletion" (not "del")
- For regions: Use hyphen format "11q13-q23" (not parentheses)
- Common CNAs: EGFR amplification, MET amplification, MTAP deletion, CDKN2A deletion

## FUSION Findings (Gene Fusions)

Extract:
- Gene 5' partner: "EML4" (5' partner in EML4-ALK)
- Gene 3' partner: "ALK" (3' partner in EML4-ALK)
- Fusion type: "in_frame", "out_of_frame", "rearrangement", "unknown" (leave null if not explicitly stated)
- Exon 5' (optional): "exon 6"
- Exon 3' (optional): "exon 20"
- Variant frequency (optional): 0.45 (as decimal)
- Origin: somatic/germline/unknown (default: somatic)

**Normalization**:
- Use standard gene names: "ALK", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3"
- Order matters: 5' partner comes first (EML4-ALK, not ALK-EML4)
- For rearrangements without partner: Use gene as both 5' and 3' (e.g., ALK rearrangement → gene_5prime="ALK", gene_3prime="ALK")

## IHC Findings (Immunohistochemistry)

Extract:
- Biomarker: "HER2", "ER", "PR", "PD-L1", "MLH1", "MSH2", "MSH6", "PMS2", "AR"
- Intensity score (optional): 0-3 (for HER2-style scoring: 0=negative, 1=1+, 2=2+, 3=3+)
- Percentage positive (optional): 50.0 (for ER/PR/PD-L1, as 0-100)
- Is positive (optional): true/false (for binary results like MMR proteins)
- Staining intensity (optional): "weak", "moderate", "strong" (only if explicitly stated)
- H-score (optional): 0-300 (if reported)
- Allred score (optional): 0-8 (if reported)
- Score type (optional): "TPS", "CPS", "IC", "TC" (for PD-L1, only if explicitly stated)
- Assay (optional): "22C3", "SP263", "HercepTest", "4B5" (antibody/assay name if stated)

**Normalization**:
- Standardize biomarker names: "HER2" (not "Her-2" or "HER-2"), "PD-L1" (not "PDL1")
- For HER2: Use intensity_score (0-3) for standard IHC scoring
- For ER/PR: Use percentage_positive for positive cell percentage, staining_intensity if mentioned
- For PD-L1: Use percentage_positive for TPS/CPS values, note score_type if mentioned
- For MMR proteins (MLH1, MSH2, MSH6, PMS2): Use is_positive (true=intact/retained, false=lost/absent)

## FISH Findings (Fluorescence In Situ Hybridization)

Extract:
- Target: "HER2", "ALK", "ROS1", "EGFR", "MET"
- Ratio (optional): 2.5 (for HER2/CEP17 ratio)
- Copy number (optional): 8.2 (average copies per cell)
- Is amplified (optional): true/false (for amplification tests)
- Is rearranged (optional): true/false (for rearrangement/break-apart tests)
- Percentage positive (optional): 15.0 (percentage of cells with signal, as 0-100)
- Reference probe (optional): "CEP17" (for HER2 FISH)

**Normalization**:
- For HER2 FISH: Extract ratio and/or copy number, set is_amplified based on result
- For ALK/ROS1 break-apart: Use is_rearranged, extract percentage_positive if reported
- Target names: Use standard gene names ("HER2" not "ERBB2" for FISH targets)

## SIGNATURE Findings (Genomic Signatures)

Extract:
- Signature type: "MSI", "TMB", "MMR", "dMMR", "HRD"
- Result: "MSI-High", "MSI-Low", "MSS", "stable", "high", "deficient", "proficient", "intact", "lost"
- Quantitative value (optional): 18.9
- Unit (optional): "mutations/Mb", "muts/Mb"
- MSI result (optional): "MSI-High", "MSI-Low", "MSS" (only if signature_type is MSI)
- MMR result (optional): "proficient", "deficient", "intact", "lost" (only if signature_type is MMR or dMMR)

**Normalization**:
- Use standard abbreviations: "MSI" (not "microsatellite instability"), "TMB" (not "tumor mutational burden")
- Preserve result as stated: "MSI-High", "stable", "high"
- For TMB, extract numeric value if given and unit
- For MSI status: Use msi_result field with standardized value if possible
- For MMR status: Use mmr_result field with standardized value if possible

## WILDTYPE Findings

Extract when a gene is explicitly reported as wildtype/not detected and this is clinically relevant:
- Gene: "KRAS", "BRAF", "EGFR"
- Origin: somatic/germline/unknown (default: somatic)
- Alteration type tested (optional): "mutations", "amplification", "fusion", "any alteration"

**When to extract**:
- Only when explicitly stated as "wildtype", "not detected", "no mutations detected", "negative"
- Skip if it's just part of a negative results list without clinical context
- Common examples: "KRAS wildtype", "EGFR no mutations detected", "BRAF negative"

## Report vs Finding Structure

**CRITICAL**: Findings are grouped under TEST REPORTS. Each report has:
- **test_name**: "Omniseq", "FoundationOne", "Guardant360", etc. (or null if not specified)
- **test_methods**: List of test methods used (e.g., ["NGS", "IHC", "FISH", "PCR"]) - REQUIRED, a single report can use multiple methods
- **specimen_type**: "tissue", "blood", "plasma", "bone marrow", etc. (or null)
- **specimen_site**: Anatomical site/source - extract from phrases like:
  - "Right upper lobe lung biopsy" → "right upper lobe, lung"
  - "Bone marrow aspirate, posterior iliac crest" → "posterior iliac crest"
  - "Liver metastasis" → "liver metastasis"
  - "Primary tumor, colon" → "primary tumor, colon"
  - Use null only if no site information in document
- **tumor_content**: Tumor content/cellularity as percentage (0-100) if reported (or null)
- **findings**: List of all findings from this report

Each report can have multiple findings. All findings from the same test report should be grouped together.

**Report Consolidation Rules** (CRITICAL - read carefully):

**CONSOLIDATE into ONE report when**:
- Same specimen source + same collection date + comprehensive diagnostic workup
- Multiple test methods (NGS, FISH, IHC, Cytogenetics) performed on the same specimen
- Document sections are part of integrated clinical assessment 

**SEPARATE into MULTIPLE reports when**:
- Different specimen sources (e.g., tissue biopsy + blood draw)
- Different time points (e.g., baseline NGS + progression biopsy months later)
- Explicitly different test orders (e.g., commercial panel "FoundationOne" + separate institutional germline testing)
- Historical results mentioned for comparison (e.g., "previous FoundationOne from 2023 showed...")

**Default assumption**: If document describes a single clinical encounter/specimen collection → ONE comprehensive report with multiple test_methods


## Shared Fields (All Finding Types)

Every finding must have:
- **finding_type**: variant/cna/fusion/ihc/fish/signature/wildtype (determines which fields are valid)
- **raw_text**: Exact excerpt from document showing this finding
- **notes**: Optional additional context (e.g., "Region-level CNA", "Gene-specific CNA", "From cytogenetics")

Most findings also have:
- **origin**: somatic/germline/unknown (not present for IHC/FISH/SIGNATURE findings)

## Negative Results

If report explicitly states "No mutations detected" or "EGFR: negative" or "ALK: negative", you can SKIP these. Only extract POSITIVE findings.

Exception: If explicitly asked to extract negative results, do so, but prioritize positive findings if constrained by variant limit.

## Output Format

Generate markdown with findings nested under reports:

```
# Report 1
Test Name: [test name or null]
Test Methods: [method1, method2, ...]
Specimen Type: [specimen type or null]
Specimen Site: [specimen site or null]
Tumor Content: [percentage or null]

## Finding 1: [VARIANT|CNA|FUSION|IHC|FISH|SIGNATURE|WILDTYPE]
Finding Type: [variant|cna|fusion|ihc|fish|signature|wildtype]

[TYPE-SPECIFIC FIELDS]

For VARIANT:
  Gene: [gene]
  Canonical Variant: [variant]
  Origin: [somatic|germline|unknown]
  Protein Change: [protein change or null]
  cDNA Change: [cdna change or null]
  Exon: [exon or null]
  Variant Frequency: [0-1 or null]
  Is Hotspot: [true|false or null]

For CNA:
  Gene: [gene]
  Alteration Direction: [amplification|deletion|gain|loss|loss_of_heterozygosity]
  Origin: [somatic|germline|unknown]
  Copy Number: [copy number or null]
  Fold Change: [fold change or null]
  Is Focal: [true|false or null]
  Zygosity: [homozygous|heterozygous|hemizygous or null]

For FUSION:
  Gene 5 Prime: [5' gene]
  Gene 3 Prime: [3' gene]
  Origin: [somatic|germline|unknown]
  Fusion Type: [in_frame|out_of_frame|rearrangement|unknown or null]
  Exon 5 Prime: [exon or null]
  Exon 3 Prime: [exon or null]
  Variant Frequency: [0-1 or null]

For IHC:
  Biomarker: [biomarker]
  Intensity Score: [0-3 or null]
  Percentage Positive: [0-100 or null]
  Is Positive: [true|false or null]
  Staining Intensity: [weak|moderate|strong or null]
  H Score: [0-300 or null]
  Allred Score: [0-8 or null]
  Score Type: [TPS|CPS|IC|TC or null]
  Assay: [assay name or null]

For FISH:
  Target: [target gene]
  Ratio: [ratio or null]
  Copy Number: [copy number or null]
  Is Amplified: [true|false or null]
  Is Rearranged: [true|false or null]
  Percentage Positive: [0-100 or null]
  Reference Probe: [probe or null]

For SIGNATURE:
  Signature Type: [MSI|TMB|MMR|dMMR|HRD]
  Result: [result]
  Quantitative Value: [value or null]
  Unit: [unit or null]
  MSI Result: [MSI-High|MSI-Low|MSS or null]
  MMR Result: [proficient|deficient|intact|lost or null]

For WILDTYPE:
  Gene: [gene]
  Origin: [somatic|germline|unknown]
  Alteration Type Tested: [mutations|amplification|fusion|any alteration or null]

Raw Text: "[excerpt from document]"
Notes: [notes or null]

---

## Finding 2: [VARIANT|CNA|FUSION|IHC|FISH|SIGNATURE|WILDTYPE]
[... more findings from same report ...]

---

===

# Report 2
[... if document mentions multiple test reports ...]
```

Separate findings with `---` on their own line.
Separate reports with `===` on its own line (if multiple reports).

## Key Principles

1. **Discriminate correctly** - Each finding is ONE type only
2. **Normalize to canonical forms** - HUGO genes, protein-level variants, standard abbreviations
3. **Prioritize clinical relevance** - For variants, actionable > pathogenic > likely pathogenic
4. **Respect the 15 variant limit** - If genomic report has 50 mutations, extract top 15 most relevant
5. **Skip VUS** - Do not extract variants of uncertain significance
6. **Preserve exact values** - Don't round or modify scores/percentages
7. **Extract from raw text** - Pull exact excerpts showing each finding
8. **Optional enum fields** - For any optional enum field (e.g., fusion_type, staining_intensity, score_type), only populate if explicitly mentioned in the report. Leave as null if not found rather than guessing
"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted molecular findings data. Your job is to verify that the markdown representation accurately captures all relevant information from the original document.

## Validation Checks

1. **Schema Compliance** (CRITICAL - check first):
   - Are all fields in the markdown valid for the target Pydantic schema?
   - No extra fields beyond: finding_type, origin, raw_text, notes, and type-specific fields
   - Variant fields: gene, canonical_variant, origin, protein_change, cdna_change, exon, variant_frequency, is_hotspot
   - CNA fields: gene, alteration_direction, origin, copy_number, fold_change, is_focal, zygosity
   - IHC fields: biomarker, intensity_score, percentage_positive, is_positive, staining_intensity, h_score, allred_score, score_type, assay
   - FISH fields: target, ratio, copy_number, is_amplified, is_rearranged, percentage_positive, reference_probe
   - Fusion fields: gene_5prime, gene_3prime, origin, fusion_type, exon_5prime, exon_3prime, variant_frequency
   - Signature fields: signature_type, result, quantitative_value, unit, msi_result, mmr_result
   - Wildtype fields: gene, origin, alteration_type_tested
   - Report fields: test_name, test_methods (required list), specimen_type, specimen_site, tumor_content
   - NO fields like: report_id, clinical_significance, test_date, etc.

2. **Report Consolidation**: Are findings from the same specimen/date/workup incorrectly split into multiple reports when they should be consolidated into one comprehensive report with multiple test_methods?

3. **Report Structure**: Are findings properly grouped under reports with report metadata?

4. **Completeness**: Are all clinically significant findings captured?

5. **Correct Classification**: Is each finding in the right category (variant/cna/expression/signature)?

6. **Variant Constraints per report**:
   - Are there more than 15 variants in any single report? (FAIL if yes)
   - Are VUS mutations included? (FAIL if yes)
   - Are only pathogenic/likely pathogenic variants included? (PASS if yes)

7. **Normalization**: Are genes, variants, biomarkers using standardized nomenclature?

8. **Field Accuracy**: Do the type-specific fields match the finding type?

9. **Format Compliance**: Does the markdown follow the expected nested structure?

10. **Supporting Evidence**: Is raw_text present and relevant for each finding?

## What to Flag

- **Fields not in schema** (e.g., report_id, clinical_significance) - FAIL
- Wrong field types (e.g., test_methods as string instead of list)
- Missing report metadata (test_methods is required)
- Findings not properly nested under a report
- **Multiple reports when should be consolidated** - FAIL
- VUS mutations included (should be excluded)
- More than 15 variants in a single report
- Benign variants included
- Wrong finding type classification (e.g., HER2 IHC classified as variant instead of ihc)
- Non-standard gene names (e.g., "Her-2" instead of "HER2")
- Missing finding_type or origin
- Missing raw_text for findings
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
    """Create extraction agent for generating markdown from documents"""
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

        ORIGINAL DOCUMENT LENGTH: {len(ctx.deps.initial_document)} characters
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


async def extract_to_pydantic(
    markdown: str, model_config: ModelConfig
) -> MolecularFindingsExtraction:
    """Convert validated markdown to Pydantic model"""
    extraction_agent = Agent(
        model=model_config.model,
        output_type=wrap_output_type(
            MolecularFindingsExtraction, model_config.requires_prompted_output
        ),
        system_prompt="You are a precise data parser. Convert the provided markdown representation of molecular findings into the structured MolecularFindingsExtraction model. The markdown has a nested structure where findings are grouped under reports. Each report has metadata (test_name, test_methods, specimen_type, specimen_site, tumor_content) and a list of findings. Note that test_methods is a list of strings. All findings have shared fields (finding_type, raw_text, notes) and most have origin. Use the finding_type field to determine which specific finding model to use (VariantFinding, CNAFinding, FusionFinding, IHCFinding, FISHFinding, SignatureFinding, or WildtypeFinding). CRITICAL: The models have strict validation (extra='forbid') - only include fields that are defined in the schema. Preserve all information accurately.",
    )

    prompt = f"""
    Convert the following markdown representation of molecular findings into the MolecularFindingsExtraction model.
    
    Structure:
    - MolecularFindingsExtraction has a list of TestReport objects
    - Each TestReport has metadata (test_name, test_methods, specimen_type, specimen_site, tumor_content) and a list of findings
    - Each finding has shared fields (finding_type, raw_text, notes) and most have origin field. Specific types:
      - variant → VariantFinding (with gene, canonical_variant, origin, protein_change, cdna_change, exon, variant_frequency, is_hotspot)
      - cna → CNAFinding (with gene, alteration_direction, origin, copy_number, fold_change, is_focal, zygosity)
      - fusion → FusionFinding (with gene_5prime, gene_3prime, origin, fusion_type, exon_5prime, exon_3prime, variant_frequency)
      - ihc → IHCFinding (with biomarker, intensity_score, percentage_positive, is_positive, staining_intensity, h_score, allred_score, score_type, assay)
      - fish → FISHFinding (with target, ratio, copy_number, is_amplified, is_rearranged, percentage_positive, reference_probe)
      - signature → SignatureFinding (with signature_type, result, quantitative_value, unit, msi_result, mmr_result)
      - wildtype → WildtypeFinding (with gene, origin, alteration_type_tested)
    
    IMPORTANT: Models use ConfigDict(extra='forbid') - only include fields that exist in the schema. Do not add any extra fields.

    {markdown}
    """

    print("\n🔄 Converting markdown to Pydantic model...")
    async with extraction_agent.run_stream(prompt) as result:
        output = await result.get_output()
    return output


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
    model_config = get_default_model()

    print("=" * 80)
    print("🚀 Starting Molecular Findings Extraction")
    print(f"Model: {model_config.model.__class__.__name__}")
    print(
        f"Output Type: {'PromptedOutput' if model_config.requires_prompted_output else 'Structured Tools'}"
    )
    print("=" * 80)

    # Step 1: Generate initial markdown
    print("\n📝 Step 1: Generating initial markdown from document...")
    print("-" * 60)

    extraction_agent = create_extraction_agent(model_config)
    async with extraction_agent.run_stream(
        f"Extract molecular/genomic findings from this document:\n\n{document_text}"
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
        initial_document=document_text,
        initial_markdown=initial_markdown,
        current_markdown=initial_markdown,
        max_iterations=max_iterations,
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
        Validate the following markdown extraction against the original document.
        
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

    extraction = await extract_to_pydantic(state.current_markdown, model_config)

    total_findings = sum(len(report.findings) for report in extraction.reports)
    print(
        f"✅ Successfully extracted {len(extraction.reports)} report(s) with {total_findings} total findings"
    )

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


SAMPLE_DOCUMENT_OMNISEQ = """
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

SAMPLE_DOCUMENT_BONE_MARROW = """
<figure> 3 PatientKeeper\u00ae </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD # BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 Status Signed Source Fawcett Memorial Hospital BONE MARROW ASPIRATE RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Tanner Health System, Lahey Hospital Banner Estrella Medical Center, Lahey Hospital SAROYAN ST, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA LOC: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell, Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: WM:A66-48657 Submitting Doctor: Vanwell, Adrik D MD ** CONTINUED ON NEXT PAGE ** RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Tanner Health System, Lahey Hospital Inova Fairfax Hospital PABLO ROAD RD, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA LOC: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell,Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: WM:A66-48657 Submitting Doctor: Vanwell, Adrik D MD CLINICAL HISTORY Our patient is a 75-year-old woman with a history of breast cancer and therapy related myeloid neoplasm (MDS) displaying monoallelic TP53 and -11q undergoing surveillance bone marrow biopsy evaluation. ADDENDUM FINDINGS NeoTYPE(TM) Analysis AML Prognostic Profile SPECIMEN TYPE: Bone Marrow BODY SITE: Bone Marrow RESULTS SUMMARY: SNVs/Indels TP53 Y234N PERTINENT NEGATIVES: NO alterations detected in the following genes: FLT3, IDH1, IDH2, NPM1 <!-- PageNumber=\"Page 1 of 7\" --> <!-- PageBreak --> <figure> 3 PatientKeeper\u00ae </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD # BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 Status Signed Source Fawcett Memorial Hospital INTERPRETATION A single TP53 mutation with evidence of TP53 copy number loss indicates \"biallelic TP53 inactivation\" and is associated with a very poor prognosis. A complex karyotype may be regarded as presumptive evidence of TP53 copy loss on the trans allele. A VAF >10% with 10-19% blasts is classified as MDS with mutated TP53; TP53 VAF >10% with at least 20% blasts is consistent with AML with mutated TP53. Molecular Testing Detail 1\\. TP53 - p.Y234N; c .. 700T>A - NM_000546.5 - VAF 14.0% For additional details, please refer to the full report provided by NeoGenomics Laboratories. TEST RESULTS BY Montgomery General Hospital AFB/tm 07/09/24 Cytogenetics Oncology Chromosome Analysis ** CONTINUED ON NEXT PAGE ** RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Lahey Hospital RIANA, Lahey Hospital Saint Peter's University Hospital Summerlin Hospital, Lahey Hospital SAROYAN ST, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA LOC: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell, Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: BZ:B79-36840 Submitting Doctor: Vanwell, Adrik D MD ADDENDUM FINDINGS: (Continued) SPECIMEN TYPE: Bone Marrow BODY SITE: Bone Marrow KARYOTYPE: 46,XX,del(11)(q13q23)[13]/46,XX[7] INTERPRETATION: ABNORMAL FEMALE KARYOTYPE - DELETION of 11q, AS PREVIOUSLY REPORTED Cytogenetic analysis shows an abnormal female karyotype. Thirteen cells show a deletion of <!-- PageNumber=\"Page 2 of 7\" --> <!-- PageBreak --> <figure> 2 <!-- PageHeader=\"PatientKeeper\u00ae\" --> </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD # BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 Status Signed Source Fawcett Memorial Hospital the long arm of chromosome 11 and seven cells show a normal karyotype. Deletion of 11q is observed in MDS and is considered an indicator of very good prognosis. This finding was reported in a previous specimen (see NeoTRACK Results below), suggesting clonal persistence. These results should be interpreted in conjunction with clinical and other laboratory findings. For additional details, please refer to the full report provided by Montgomery General Hospital. TEST RESULTS BY Henry Ford Hospital LABORATORIES AFB/bs/CL/rc/AFB/ctm 07/06/24-07/11/24 Addendum Electronically signed Ege J Galba, MD 07/15/24 1547 FISH Analysis Multiple Probe Panel Results: Abnormal Interpretation: 11q23 KMT2A (MLL) rearrangement: Not Detected KMT2A (11q23) loss: Detected Del(17p): Not Detected Fluorescence in situ hybridization (FISH) analysis was performed using a dual color KMT2A (MLL) break apart probe set to detect rearrangement of the KMT2A (MLL) gene locus reported ** CONTINUED ON NEXT PAGE ** RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Lahey Hospital Front Royal Warren Memorial Hospital, Lahey Hospital Saint Peter's University Hospital Craig Hospital SAROYAN ST, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA JAYNI: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell, Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: WM:B79-36840 Submitting Doctor: Vanwell, Adrik D MD ADDENDUM FINDINGS: (Continued) in B-cell acute lymphoblastic leukemia (B-ALL). The abnormal signal pattern (1F) was observed in 54.5% of the analyzed nuclei. The normal reference range is < 3.1% and as such, this represents an ABNORMAL result indicative of loss of the KMT2A (MLL) gene region on chromosome 11 or loss of chromosome 11. Counts for the other probes are within the normal limits. For further information please reference the complete outside NeoGenomics report (QSY59-747010). PR/PR <!-- PageNumber=\"Page 3 of 7\" --> <!-- PageBreak --> <figure> 3 PatientKeeper </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD ## BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 Status Signed Source Fawcett Memorial Hospital 7/04/24 Addendum Electronically signed FINAL DIAGNOSIS Adrik I Farhat, MD 07/04/24 0828 A. BONE MARROW CORE BIOPSY, ASPIRATE SMEARS, TOUCH IMPRINTS, CLOT SECTIONS: \\- NORMAL CELLULAR MARROW (20-25% CELLULAR) WITH TRILINEAGE HEMATOPOIESIS. \\- NO CONVINCING MORPHOLOGIC EVIDENCE OF HEMATOLYMPHOID NEOPLASM OR MYELODYSPLASIA. \\- 3% BLASTS BY MORPHOLOGY; 6% BY CD34 IHC. \\- PENDING CYTOGENIC, FISH, AND MOLECULAR STUDIES. \\- MILDLY INCREASED STORAGE IRON BY SPECIAL STAIN. \\- SEE COMMENT. COMMENT: Histologic evaluation paired with immunohistochemical stains and flow cytometric analysis reveals a normal cellular and well-organized trilineage hematopoietic marrow with no convincing evidence of increased blast forms, myelodysplasia related change, or progression to acute leukemia. Pending cytogenic, FISH, and molecular studies will be reported in addendum to follow. AFB/ra 06/27/24 (1) GROSS DESCRIPTION A. Received in formalin labeled \"bone marrow aspirate\" is a 1.7 x 0.7 x 0.7 cm fragment of clotted blood. The specimen is sectioned and entirely submitted in cassette A1. B. Received in formalin labeled \"bone marrow biopsy\" are three pink tan trabecular bone ** CONTINUED ON NEXT PAGE ** RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Lahey Hospital Front Royal Warren Memorial Hospital, Lahey Hospital Inspira Medical Center SAROYAN ST, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA LOC: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell, Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: :B79-36840 Submitting Doctor: Vanwell, Adrik D MD GROSS DESCRIPTION: (Continued) core biopsy specimens ranging from 0.3 to 1.3 cm in length (diameter 2 mm). The specimen is submitted in toto in cassette B1 following bone marrow decalcification protocol. CMS/gs 06/24/24 MICROSCOPIC DESCRIPTION BONE MARROW ASPIRATE EVALUATION: Contents: Received for morphologic review are five aspirate smears including one stain for iron, two touch imprints, three sections of clot, and three sections of core biopsy. Cellularity: Spicular and cellular; adequate. <!-- PageNumber=\"Page 4 of 7\" --> <!-- PageBreak --> <figure> 3 PatientKeeper\u00ae </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD # BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 Status Signed Source Fawcett Memorial Hospital M/E ratio: 1.5. Iron Stain: Mildly increased storage iron; no ringed sideroblasts identified. DIFFERENTIAL COUNT: <table> <tr> <th>NORMAL %</th> <th>CELL TYPE</th> <th></th> <th></th> <th></th> <th colspan=\"2\">%</th> </tr> <tr> <td>0-5</td> <td>BLASTS</td> <td>3</td> <td></td> <td></td> <td colspan=\"2\"></td> </tr> <tr> <td>1-8</td> <td colspan=\"3\">PROMYELOCYTES</td> <td colspan=\"3\">1.8</td> </tr> <tr> <td>6-23</td> <td>MYELOCYTES</td> <td colspan=\"2\"></td> <td colspan=\"3\">11.2</td> </tr> <tr> <td>13-32</td> <td colspan=\"3\">METAMYELOCYTES</td> <td colspan=\"3\">13.6</td> </tr> <tr> <td>7-30</td> <td>MATURE NEUTROPHILS</td> <td colspan=\"2\"></td> <td></td> <td colspan=\"2\">23</td> </tr> <tr> <td>1-5</td> <td>MATURE EOSINOPHILS</td> <td colspan=\"2\"></td> <td></td> <td colspan=\"2\">2.2</td> </tr> <tr> <td>0-1</td> <td>MATURE BASOPHILS</td> <td colspan=\"2\"></td> <td colspan=\"3\">0</td> </tr> <tr> <td>1-5</td> <td>MONOCYTES</td> <td colspan=\"2\"></td> <td>1.2</td> <td colspan=\"2\"></td> </tr> <tr> <td>3-23</td> <td>LYMPHOCYTES</td> <td colspan=\"2\"></td> <td></td> <td colspan=\"2\">8</td> </tr> <tr> <td>0-4</td> <td>PLASMA CELLS</td> <td colspan=\"2\"></td> <td>0</td> <td colspan=\"2\"></td> </tr> <tr> <td>1-8</td> <td>PRONORMOBLASTS</td> <td colspan=\"2\"></td> <td></td> <td colspan=\"2\">6.8</td> </tr> <tr> <td>7-34</td> <td colspan=\"3\">NORMOBLASTS</td> <td></td> <td>29.2</td> <td></td> </tr> </table> Differential performed on 500 cells from the aspirate smear. MORPHOLOGY: Erythroids: Relative erythroid hyperplasia with predominance of polychromatophilic forms. ** CONTINUED ON NEXT PAGE ** RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Lahey Hospital RIANA, Lahey Hospital Saint Peter's University Hospital Craig Hospital SAROYAN ST, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA LOC: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell, Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: WM:B79-36840 Submitting Doctor: Vanwell, Adrik D MD MICROSCOPIC DESCRIPTION: (Continued) Myeloids: Appropriate full spectrum maturation with slight left shift lacking evidence of increased blast forms or convincing dysplastic change. Megakaryocytes: Scattered with occasional immature hypolobated forms. Others: Frequent macrophages. TOUCH IMPRINTS: Morphologically similar in cellular composition to the aspirate smear displaying trilineage hematopoietic elements with left shifted myeloid precursors lacking evidence of increased <!-- PageNumber=\"Page 5 of 7\" --> <!-- PageBreak --> <figure> 2 <!-- PageHeader=\"Patroller\" --> </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD # BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 Status Signed Source Fawcett Memorial Hospital blast forms or convincing dysplastic change. BONE MARROW CORE BIOPSY AND CLOT SECTION EVALUATION: Quality: Adequate bone marrow core biopsy (14 mm/18 mm) and clot section with few marrow particles. Microscopic Description: Histologic sections reveal a normal cellular marrow (20-25% cellular) with somewhat patchy, but well-organized trilineage hematopoietic elements displaying background erythroid hyperplasia and scattered hemosiderin deposition lacking convincing evidence of myelodysplastic syndrome, increased blasts, or malignancy. Immunohistochemical stains performed at Fawcett Memorial Hospital Laboratory with appropriate reactive controls on block B1 displays slightly increased CD34-positive myeloblasts comprising approximately 6% of the total bone marrow cellularity by CD34 IHC with partial coexpression of CD117. CD61 highlights scattered appropriate quantities of megakaryocytes. Special stain performed at Fawcett Memorial Hospital Laboratory with appropriate reactive controls on block B1 displays no significant reticulin fibrosis by special stain (MF 0/3). Flow cytometry performed at Fawcett Memorial Hospital Laboratory with appropriate quality metrics on the bone marrow aspirate specimen displays no evidence of increased blast forms or acute leukemia. For additional details, please see the full report (A66-69750). IMMUNOSTAIN REFERENCE: These tests were developed and their performance characteristics determined by Fawcett Memorial Hospital. It has not been cleared or approved by the U.S. Food and Drug Administration. The FDA has determined that such clearance or approval is not necessary. These tests are used for clinical purposes. They should not be regarded as investigational or for research. The laboratory is certified under the Clinical Laboratory Improvement Amendments of 1988 (CLIA) as qualified to perform high complexity clinical laboratory testing. ** CONTINUED ON NEXT PAGE ** RUN DATE: 07/15/24 Fawcett Memorial Hospital RUN TIME: 8241 Howard County General Hospital RUN USER: Tanner Health System, Lahey Hospital Banner Estrella Medical Center, METHODIST SHAFTER TER, NYU Langone Hospital 2703 FRANCONIA DRIVE GIBBON, NEW MEXICO 12094 (287)106-8560 NAME: SCHOMP,GIOVANNA LOC: MR#: N40711076 ACCT#: D616052344 DOB: 10/05/48 AGE/SEX:75/F REG: 06/24/24 ATTEND DR: Vanwell, Adrik D MD STATUS: DEP CLI DIS: Collection Date: 06/24/24 Received Date: 06/24/24 Pathology Number: WM:B79-36840 Submitting Doctor: Vanwell, Adrik D MD MICROSCOPIC DESCRIPTION: (Continued) REFERENCES: 1\\. Department of Health and Human Services, Food and Drug Administration. Medical devices: classification/reclassification; restricted devices; analyte specific reagents. Final rule. Federal register. 1977 (Nov. 10);62243 {21CFR809 and <!-- PageNumber=\"Page 6 of 7\" --> <!-- PageBreak --> <figure> 2 PatientKeeper </figure> Printed: 10/11/24 12:46 By: Coppick, Casey MD # BONE MARROW ASPIRATE SCHOMP, GIOVANNA Age: 76Y Gender: F DOB: 10/05/1948 MRN: N40711076 Phone: Date/Time 06/24/24 11:22 864}. Status Signed Source Fawcett Memorial Hospital 2\\. Jerrod CR. FDA issues final rule for classification and reclassification of immunochemistry reagents and kits. Am J Clin Pathol. 1999;111:443-470. 3\\. Swanson PE. Labels, disclaimer, and rules (oh, my!). Am J Clin Pathol. 1999;111:580-886. AFB/ra 06/27/24 <table> <tr> <td>Electronically signed by</td> <td>Ege J Galba, MD</td> <td>06/28/24 1650</td> </tr> <tr> <td>Electronically signed by</td> <td>Adrik I Farhat, MD</td> <td>07/04/24 0844</td> </tr> </table> ** END OF REPORT ** BONE MARROW ASPIRATE Specimen: WM:A66-30537 Status: SOUT Requisition: 36950706 Specimen Type: BONE MARROW ASPIRATE Coll Date: Jun 24, 2024 11:23am Recv Date: Jun 24, 2024 11:22am Received By: NICAISE,LILLIANA A Submitted By: Vanwell, Adrik D MD Ordered By: Vanwell, Adrik D MD Facility: Suburban Hospital <!-- PageNumber=\"Page 7 of 7\" --> <!-- PageBreak --> Maryland General Hospital (COCWM) PROGRESS NOTE REPORT#:9244-6046 REPORT STATUS: Signed DATE:07/18/24 TIME: 1516 PATIENT: SCHOMP, GIOVANNA UNIT #: N40711076 ACCOUNT#: M219984609 ROOM/BED: DOB: 10/05/48 AGE: 75 SEX: F ATTEND: Vanwell, Adrik D MD ADM DT: 07/13/24 AUTHOR: Vanwell, Adrik D MD RPT SRV DT: 07/18/24 RPT SRV TM: 1658 \\* ALL edits or amendments must be made on the electronic/computer document * CLINICAL NOTE - BMT CLINICAL NOTE NOTE: BONE MARROW TRANSPLANT CLINIC NOTE DIAGNOSIS: High-risk myelodysplastic syndrome with excess blasts. TRANSPLANT PROVIDER: Adrik D. Vanwell, MD REFERRING PROVIDER: Dr. Matier. MOLECULAR PROFILE: TP53 mutated MLL rearranged. DISEASE REMISSION STATUS: Stable disease on treatment oral venetaclax HISTORY OF PRESENT ILLNESS: The patient was found to have pancytopenia in May 2023, underwent bone marrow biopsy that revealed hypercellular marrow with trilineage dysplasia. Cytogenetics showed 11q deletion consistent with KMT2A MLL deletion as well as molecular studies revealed TP53. The patient was started on Vidaza and venetoclax. She states the Vidaza did not significantly help her blood counts. She did have reactions to the subcutaneous injections and once given IV Vidaza and then this was discontinued. She remained on 200mg daily of venetoclax and this was eventually lowered to 100mg p.o. day. The patient has remained on venetoclax 100mg p.o. day for the last several months and she reports that her blood counts have gradually improved. Blood counts from April 25, 2024 revealed a white cell count of 4.5, hemoglobin 10, platelets 107,000, neutrophils 71%, lymphocytes 20%, monocytes 5%. No circulating blasts. LDH was 110. Electrolytes normal. BUN 25, creatinine 1.38 with a creatinine clearance of 30. LFTs were normal. The patient states she has not needed any blood or platelet transfusions during this time. She otherwise feels well. She is working full time. Has no night sweats, fevers, or unintended weight loss. 7/18/24 here today to review bmbx results No MDS on marrow, 6% blasts, no evidence of leukemia, still has tp53 and 11q present at low levels, normocellulr 25% cellularity REVIEW OF SYSTEMS: CONSTITUTIONAL: Anxious. HEENT: Negative. PULMONARY: Negative. CARDIAC: Negative.
"""

# ============================================================================
# TEST FUNCTIONS
# ============================================================================


@pytest.mark.skip
@pytest.mark.integration
async def test_extract_molecular_findings_omniseq():
    """
    Integration test for molecular findings extraction with real API - Omniseq Report

    Clinical scenario: Comprehensive genomic profiling (Omniseq) on lung adenocarcinoma specimen
    showing actionable KRAS G12D mutation, TP53 mutation, copy number alterations, and
    high tumor mutational burden.

    Key features to validate:
    - Test report metadata (Omniseq, NGS/IHC methods, tissue specimen from right upper lobe lung)
    - Somatic variants: KRAS G12D (p.G12D, c.35G>A, VAF 42%), TP53 R175H (p.R175H, c.524G>A, VAF 38%)
    - Copy number alterations: MTAP deletion, CDKN2A homozygous deletion
    - Tumor mutational burden: 18.9 mutations/Mb (High)
    - Microsatellite instability: MSI-Stable
    - PD-L1 expression: TPS 1% (Low) by IHC with 22C3 antibody
    - Negative findings documented: EGFR, ALK, ROS1, HER2, MET, RET, NTRK
    """
    # Run extraction
    result = await extract_molecular_findings_async(
        SAMPLE_DOCUMENT_OMNISEQ, max_iterations=3
    )

    # Verify result success
    assert result.success is True, "Extraction should complete successfully"
    assert (
        result.extraction is not None
    ), "Extraction should contain molecular findings data"

    # Should have exactly one report (single Omniseq comprehensive panel)
    assert (
        len(result.extraction.reports) == 1
    ), "Should have one test report (Omniseq comprehensive panel)"

    report = result.extraction.reports[0]

    # ============================================================================
    # VALIDATE REPORT METADATA
    # ============================================================================

    # Test name should be Omniseq or similar comprehensive genomic profiling
    assert report.test_name is not None, "Test name should be documented"
    assert (
        "omniseq" in report.test_name.lower() or "genomic" in report.test_name.lower()
    ), f"Test name should reference Omniseq or genomic profiling, got: {report.test_name}"

    # Test methods should include NGS (for variants/CNAs) and IHC (for PD-L1)
    assert len(report.test_methods) >= 1, "Should document at least one test method"
    test_methods_str = " ".join(report.test_methods).upper()
    assert (
        "NGS" in test_methods_str or "SEQUENCING" in test_methods_str
    ), f"Should include NGS/sequencing method, got: {report.test_methods}"
    assert (
        "IHC" in test_methods_str or "IMMUNOHISTOCHEMISTRY" in test_methods_str
    ), f"Should include IHC method for PD-L1, got: {report.test_methods}"

    # Specimen should be tissue from right upper lobe lung
    assert report.specimen_type is not None, "Specimen type should be documented"
    assert (
        "tissue" in report.specimen_type.lower()
    ), f"Specimen type should be tissue, got: {report.specimen_type}"

    assert report.specimen_site is not None, "Specimen site should be documented"
    specimen_site_lower = report.specimen_site.lower()
    assert (
        "lung" in specimen_site_lower
    ), f"Specimen site should include lung, got: {report.specimen_site}"
    assert (
        "right" in specimen_site_lower and "upper" in specimen_site_lower
    ), f"Specimen site should specify right upper lobe, got: {report.specimen_site}"

    # Should have multiple findings (variants, CNAs, signatures, IHC)
    assert (
        len(report.findings) >= 6
    ), f"Should have at least 6 findings (2 variants, 2 CNAs, 2 signatures, 1 IHC minimum), got {len(report.findings)}"

    # ============================================================================
    # SEPARATE FINDINGS BY TYPE
    # ============================================================================

    variants = [f for f in report.findings if f.finding_type == FindingType.VARIANT]
    cnas = [f for f in report.findings if f.finding_type == FindingType.CNA]
    ihc_findings = [f for f in report.findings if f.finding_type == FindingType.IHC]
    fish_findings = [f for f in report.findings if f.finding_type == FindingType.FISH]
    fusions = [f for f in report.findings if f.finding_type == FindingType.FUSION]
    signatures = [f for f in report.findings if f.finding_type == FindingType.SIGNATURE]
    wildtypes = [f for f in report.findings if f.finding_type == FindingType.WILDTYPE]

    # ============================================================================
    # VALIDATE VARIANTS (SOMATIC MUTATIONS)
    # ============================================================================

    # Should have exactly 2 pathogenic variants documented (KRAS G12D, TP53 R175H)
    assert (
        len(variants) == 2
    ), f"Should extract exactly 2 pathogenic variants, got {len(variants)}"

    # Check KRAS G12D variant
    kras = next((v for v in variants if v.gene.upper() == "KRAS"), None)
    assert kras is not None, "KRAS variant should be extracted"
    assert (
        "G12D" in kras.canonical_variant
    ), f"KRAS variant should be G12D, got: {kras.canonical_variant}"
    assert kras.origin == Origin.SOMATIC, "KRAS should be somatic"

    # Check protein and cDNA changes if captured
    if kras.protein_change:
        assert (
            "G12D" in kras.protein_change
        ), f"KRAS protein change should contain G12D, got: {kras.protein_change}"
    if kras.cdna_change:
        assert (
            "35G>A" in kras.cdna_change or "c.35" in kras.cdna_change
        ), f"KRAS cDNA change should be c.35G>A, got: {kras.cdna_change}"

    # Check VAF
    assert kras.variant_frequency is not None, "KRAS VAF should be documented"
    assert (
        abs(kras.variant_frequency - 0.42) < 0.01
    ), f"KRAS VAF should be ~0.42 (42%), got: {kras.variant_frequency}"

    # Check TP53 R175H variant
    tp53 = next((v for v in variants if v.gene.upper() == "TP53"), None)
    assert tp53 is not None, "TP53 variant should be extracted"
    assert (
        "R175H" in tp53.canonical_variant
    ), f"TP53 variant should be R175H, got: {tp53.canonical_variant}"
    assert tp53.origin == Origin.SOMATIC, "TP53 should be somatic"

    # Check protein and cDNA changes if captured
    if tp53.protein_change:
        assert (
            "R175H" in tp53.protein_change
        ), f"TP53 protein change should contain R175H, got: {tp53.protein_change}"
    if tp53.cdna_change:
        assert (
            "524G>A" in tp53.cdna_change or "c.524" in tp53.cdna_change
        ), f"TP53 cDNA change should be c.524G>A, got: {tp53.cdna_change}"

    # Check VAF
    assert tp53.variant_frequency is not None, "TP53 VAF should be documented"
    assert (
        abs(tp53.variant_frequency - 0.38) < 0.01
    ), f"TP53 VAF should be ~0.38 (38%), got: {tp53.variant_frequency}"

    # ============================================================================
    # VALIDATE COPY NUMBER ALTERATIONS
    # ============================================================================

    # Should have 2 CNAs (MTAP deletion, CDKN2A deletion)
    assert len(cnas) == 2, f"Should extract 2 copy number alterations, got {len(cnas)}"

    # Check MTAP deletion
    mtap = next((c for c in cnas if c.gene.upper() == "MTAP"), None)
    assert mtap is not None, "MTAP deletion should be extracted"
    assert (
        mtap.alteration_direction == CNADirection.DELETION
    ), f"MTAP should be deletion, got: {mtap.alteration_direction}"
    assert mtap.origin == Origin.SOMATIC, "MTAP deletion should be somatic"

    # Check CDKN2A deletion
    cdkn2a = next((c for c in cnas if c.gene.upper() == "CDKN2A"), None)
    assert cdkn2a is not None, "CDKN2A deletion should be extracted"
    assert (
        cdkn2a.alteration_direction == CNADirection.DELETION
    ), f"CDKN2A should be deletion, got: {cdkn2a.alteration_direction}"
    assert cdkn2a.origin == Origin.SOMATIC, "CDKN2A deletion should be somatic"

    # Zygosity should be homozygous (explicitly stated in report)
    assert (
        cdkn2a.zygosity == Zygosity.HOMOZYGOUS
    ), f"CDKN2A should be homozygous deletion (explicitly stated in report), got: {cdkn2a.zygosity}"

    # ============================================================================
    # VALIDATE GENOMIC SIGNATURES (TMB, MSI)
    # ============================================================================

    # Should have 2 signature findings (TMB, MSI)
    assert (
        len(signatures) >= 2
    ), f"Should extract at least 2 signatures (TMB, MSI), got {len(signatures)}"

    # Check TMB
    tmb = next((s for s in signatures if s.signature_type == SignatureType.TMB), None)
    assert tmb is not None, "TMB should be extracted"
    assert (
        "high" in tmb.result.lower()
    ), f"TMB result should indicate high, got: {tmb.result}"
    assert (
        tmb.quantitative_value is not None
    ), "TMB quantitative value should be documented"
    assert (
        abs(tmb.quantitative_value - 18.9) < 0.1
    ), f"TMB should be 18.9, got: {tmb.quantitative_value}"
    assert tmb.unit is not None, "TMB unit should be documented"
    assert (
        "mut" in tmb.unit.lower() and "mb" in tmb.unit.lower()
    ), f"TMB unit should be mutations/Mb, got: {tmb.unit}"

    # Check MSI
    msi = next((s for s in signatures if s.signature_type == SignatureType.MSI), None)
    assert msi is not None, "MSI should be extracted"
    assert (
        "stable" in msi.result.lower() or "mss" in msi.result.lower()
    ), f"MSI result should indicate stable/MSS, got: {msi.result}"
    # Should standardize to MSI_RESULT enum if possible
    if msi.msi_result:
        assert (
            msi.msi_result == MSIResult.MSS
        ), f"MSI standardized result should be MSS, got: {msi.msi_result}"

    # ============================================================================
    # VALIDATE IHC FINDINGS (PD-L1)
    # ============================================================================

    # Should have 1 IHC finding (PD-L1)
    assert (
        len(ihc_findings) >= 1
    ), f"Should extract at least 1 IHC finding (PD-L1), got {len(ihc_findings)}"

    # Check PD-L1
    pdl1 = next(
        (
            f
            for f in ihc_findings
            if "PD-L1" in f.biomarker.upper() or "PDL1" in f.biomarker.upper()
        ),
        None,
    )
    assert pdl1 is not None, "PD-L1 IHC should be extracted"
    assert pdl1.biomarker.upper() in [
        "PD-L1",
        "PDL1",
        "PD-L1",
    ], f"PD-L1 biomarker name should be standardized, got: {pdl1.biomarker}"

    # TPS 1% should be captured as percentage_positive
    assert (
        pdl1.percentage_positive is not None
    ), "PD-L1 TPS percentage should be documented"
    assert (
        abs(pdl1.percentage_positive - 1.0) < 0.1
    ), f"PD-L1 TPS should be 1%, got: {pdl1.percentage_positive}"

    # Score type should be TPS if captured
    if pdl1.score_type:
        assert (
            pdl1.score_type == PDL1ScoreType.TPS
        ), f"PD-L1 score type should be TPS, got: {pdl1.score_type}"

    # Assay should be 22C3 if captured
    if pdl1.assay:
        assert "22C3" in pdl1.assay, f"PD-L1 assay should be 22C3, got: {pdl1.assay}"

    # ============================================================================
    # VALIDATE NO FUSIONS OR FISH (none in this report)
    # ============================================================================

    # Should have no fusion findings
    assert (
        len(fusions) == 0
    ), f"Should have no fusion findings in this report, got {len(fusions)}"

    # Should have no FISH findings (ALK, ROS1 tested by IHC, not FISH in this report)
    assert (
        len(fish_findings) == 0
    ), f"Should have no FISH findings in this report, got {len(fish_findings)}"

    # ============================================================================
    # VALIDATE WILDTYPE/NEGATIVE FINDINGS (optional)
    # ============================================================================

    # Negative findings may or may not be extracted (EGFR, ALK, ROS1, etc.)
    # Per instructions, these are typically skipped unless explicitly requested
    # So we don't assert their presence, but if they're there, they should be correct

    if len(wildtypes) > 0:
        print(f"✓ Found {len(wildtypes)} wildtype findings (negative results)")
        for wt in wildtypes:
            assert (
                wt.origin == Origin.SOMATIC
            ), f"Wildtype findings should be somatic, got: {wt.origin} for {wt.gene}"
            assert wt.gene in [
                "EGFR",
                "ALK",
                "ROS1",
                "HER2",
                "MET",
                "RET",
                "NTRK1",
                "NTRK2",
                "NTRK3",
            ], f"Wildtype gene should be one of the tested genes, got: {wt.gene}"

    # ============================================================================
    # VALIDATE RAW TEXT AND EVIDENCE
    # ============================================================================

    # All findings should have raw_text
    for finding in report.findings:
        assert (
            finding.raw_text
        ), f"Finding of type {finding.finding_type} should have raw_text"
        assert (
            len(finding.raw_text) > 0
        ), f"Raw text should not be empty for {finding.finding_type}"

    # ============================================================================
    # CLINICAL SUMMARY OUTPUT
    # ============================================================================

    print("\n" + "=" * 80)
    print("TEST PASSED - OMNISEQ COMPREHENSIVE GENOMIC PROFILING")
    print("=" * 80)
    print(f"\nTest Report:")
    print(f"  Name: {report.test_name}")
    print(f"  Methods: {', '.join(report.test_methods)}")
    print(f"  Specimen: {report.specimen_type} from {report.specimen_site}")
    if report.tumor_content:
        print(f"  Tumor Content: {report.tumor_content}%")

    print(f"\nFindings Summary ({len(report.findings)} total):")
    print(f"  ├─ Variants: {len(variants)}")
    for v in variants:
        vaf_str = (
            f", VAF: {v.variant_frequency*100:.1f}%" if v.variant_frequency else ""
        )
        print(f"  │  └─ {v.gene} {v.canonical_variant} ({v.origin.value}{vaf_str})")

    print(f"  ├─ Copy Number Alterations: {len(cnas)}")
    for c in cnas:
        print(f"  │  └─ {c.gene} {c.alteration_direction.value}")

    print(f"  ├─ Signatures: {len(signatures)}")
    for s in signatures:
        quant_str = (
            f" ({s.quantitative_value} {s.unit})" if s.quantitative_value else ""
        )
        print(f"  │  └─ {s.signature_type.value}: {s.result}{quant_str}")

    print(f"  ├─ IHC: {len(ihc_findings)}")
    for i in ihc_findings:
        if i.percentage_positive is not None:
            print(f"  │  └─ {i.biomarker}: {i.percentage_positive}%")
        elif i.intensity_score is not None:
            print(f"  │  └─ {i.biomarker}: {i.intensity_score}+")
        else:
            print(f"  │  └─ {i.biomarker}")

    if len(fusions) > 0:
        print(f"  ├─ Fusions: {len(fusions)}")
    if len(fish_findings) > 0:
        print(f"  ├─ FISH: {len(fish_findings)}")
    if len(wildtypes) > 0:
        print(f"  └─ Wildtype: {len(wildtypes)}")

    print("\n" + "=" * 80)
    print("CLINICAL IMPLICATIONS")
    print("=" * 80)
    print("• KRAS G12D: Actionable mutation (consider KRAS G12D inhibitors)")
    print("• TP53 R175H: Pathogenic mutation, associated with treatment resistance")
    print("• TMB-High (18.9 mut/Mb): May benefit from immune checkpoint inhibitors")
    print("• PD-L1 TPS 1%: Low expression, but TMB-high supports immunotherapy")
    print("• MTAP/CDKN2A deletions: Cell cycle pathway alterations")
    print("=" * 80)

    print("\nFull extraction:")
    print(result.extraction.model_dump_json(indent=2))


@pytest.mark.integration
async def test_extract_molecular_findings_bone_marrow():
    """
    Integration test for molecular findings extraction - Bone Marrow Aspirate with MDS

    Clinical scenario: 75-year-old woman with therapy-related myeloid neoplasm (MDS)
    with biallelic TP53 inactivation and 11q deletion undergoing surveillance bone marrow biopsy.

    Key features to validate:
    - Test report metadata (NeoTYPE NGS panel, cytogenetics, FISH, bone marrow specimen)
    - Variant: TP53 Y234N (c.700T>A, VAF 14.0%)
    - CNA: del(11)(q13q23) - chromosomal region deletion from cytogenetics
    - CNA: KMT2A (MLL) loss at 11q23 from FISH (54.5% of nuclei)
    - Pertinent negatives documented: FLT3, IDH1, IDH2, NPM1
    - Clinical context: Biallelic TP53 inactivation (mutation + copy loss)
    """
    # Run extraction
    result = await extract_molecular_findings_async(
        SAMPLE_DOCUMENT_BONE_MARROW, max_iterations=3
    )

    # Verify result success
    assert result.success is True, "Extraction should complete successfully"
    assert (
        result.extraction is not None
    ), "Extraction should contain molecular findings data"

    # Should have one comprehensive report (same specimen, same date, integrated workup)
    # Per consolidation rules: bone marrow 06/24/24 with NGS + Cytogenetics + FISH + IHC = ONE report
    assert (
        len(result.extraction.reports) == 1
    ), f"Should consolidate findings into one comprehensive report, got {len(result.extraction.reports)}"

    report = result.extraction.reports[0]

    # ============================================================================
    # VALIDATE REPORT METADATA
    # ============================================================================

    # Test name should reference comprehensive molecular testing
    # Could be NeoTYPE, or generic "Comprehensive Bone Marrow Analysis", etc.
    if report.test_name:
        test_name_lower = report.test_name.lower()
        # Accept various descriptors for comprehensive molecular workup
        assert any(
            term in test_name_lower
            for term in [
                "neotype",
                "molecular",
                "genomic",
                "comprehensive",
                "bone marrow",
            ]
        ), f"Test name should reference molecular/comprehensive testing, got: {report.test_name}"

    # Test methods should include multiple methods (NGS, Cytogenetics, FISH, IHC)
    assert (
        len(report.test_methods) >= 2
    ), f"Comprehensive workup should include multiple test methods, got {len(report.test_methods)}"

    test_methods_str = " ".join(report.test_methods).upper()

    # Should include NGS or sequencing
    assert (
        "NGS" in test_methods_str
        or "SEQUENCING" in test_methods_str
        or "MOLECULAR" in test_methods_str
    ), f"Should include NGS/sequencing method, got: {report.test_methods}"

    # Specimen should be bone marrow
    assert report.specimen_type is not None, "Specimen type should be documented"
    specimen_type_lower = report.specimen_type.lower()
    assert (
        "bone marrow" in specimen_type_lower or "marrow" in specimen_type_lower
    ), f"Specimen type should be bone marrow, got: {report.specimen_type}"

    # Specimen site should reference bone marrow source (if captured)
    if report.specimen_site:
        specimen_site_lower = report.specimen_site.lower()
        assert (
            "marrow" in specimen_site_lower
            or "iliac" in specimen_site_lower
            or "aspirate" in specimen_site_lower
        ), f"Specimen site should reference bone marrow location, got: {report.specimen_site}"

    # Should have multiple findings (at least TP53 variant + CNAs)
    assert (
        len(report.findings) >= 3
    ), f"Should have at least 3 findings (TP53 variant + 2 CNAs minimum), got {len(report.findings)}"

    # ============================================================================
    # SEPARATE FINDINGS BY TYPE
    # ============================================================================

    variants = [f for f in report.findings if f.finding_type == FindingType.VARIANT]
    cnas = [f for f in report.findings if f.finding_type == FindingType.CNA]
    ihc_findings = [f for f in report.findings if f.finding_type == FindingType.IHC]
    fish_findings = [f for f in report.findings if f.finding_type == FindingType.FISH]
    fusions = [f for f in report.findings if f.finding_type == FindingType.FUSION]
    signatures = [f for f in report.findings if f.finding_type == FindingType.SIGNATURE]
    wildtypes = [f for f in report.findings if f.finding_type == FindingType.WILDTYPE]

    # ============================================================================
    # VALIDATE VARIANTS (TP53 MUTATION)
    # ============================================================================

    # Should have at least 1 variant (TP53 Y234N)
    assert (
        len(variants) >= 1
    ), f"Should extract at least 1 variant (TP53), got {len(variants)}"

    # Check TP53 Y234N variant
    tp53 = next((v for v in variants if v.gene.upper() == "TP53"), None)
    assert tp53 is not None, "TP53 variant should be extracted"
    assert (
        "Y234N" in tp53.canonical_variant
    ), f"TP53 variant should be Y234N, got: {tp53.canonical_variant}"

    # Origin should be somatic (this is therapy-related MDS)
    assert (
        tp53.origin == Origin.SOMATIC
    ), f"TP53 should be somatic in therapy-related MDS, got: {tp53.origin}"

    # Check cDNA change if captured
    if tp53.cdna_change:
        assert (
            "700T>A" in tp53.cdna_change or "c.700" in tp53.cdna_change
        ), f"TP53 cDNA change should be c.700T>A, got: {tp53.cdna_change}"

    # Check VAF
    assert tp53.variant_frequency is not None, "TP53 VAF should be documented"
    assert (
        abs(tp53.variant_frequency - 0.14) < 0.01
    ), f"TP53 VAF should be ~0.14 (14%), got: {tp53.variant_frequency}"

    # ============================================================================
    # VALIDATE COPY NUMBER ALTERATIONS
    # ============================================================================

    # Should have at least 1-2 CNAs (11q deletion from cytogenetics and/or KMT2A loss from FISH)
    # Note: The cytogenetics shows del(11)(q13q23) and FISH shows KMT2A loss at 11q23
    # These could be reported as one or two findings depending on how model interprets
    assert (
        len(cnas) >= 1
    ), f"Should extract at least 1 copy number alteration (11q deletion), got {len(cnas)}"

    # Check for 11q deletion (either as chromosomal region or KMT2A gene-specific)
    # Could be reported as "11q13-q23" (cytogenetics) or "KMT2A" (FISH)
    deletion_11q = next(
        (c for c in cnas if "11q" in c.gene or c.gene.upper() in ["KMT2A", "MLL"]), None
    )
    assert deletion_11q is not None, "11q deletion or KMT2A loss should be extracted"
    assert (
        deletion_11q.alteration_direction == CNADirection.DELETION
        or deletion_11q.alteration_direction == CNADirection.LOSS
    ), f"11q/KMT2A should be deletion or loss, got: {deletion_11q.alteration_direction}"

    # If reported as chromosomal region, should use band notation
    if "q" in deletion_11q.gene.lower() and "11" in deletion_11q.gene:
        assert (
            "11q" in deletion_11q.gene
        ), f"Chromosomal deletion should use band notation like 11q13-q23, got: {deletion_11q.gene}"
        # Should note it's from cytogenetics
        if deletion_11q.notes:
            assert (
                "cytogenetic" in deletion_11q.notes.lower()
                or "karyotype" in deletion_11q.notes.lower()
                or "region" in deletion_11q.notes.lower()
            ), f"Chromosomal CNA should note source in notes, got: {deletion_11q.notes}"

    # If reported as KMT2A, should be gene-specific
    if deletion_11q.gene.upper() in ["KMT2A", "MLL"]:
        # Should note it's from FISH if captured
        if deletion_11q.notes:
            assert (
                "fish" in deletion_11q.notes.lower()
                or "gene" in deletion_11q.notes.lower()
            ), f"Gene-specific CNA should note FISH source, got: {deletion_11q.notes}"

    # ============================================================================
    # VALIDATE FISH FINDINGS (KMT2A/MLL)
    # ============================================================================

    # The KMT2A loss could be reported as:
    # 1. A CNA finding (loss/deletion of KMT2A gene)
    # 2. A FISH finding (abnormal FISH signal pattern)
    # Either is acceptable depending on interpretation

    # If extracted as FISH finding:
    if len(fish_findings) > 0:
        kmt2a_fish = next(
            (
                f
                for f in fish_findings
                if "KMT2A" in f.target.upper() or "MLL" in f.target.upper()
            ),
            None,
        )
        if kmt2a_fish:
            # Should not show rearrangement (report says "not detected")
            assert (
                kmt2a_fish.is_rearranged is None or kmt2a_fish.is_rearranged is False
            ), f"KMT2A rearrangement should be negative, got: {kmt2a_fish.is_rearranged}"

            # May capture percentage with abnormal signal (54.5%)
            if kmt2a_fish.percentage_positive:
                assert (
                    abs(kmt2a_fish.percentage_positive - 54.5) < 1.0
                ), f"KMT2A FISH abnormal signal should be ~54.5%, got: {kmt2a_fish.percentage_positive}"

    # ============================================================================
    # VALIDATE WILDTYPE/NEGATIVE FINDINGS
    # ============================================================================

    # Report documents pertinent negatives: FLT3, IDH1, IDH2, NPM1
    # These may or may not be extracted as wildtype findings

    if len(wildtypes) > 0:
        print(f"✓ Found {len(wildtypes)} wildtype findings (pertinent negatives)")

        # If extracted, should include some of: FLT3, IDH1, IDH2, NPM1
        wildtype_genes = [wt.gene.upper() for wt in wildtypes]
        pertinent_negatives = ["FLT3", "IDH1", "IDH2", "NPM1"]
        found_negatives = [
            gene for gene in pertinent_negatives if gene in wildtype_genes
        ]

        if found_negatives:
            print(f"  Documented negatives: {', '.join(found_negatives)}")

    # ============================================================================
    # VALIDATE RAW TEXT AND EVIDENCE
    # ============================================================================

    # All findings should have raw_text
    for finding in report.findings:
        assert (
            finding.raw_text
        ), f"Finding of type {finding.finding_type} should have raw_text"
        assert (
            len(finding.raw_text) > 0
        ), f"Raw text should not be empty for {finding.finding_type}"

    # ============================================================================
    # CLINICAL SUMMARY OUTPUT
    # ============================================================================

    print("\n" + "=" * 80)
    print("TEST PASSED - BONE MARROW MDS WITH BIALLELIC TP53 INACTIVATION")
    print("=" * 80)
    print(f"\nTest Report:")
    print(f"  Name: {report.test_name}")
    print(f"  Methods: {', '.join(report.test_methods)}")
    print(
        f"  Specimen: {report.specimen_type}"
        + (f" from {report.specimen_site}" if report.specimen_site else "")
    )
    if report.tumor_content:
        print(f"  Tumor Content: {report.tumor_content}%")

    print(f"\nFindings Summary ({len(report.findings)} total):")

    if len(variants) > 0:
        print(f"  ├─ Variants: {len(variants)}")
        for v in variants:
            vaf_str = (
                f", VAF: {v.variant_frequency*100:.1f}%" if v.variant_frequency else ""
            )
            print(f"  │  └─ {v.gene} {v.canonical_variant} ({v.origin.value}{vaf_str})")

    if len(cnas) > 0:
        print(f"  ├─ Copy Number Alterations: {len(cnas)}")
        for c in cnas:
            notes_str = f" ({c.notes})" if c.notes else ""
            print(f"  │  └─ {c.gene} {c.alteration_direction.value}{notes_str}")

    if len(fish_findings) > 0:
        print(f"  ├─ FISH: {len(fish_findings)}")
        for f in fish_findings:
            print(f"  │  └─ {f.target}")

    if len(signatures) > 0:
        print(f"  ├─ Signatures: {len(signatures)}")
        for s in signatures:
            quant_str = (
                f" ({s.quantitative_value} {s.unit})" if s.quantitative_value else ""
            )
            print(f"  │  └─ {s.signature_type.value}: {s.result}{quant_str}")

    if len(ihc_findings) > 0:
        print(f"  ├─ IHC: {len(ihc_findings)}")
        for i in ihc_findings:
            if i.percentage_positive is not None:
                print(f"  │  └─ {i.biomarker}: {i.percentage_positive}%")
            elif i.intensity_score is not None:
                print(f"  │  └─ {i.biomarker}: {i.intensity_score}+")
            else:
                print(f"  │  └─ {i.biomarker}")

    if len(fusions) > 0:
        print(f"  ├─ Fusions: {len(fusions)}")
        for fusion in fusions:
            print(f"  │  └─ {fusion.gene_5prime}-{fusion.gene_3prime}")

    if len(wildtypes) > 0:
        print(f"  └─ Wildtype/Negative: {len(wildtypes)}")
        for wt in wildtypes:
            print(f"     └─ {wt.gene} (no alterations detected)")

    print("\n" + "=" * 80)
    print("CLINICAL IMPLICATIONS")
    print("=" * 80)
    print("• TP53 Y234N mutation: Pathogenic, associated with poor prognosis in MDS")
    print(
        "• Biallelic TP53 inactivation: Mutation + 11q copy loss (very poor prognosis)"
    )
    print("• 11q deletion (del(11)(q13q23)): Involves KMT2A/MLL gene region")
    print(
        "• KMT2A (MLL) loss by FISH: Confirms deletion at gene level (54.5% of cells)"
    )
    print("• Therapy-related MDS: Secondary to prior breast cancer treatment")
    print("• Blast count: 3% (morphology) / 6% (CD34 IHC) - below AML threshold")
    print("• Pertinent negatives: FLT3, IDH1, IDH2, NPM1 wildtype")
    print("=" * 80)

    print("\nFull extraction:")
    print(result.extraction.model_dump_json(indent=2))
