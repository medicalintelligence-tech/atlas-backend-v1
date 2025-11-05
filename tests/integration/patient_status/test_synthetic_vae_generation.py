# Synthetic Patient Data Generation
#
# This test demonstrates direct LLM-based generation of synthetic patient cancer data.
# Unlike the extraction pipeline (which processes medical documents), this directly
# generates structured PatientCancerExtraction objects from natural language prompts.
#
# Use cases:
# - Generate test data for trial matching algorithms
# - Create evaluation datasets with known ground truth
# - Rapidly prototype with realistic patient profiles
# - Test edge cases and specific molecular profiles

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from config.settings import settings
import pytest

# Import the models from test_vae_structure
from tests.integration.patient_status.test_vae_structure import (
    PatientCancerExtraction,
    SolidCancer,
    BloodCancer,
    NeuroendocrineTumor,
    CancerType,
    DiseaseStatus,
)


# ============================================================================
# SYSTEM PROMPT FOR SYNTHETIC DATA GENERATION
# ============================================================================

SYNTHETIC_GENERATION_SYSTEM_PROMPT = """You are a world-class synthetic medical data generator specializing in oncology. Your role is to create realistic, clinically coherent cancer patient profiles for Phase I clinical trial matching and evaluation purposes.

## Core Philosophy

**Realism Over Perfection**: Generate patients that reflect real-world oncology practice:
- Not every field needs to be filled
- Molecular profiles have natural variation
- Treatment histories follow logical clinical sequences
- Disease progression patterns are biologically plausible

**Clinical Coherence**: Ensure internal consistency:
- Treatment choices match molecular profile and disease stage
- Disease status aligns with clinical trajectory
- Metastatic patterns follow typical biology
- Prior therapy lines build logically

**Diversity in Specificity**: Balance detail appropriately:
- Some measurements are precise (tumor size, lab values)
- Some are descriptive (anatomic locations, drug names)
- Some may be unknown or not documented

## Generation Principles

### 1. Molecular Profiles
- Include actionable alterations when relevant to the cancer type
- Don't force rare mutations unless specifically requested
- Expression markers (PD-L1, HER2, hormone receptors) should match cancer type
- TMB and MSI status should reflect typical distributions
- Not every patient needs a complete genomic profile

### 2. Treatment Histories
- Use actual drug names, not generic classes
- Follow standard-of-care sequences for the cancer type
- Earlier lines: combination chemotherapy or targeted therapy
- Later lines: single agents, clinical trials, experimental approaches
- Some patients may be newly diagnosed (prior_therapy_lines: 0)

### 3. Tumor Hierarchies (Solid/NET cancers)
- Primary tumor is the origin site
- Metastases are common sites for that cancer type
  - Lung cancer → brain, bone, liver, adrenal
  - Breast cancer → bone, liver, lung, brain
  - Colorectal → liver, lung, peritoneum
- Use precise anatomic locations: "right upper lobe lung", "liver segment 6", "left frontal brain", "L3 vertebra"
- Not every metastatic site needs detailed molecular profiling

### 4. Disease Status Logic
- newly diagnosed → active, untreated
- on treatment, responding → responding or stable
- on treatment, not working → progressing
- completed curative-intent treatment → no_evidence_of_disease or remission
- Be realistic about prognosis and trajectory

### 5. Blood Cancers
- Use standard classification (AML, CLL, DLBCL, multiple myeloma, etc.)
- Disease burden metrics appropriate to subtype (blast %, M-protein, etc.)
- Molecular markers common to blood cancers (TP53, del17p, complex karyotype, etc.)
- Prior transplant history when relevant

### 6. Field Completion
- **Always include**: cancer_type, histology/subtype, disease_status, primary tumor location
- **Often include**: molecular profile (if testing done), prior therapies (if any), metastatic sites (if stage IV)
- **Sometimes include**: specific tumor sizes, detailed molecular data, precise measurements
- **Optional/Unknown**: End dates for ongoing therapies, specific test dates, less common molecular markers

## Output Structure

Generate realistic data that fits the PatientCancerExtraction schema:
- cancers: List of cancers (patient can have multiple cancers of any type)
  - Each cancer can be: SolidCancer | BloodCancer | NeuroendocrineTumor
  - Most patients have 1 cancer, but some may have 2+ (e.g., prior breast cancer, now lung cancer)

## Quality Markers

**Good synthetic data**:
- Follows disease biology and natural history
- Reflects standard treatment paradigms
- Has appropriate level of detail
- Contains some uncertainty (not every field populated)
- Could plausibly exist in a real medical record

**Avoid**:
- Overly exotic or rare presentations (unless requested)
- Every possible field filled with data
- Contradictory information (responding while progressing)
- Non-standard nomenclature
- Biologically implausible scenarios

## Common Cancer Type Patterns

**Lung Cancer (NSCLC)**:
- Histology: adenocarcinoma > squamous > large cell
- Mutations: EGFR (15%), KRAS (25%), ALK (5%), ROS1 (2%), BRAF V600E (2%)
- Common mets: brain, bone, liver, adrenal, contralateral lung
- Standard therapies: platinum doublet, immunotherapy, targeted therapy (if mutation+)

**Breast Cancer**:
- Subtypes: ER+/HER2- (70%), HER2+ (15%), triple-negative (15%)
- Mutations: PIK3CA, TP53, BRCA1/2 (germline consideration)
- Common mets: bone, liver, lung, brain
- Standard therapies: endocrine (ER+), trastuzumab (HER2+), chemotherapy

**Colorectal Cancer**:
- Histology: adenocarcinoma (>95%)
- Mutations: KRAS (40%), BRAF V600E (8%), MSI-high (15%), NTRK fusions (rare)
- Common mets: liver (most common), lung, peritoneum
- Standard therapies: FOLFOX, FOLFIRI, bevacizumab, cetuximab (if RAS WT)

**AML**:
- Subtypes: varied FAB classification
- Mutations: FLT3-ITD, NPM1, IDH1/2, TP53
- Disease burden: blast percentage, cytogenetics
- Standard therapies: 7+3 induction, HiDAC, venetoclax, FLT3 inhibitors

## Response Format

Generate the complete structured PatientCancerExtraction object with realistic, internally consistent data. Think like a clinician documenting a real patient."""


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


def create_synthetic_patient_agent(model) -> Agent:
    """Create agent for generating synthetic patient data"""
    return Agent(
        model=model,
        output_type=PatientCancerExtraction,
        system_prompt=SYNTHETIC_GENERATION_SYSTEM_PROMPT,
    )


# ============================================================================
# GENERATION FUNCTION
# ============================================================================


async def generate_synthetic_patient(prompt: str) -> PatientCancerExtraction:
    """
    Generate a synthetic patient directly from a natural language prompt.

    Args:
        prompt: Natural language description of the patient to generate
                Examples:
                - "Stage IV lung adenocarcinoma with EGFR mutation and brain metastases"
                - "Newly diagnosed triple-negative breast cancer, no prior treatment"
                - "Relapsed AML with FLT3-ITD mutation after failed induction"

    Returns:
        PatientCancerExtraction: Structured patient data
    """
    model = get_default_model()
    agent = create_synthetic_patient_agent(model)

    print("=" * 80)
    print("🔬 Generating Synthetic Patient")
    print("=" * 80)
    print(f"\nPrompt: {prompt}\n")
    print("-" * 80)

    result = await agent.run(prompt)
    patient = result.output

    print("\n✅ Generated Patient:")
    print("-" * 80)
    print(patient.model_dump_json(indent=2))
    print("=" * 80)

    return patient


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


@pytest.mark.integration
async def test_generate_realistic_lung_cancer_patient():
    """Generate a realistic lung cancer patient with common characteristics"""

    prompt = """
    Generate a realistic stage IV lung adenocarcinoma patient:
    - EGFR exon 19 deletion
    - Brain and bone metastases
    - Failed first-line osimertinib
    - Now on platinum-based chemotherapy
    - Include PD-L1 status and other relevant molecular markers
    """

    patient = await generate_synthetic_patient(prompt)

    # Verify basic structure
    assert patient.cancers is not None
    assert len(patient.cancers) >= 1
    assert isinstance(patient.cancers[0], SolidCancer)

    cancer = patient.cancers[0]

    # Verify cancer type and histology
    assert cancer.cancer_type == CancerType.SOLID
    assert "adenocarcinoma" in cancer.histology.lower()

    # Verify primary tumor
    primary = cancer.primary_tumor
    assert primary.is_primary is True
    assert "lung" in primary.anatomic_location.lower()

    # Verify metastases present (requested brain and bone)
    assert primary.metastases is not None
    assert len(primary.metastases) >= 2, "Should have multiple metastases"

    # Check that we have brain and bone mets as requested
    met_locations = [m.anatomic_location.lower() for m in primary.metastases]
    has_brain = any("brain" in loc or "cerebral" in loc for loc in met_locations)
    has_bone = any(
        "bone" in loc or "vertebra" in loc or "spine" in loc for loc in met_locations
    )

    assert has_brain, f"Should have brain metastasis. Found mets: {met_locations}"
    assert has_bone, f"Should have bone metastasis. Found mets: {met_locations}"

    # Verify molecular profile - should have EGFR exon 19 deletion
    assert primary.molecular_profile is not None
    profile = primary.molecular_profile

    # Check for EGFR mutation (requested exon 19 deletion)
    assert profile.mutations is not None, "Should have mutations (EGFR requested)"
    egfr_found = any("EGFR" in m.upper() for m in profile.mutations)
    assert egfr_found, f"Should have EGFR mutation. Found: {profile.mutations}"

    # Check if exon 19 deletion is mentioned
    exon_19 = any("19" in m or "deletion" in m.lower() for m in profile.mutations)
    assert exon_19, f"Should have exon 19 deletion. Found: {profile.mutations}"

    # Should have PD-L1 status (was requested)
    assert profile.pdl1_expression is not None, "Should have PD-L1 status"

    # Verify treatment history - should include osimertinib
    assert cancer.prior_therapy_lines is not None
    assert cancer.prior_therapy_lines >= 1, "Should have prior therapy lines"
    assert cancer.prior_therapies is not None

    # Check for osimertinib (was requested as first-line)
    has_osimertinib = any("osimertinib" in t.lower() for t in cancer.prior_therapies)
    assert (
        has_osimertinib
    ), f"Should have osimertinib in prior therapies. Found: {cancer.prior_therapies}"

    # Verify disease status
    assert cancer.disease_status in [
        DiseaseStatus.ACTIVE,
        DiseaseStatus.STABLE,
        DiseaseStatus.PROGRESSING,
        DiseaseStatus.RESPONDING,
    ]

    # Print summary
    print("\n" + "=" * 80)
    print("✅ TEST PASSED - Realistic Lung Cancer Patient Generated")
    print("=" * 80)
    print(f"\nPatient Summary:")
    print(f"  Number of Cancers: {len(patient.cancers)}")
    print(f"  Cancer Type: {cancer.cancer_type.value}")
    print(f"  Histology: {cancer.histology}")
    print(f"  Primary: {primary.anatomic_location}")
    print(f"  Metastases: {len(primary.metastases)}")
    met_sites = [m.anatomic_location for m in primary.metastases]
    print(f"  Met Sites: {', '.join(met_sites[:3])}")
    print(f"  Prior Lines: {cancer.prior_therapy_lines}")
    print(f"  Prior Therapies: {', '.join(cancer.prior_therapies[:3])}")
    print(f"  Disease Status: {cancer.disease_status.value}")
    if profile.mutations:
        print(f"  Key Mutations: {', '.join(profile.mutations[:3])}")
    print("=" * 80)
