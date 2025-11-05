"""
Synthetic Patient Cohort Generation Script

This script generates synthetic patient cancer data with embeddings for use in
clinical trial matching and portfolio analysis.

Usage:
    uv run -m scripts.synthetic_patient_generation.generate_cohort

Configuration:
    Edit the variables below to customize the generation.
"""

import asyncio
from enum import Enum
from pydantic import BaseModel, Field

from src.schemas.v1_vae_patient_extraction import EmbedderType
from src.services.synthetic_patient.service import SyntheticPatientService
from src.services.synthetic_patient.embedders.openai import OpenAIEmbedder
from src.services.synthetic_patient.embedders.medembed import MedEmbedder
from src.config.settings import settings


# ============================================================================
# COHORT TYPE DEFINITIONS
# ============================================================================


class CohortType(str, Enum):
    """Available cancer cohort types."""

    BREAST = "breast"
    LUNG = "lung"
    COLORECTAL = "colorectal"


class CohortConfig(BaseModel):
    """Configuration for a specific cancer cohort type."""

    cancer_type: str = Field(description="Full name of the cancer type")
    default_prompt: str = Field(description="Default prompt template for generation")
    default_output_prefix: str = Field(description="Default prefix for output file")


# Predefined cohort configurations
COHORT_CONFIGS: dict[CohortType, CohortConfig] = {
    CohortType.BREAST: CohortConfig(
        cancer_type="breast cancer",
        default_prompt="""
Generate diverse breast cancer patients with the following characteristics:
- Mix of subtypes: ER+/HER2-, HER2+, triple-negative
- Varying stages (II-IV)
- Range of prior therapy lines (0-3)
- Mix of disease statuses (responding, stable, progressing)
- Include metastatic sites where appropriate (bone, liver, lung, brain)
- Various performance statuses (ECOG 0-2)

IMPORTANT - INCLUDE MOLECULAR DATA:
Most patients should have populated molecular_profile fields with actual mutation/biomarker data:
- mutations: Include specific mutations like "PIK3CA H1047R", "TP53 R248Q", "BRCA1 c.5266dupC", "BRCA2 6174delT"
- hormone_receptors: Include values like "ER positive 90%", "PR positive 70%", "PR negative"
- her2_expression: Include values like "IHC 3+", "IHC 2+ FISH amplified", "IHC 0 (negative)"
- For triple-negative: Include at least mutations (PIK3CA, TP53) and set hormone_receptors and her2_expression to negative
It's okay if 1-2 patients have incomplete molecular data, but the majority should have comprehensive profiles.

IMPORTANT: For each tumor and metastasis, you MUST include ALL required fields:
- Primary tumor: is_primary = true
- Each metastasis: is_primary = false (this field is REQUIRED for every metastasis)

Ensure there is realistic but actual variation in patient characteristics.
""",
        default_output_prefix="breast_cohort",
    ),
    CohortType.LUNG: CohortConfig(
        cancer_type="lung cancer",
        default_prompt="""
Generate diverse lung cancer patients with the following characteristics:
- Mix of histologies: adenocarcinoma, squamous cell, small cell
- Varying stages (IIB-IV)
- Range of prior therapy lines (0-3)
- Mix of disease statuses (responding, stable, progressing)
- Include metastatic sites where appropriate (brain, bone, liver, adrenal, contralateral lung)
- Various smoking histories (never, former, current smokers)
- Various performance statuses (ECOG 0-2)

IMPORTANT - INCLUDE MOLECULAR DATA:
Most patients should have populated molecular_profile fields with actual mutation/biomarker data:
- For adenocarcinoma: Include mutations like "EGFR L858R", "EGFR exon 19 deletion", "KRAS G12C", "KRAS G12D", or fusions like "ALK-EML4", "ROS1 fusion", "RET fusion", "NTRK fusion"
- For all NSCLC: Include pdl1_expression like "TPS 90%", "TPS 50%", "TPS 5%", "CPS 20", or "negative"
- mutations: Include specific mutations like "BRAF V600E", "MET exon 14 skipping", "ERBB2 exon 20 insertion"
- amplifications: Include values like "MET amplification", "FGFR1 amplification" where appropriate
- tmb: Include values like "TMB-high 18 mut/Mb", "TMB-low 3 mut/Mb"
It's okay if 1-2 patients have incomplete molecular data, but the majority should have comprehensive profiles.

IMPORTANT: For each tumor and metastasis, you MUST include ALL required fields:
- Primary tumor: is_primary = true
- Each metastasis: is_primary = false (this field is REQUIRED for every metastasis)

Ensure there is realistic but actual variation in patient characteristics.
""",
        default_output_prefix="lung_cohort",
    ),
    CohortType.COLORECTAL: CohortConfig(
        cancer_type="colorectal cancer",
        default_prompt="""
Generate diverse colorectal cancer patients with the following characteristics:
- Mix of primary sites: colon, rectum, with different locations (right-sided, left-sided, rectal)
- Varying stages (II-IV)
- Range of prior therapy lines (0-3)
- Mix of disease statuses (responding, stable, progressing)
- Include metastatic sites where appropriate (liver, lung, peritoneum, lymph nodes)
- Various performance statuses (ECOG 0-2)

IMPORTANT - INCLUDE MOLECULAR DATA:
Most patients should have populated molecular_profile fields with actual mutation/biomarker data:
- mutations: Include specific mutations like "KRAS G12D", "KRAS G12V", "NRAS Q61K", "BRAF V600E", "PIK3CA H1047R"
- msi_status: Include values like "MSI-high", "MSS (microsatellite stable)", "dMMR"
- her2_expression: Include values like "IHC 3+", "IHC 2+ FISH amplified", "negative"
- amplifications: For HER2+ patients include "HER2 amplification"
- tmb: Include values like "TMB-high 45 mut/Mb" for MSI-high, "TMB-low 2 mut/Mb" for MSS
It's okay if 1-2 patients have incomplete molecular data, but the majority should have comprehensive profiles.

IMPORTANT: For each tumor and metastasis, you MUST include ALL required fields:
- Primary tumor: is_primary = true
- Each metastasis: is_primary = false (this field is REQUIRED for every metastasis)

Ensure there is realistic but actual variation in patient characteristics.
""",
        default_output_prefix="colorectal_cohort",
    ),
}


# ============================================================================
# CONFIGURATION - Edit these variables
# ============================================================================

# Select cohort type: CohortType.BREAST, CohortType.LUNG, or CohortType.COLORECTAL
COHORT_TYPE: CohortType = CohortType.LUNG

# Number of patients to generate
NUM_PATIENTS = 5

# Custom prompt (optional) - if None, uses the default prompt for the cohort type
# Set this to override the default prompt with your own
CUSTOM_PROMPT: str | None = None

# Embedder type: EmbedderType.OPENAI or EmbedderType.MEDEMBED
EMBEDDER_TYPE = EmbedderType.OPENAI

# OpenAI embedding model (if using OPENAI embedder)
OPENAI_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"

# MedEmbed model (if using MEDEMBED embedder)
MEDEMBED_MODEL = "abhinand/MedEmbed-base-v0.1"

# Output name for JSONL file (optional) - if None, uses default prefix + timestamp
# Set this to override the default output name
CUSTOM_OUTPUT_NAME: str | None = None


# ============================================================================
# MAIN FUNCTION
# ============================================================================


async def main():
    """Generate synthetic patient cohort and save to JSONL."""

    # Get cohort configuration
    cohort_config = COHORT_CONFIGS[COHORT_TYPE]

    # Determine prompt and output name
    prompt = (
        CUSTOM_PROMPT if CUSTOM_PROMPT is not None else cohort_config.default_prompt
    )
    output_name = (
        CUSTOM_OUTPUT_NAME
        if CUSTOM_OUTPUT_NAME is not None
        else cohort_config.default_output_prefix
    )

    # Ensure prompt includes the correct number of patients
    if "{n}" not in prompt:
        prompt = f"Generate {NUM_PATIENTS} patients.\n\n" + prompt

    print("\n" + "=" * 80)
    print("🏥 SYNTHETIC PATIENT COHORT GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Cohort Type: {cohort_config.cancer_type}")
    print(f"  Number of Patients: {NUM_PATIENTS}")
    print(f"  Prompt: {'Custom' if CUSTOM_PROMPT else 'Default'}")
    print(f"  Embedder: {EMBEDDER_TYPE.value}")
    print(f"  Output Name: {output_name}")
    print("=" * 80 + "\n")

    # Initialize service
    service = SyntheticPatientService(openai_api_key=settings.openai_api_key)

    # Create embedder based on type
    if EMBEDDER_TYPE == EmbedderType.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured in settings")
        embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model=OPENAI_MODEL)
        print(f"✅ Using OpenAI embedder: {OPENAI_MODEL}")
    elif EMBEDDER_TYPE == EmbedderType.MEDEMBED:
        embedder = MedEmbedder(model=MEDEMBED_MODEL)
        print(f"✅ Using MedEmbed: {MEDEMBED_MODEL}")
    else:
        raise ValueError(f"Unknown embedder type: {EMBEDDER_TYPE}")

    print(f"   Embedding dimension: {embedder.get_dimension()}\n")

    # Generate cohort
    cohort_data = await service.generate_patient_cohort(
        prompt=prompt, n=NUM_PATIENTS, embedder=embedder
    )

    # Save to JSONL
    output_path = service.save_cohort_to_jsonl(
        cohort_data=cohort_data,
        cohort_name=output_name,
        output_subdir="synthetic_vae_data",
    )

    # Print summary
    print("\n" + "=" * 80)
    print("📊 GENERATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully generated {len(cohort_data)} patients")
    print(f"📁 Saved to: {output_path}")
    print(f"💾 File contains:")
    print(f"   - Structured patient data (Pydantic models)")
    print(f"   - Dense narrative text for embeddings")
    print(f"   - {embedder.get_dimension()}-dimensional embeddings")
    print(f"   - Generation metadata")
    print("\n" + "=" * 80)
    print("✅ COMPLETE - Ready to send to your brother!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
