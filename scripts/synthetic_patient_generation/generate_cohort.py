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

from src.schemas.v1_vae_patient_extraction import EmbedderType
from src.services.synthetic_patient.service import SyntheticPatientService
from src.services.synthetic_patient.embedders.openai import OpenAIEmbedder
from src.services.synthetic_patient.embedders.medembed import MedEmbedder
from src.config.settings import settings


# ============================================================================
# CONFIGURATION - Edit these variables
# ============================================================================

# Type of cancer patients to generate
COHORT_TYPE = "breast cancer"

# Number of patients to generate
NUM_PATIENTS = 2

# Detailed prompt for generation
PROMPT = """
Generate 2 diverse breast cancer patients with the following characteristics:
- Mix of subtypes: ER+/HER2-, HER2+, triple-negative
- Varying stages (II-IV)
- Different molecular profiles (PIK3CA, TP53, BRCA1/2 mutations)
- Range of prior therapy lines (0-3)
- Mix of disease statuses (responding, stable, progressing)
- Include metastatic sites where appropriate (bone, liver, lung, brain)
"""

# Embedder type: EmbedderType.OPENAI or EmbedderType.MEDEMBED
EMBEDDER_TYPE = EmbedderType.OPENAI

# OpenAI embedding model (if using OPENAI embedder)
OPENAI_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"

# MedEmbed model (if using MEDEMBED embedder)
MEDEMBED_MODEL = "abhinand/MedEmbed-base-v0.1"

# Output name for JSONL file (will be timestamped)
OUTPUT_NAME = "breast_cohort"


# ============================================================================
# MAIN FUNCTION
# ============================================================================


async def main():
    """Generate synthetic patient cohort and save to JSONL."""

    print("\n" + "=" * 80)
    print("🏥 SYNTHETIC PATIENT COHORT GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Cohort Type: {COHORT_TYPE}")
    print(f"  Number of Patients: {NUM_PATIENTS}")
    print(f"  Embedder: {EMBEDDER_TYPE.value}")
    print(f"  Output Name: {OUTPUT_NAME}")
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
        prompt=PROMPT, n=NUM_PATIENTS, embedder=embedder
    )

    # Save to JSONL
    output_path = service.save_cohort_to_jsonl(
        cohort_data=cohort_data,
        cohort_name=OUTPUT_NAME,
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
