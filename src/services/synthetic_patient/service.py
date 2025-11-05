import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from src.schemas.v1_vae_patient_extraction import (
    PatientCancerExtraction,
    BatchPatientExtraction,
    SolidCancer,
    BloodCancer,
    NeuroendocrineTumor,
    Tumor,
)
from src.services.synthetic_patient.embedders.base import EmbeddingProvider
from src.utils.paths import get_output_dir
from src.config.settings import settings


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

**BATCH DIVERSITY** (when generating multiple patients):
- Ensure variation across the cohort in molecular profiles, disease stages, and treatment histories
- For breast cancer: include mix of ER+/HER2-, HER2+, and triple-negative subtypes
- For lung cancer: vary EGFR, KRAS, ALK, ROS1 mutations and wild-type patients
- Vary prior therapy lines (0-4+ lines)
- Mix disease statuses (newly diagnosed, responding, stable, progressing)
- Include different age ranges and performance statuses

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
# SERVICE CLASS
# ============================================================================


class SyntheticPatientService:
    """
    Service for generating synthetic patient cancer data with embeddings.

    This service orchestrates:
    - LLM-based generation of structured patient data
    - Conversion to dense narrative text for embeddings
    - Embedding generation via swappable providers
    - JSONL file output with all data
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the synthetic patient service.

        Args:
            openai_api_key: OpenAI API key (defaults to settings if not provided)
        """
        self.openai_api_key = openai_api_key or settings.openai_api_key
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for patient generation")

    def _create_generation_agent(self, n: int) -> Agent:
        """
        Create pydantic-ai agent for generating synthetic patients.

        Args:
            n: Number of patients to generate (used for output type)

        Returns:
            Configured Agent instance
        """
        model_settings = OpenAIResponsesModelSettings(max_tokens=25000)
        model = OpenAIResponsesModel(
            "gpt-5",
            provider=OpenAIProvider(api_key=self.openai_api_key),
            settings=model_settings,
        )

        # Use batch output if n > 1, single if n == 1
        output_type = BatchPatientExtraction if n > 1 else PatientCancerExtraction

        return Agent(
            model=model,
            output_type=output_type,
            system_prompt=SYNTHETIC_GENERATION_SYSTEM_PROMPT,
        )

    def patient_to_dense_narrative(self, patient: PatientCancerExtraction) -> str:
        """
        Convert structured patient data to dense narrative text for embeddings.

        This creates a paragraph-style description optimized for semantic similarity
        matching with trial criteria.

        Args:
            patient: PatientCancerExtraction object

        Returns:
            Dense narrative text string
        """
        if not patient.cancers or len(patient.cancers) == 0:
            return "No cancer information available."

        narratives = []

        for cancer in patient.cancers:
            parts = []

            # Cancer type and histology
            if isinstance(cancer, SolidCancer):
                parts.append(f"Patient with {cancer.histology}")

                # Primary tumor location
                parts.append(
                    f"Primary tumor located in {cancer.primary_tumor.anatomic_location}."
                )

                # Tumor size if available
                if cancer.primary_tumor.size_mm:
                    parts.append(f"Tumor size {cancer.primary_tumor.size_mm}mm.")

                # Metastases
                if cancer.primary_tumor.metastases:
                    met_locs = [
                        m.anatomic_location for m in cancer.primary_tumor.metastases
                    ]
                    parts.append(
                        f"Metastatic disease present in {', '.join(met_locs)}."
                    )

                    # Add sizes for metastases if available
                    sized_mets = [
                        f"{m.anatomic_location} ({m.size_mm}mm)"
                        for m in cancer.primary_tumor.metastases
                        if m.size_mm
                    ]
                    if sized_mets:
                        parts.append(f"Metastatic lesions: {', '.join(sized_mets)}.")

                # Molecular profile from primary tumor
                profile = cancer.primary_tumor.molecular_profile

            elif isinstance(cancer, BloodCancer):
                parts.append(f"Patient with {cancer.subtype}")

                # Disease burden
                if cancer.disease_burden:
                    burden_strs = [
                        f"{k}: {v}" for k, v in cancer.disease_burden.items()
                    ]
                    parts.append(f"Disease burden: {', '.join(burden_strs)}.")

                profile = cancer.molecular_profile

            elif isinstance(cancer, NeuroendocrineTumor):
                parts.append(f"Patient with neuroendocrine tumor, {cancer.grade}")
                parts.append(
                    f"Primary tumor located in {cancer.primary_tumor.anatomic_location}."
                )

                if cancer.functional is not None:
                    parts.append(
                        f"Tumor is {'functional' if cancer.functional else 'non-functional'}."
                    )

                if cancer.sstr_expression:
                    parts.append(f"SSTR status: {cancer.sstr_expression}.")

                # Metastases
                if cancer.primary_tumor.metastases:
                    met_locs = [
                        m.anatomic_location for m in cancer.primary_tumor.metastases
                    ]
                    parts.append(f"Metastatic disease in {', '.join(met_locs)}.")

                profile = cancer.primary_tumor.molecular_profile

            # Add molecular profile information
            if profile:
                mol_parts = []

                if profile.mutations:
                    mol_parts.append(f"Mutations: {', '.join(profile.mutations)}")

                if profile.amplifications:
                    mol_parts.append(
                        f"Amplifications: {', '.join(profile.amplifications)}"
                    )

                if profile.deletions:
                    mol_parts.append(f"Deletions: {', '.join(profile.deletions)}")

                if profile.fusions:
                    mol_parts.append(f"Fusions: {', '.join(profile.fusions)}")

                if profile.pdl1_expression:
                    mol_parts.append(f"PD-L1: {profile.pdl1_expression}")

                if profile.her2_expression:
                    mol_parts.append(f"HER2: {profile.her2_expression}")

                if profile.hormone_receptors:
                    mol_parts.append(
                        f"Hormone receptors: {', '.join(profile.hormone_receptors)}"
                    )

                if profile.msi_status:
                    mol_parts.append(f"MSI: {profile.msi_status}")

                if profile.tmb:
                    mol_parts.append(f"TMB: {profile.tmb}")

                if profile.hrd_status:
                    mol_parts.append(f"HRD: {profile.hrd_status}")

                if mol_parts:
                    parts.append(f"Molecular profile: {'; '.join(mol_parts)}.")

            # Treatment history
            if cancer.prior_therapy_lines is not None:
                parts.append(f"Prior lines of therapy: {cancer.prior_therapy_lines}.")

            if cancer.prior_therapies:
                therapies_str = ", ".join(
                    cancer.prior_therapies[:5]
                )  # Limit to first 5
                if len(cancer.prior_therapies) > 5:
                    therapies_str += f", and {len(cancer.prior_therapies) - 5} others"
                parts.append(f"Prior treatments: {therapies_str}.")

            # Disease status
            parts.append(f"Disease status: {cancer.disease_status.value}.")

            narratives.append(" ".join(parts))

        return " ".join(narratives)

    async def generate_patient_cohort(
        self, prompt: str, n: int, embedder: EmbeddingProvider
    ) -> list[dict]:
        """
        Generate a cohort of synthetic patients with embeddings.

        Args:
            prompt: Natural language prompt describing patients to generate
            n: Number of patients to generate
            embedder: Embedding provider to use

        Returns:
            List of dicts containing patient data, narrative, and embeddings
        """
        print("=" * 80)
        print(f"🔬 Generating {n} Synthetic Patient(s)")
        print("=" * 80)
        print(f"\nPrompt: {prompt}\n")
        print("-" * 80)

        # Create agent and generate
        agent = self._create_generation_agent(n)
        result = await agent.run(prompt)

        # Extract patients from result
        if n == 1:
            patients = [result.output]
        else:
            patients = result.output.patients

        print(f"\n✅ Generated {len(patients)} patient(s)")
        print("-" * 80)

        # Process each patient
        cohort_data = []
        for idx, patient in enumerate(patients):
            print(f"\nProcessing patient {idx}...")

            # Generate unique patient ID
            patient_id = str(uuid.uuid4())

            # Convert to narrative
            narrative = self.patient_to_dense_narrative(patient)
            print(f"  Patient ID: {patient_id}")
            print(f"  Narrative: {len(narrative)} characters")

            # Generate embedding
            embedding = embedder.embed(narrative)
            print(f"  Embedding: {len(embedding)} dimensions")

            # Create data dict
            patient_data = {
                "patient_id": patient_id,
                "structured_data": patient.model_dump(),
                "embedding_text": narrative,
                "embedding": embedding,
                "metadata": {
                    "prompt": prompt,
                    "generated_at": datetime.now().isoformat(),
                    "model_used": "gpt-5",
                    "embedder_used": embedder.get_name(),
                    "embedding_dimension": embedder.get_dimension(),
                    "batch_size": n,
                    "index_in_batch": idx,
                },
            }

            cohort_data.append(patient_data)

        print("\n" + "=" * 80)
        print(f"✅ Cohort generation complete: {len(cohort_data)} patients")
        print("=" * 80)

        return cohort_data

    def save_cohort_to_jsonl(
        self,
        cohort_data: list[dict],
        cohort_name: str,
        output_subdir: str = "synthetic_vae_data",
    ) -> str:
        """
        Save cohort data to JSONL file.

        Args:
            cohort_data: List of patient data dicts
            cohort_name: Base name for the output file
            output_subdir: Subdirectory within outputs/ (default: synthetic_vae_data)

        Returns:
            Path to the saved file as string
        """
        # Get output directory
        output_dir = get_output_dir(output_subdir)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{cohort_name}_{timestamp}.jsonl"
        output_path = output_dir / filename

        # Write JSONL file (one patient per line)
        with open(output_path, "w") as f:
            for patient_data in cohort_data:
                f.write(json.dumps(patient_data) + "\n")

        print(f"\n💾 Saved cohort to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        print(f"   Patients: {len(cohort_data)}")

        return str(output_path)
