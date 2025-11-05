from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

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


class EmbedderType(str, Enum):
    """Type of embedder to use for generation"""

    OPENAI = "openai"
    MEDEMBED = "medembed"


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
# BATCH EXTRACTION
# ============================================================================


class BatchPatientExtraction(BaseModel):
    """Multiple patients generated in a single batch"""

    patients: list[PatientCancerExtraction] = Field(
        description="List of patient cancer extractions"
    )

