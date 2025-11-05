# Synthetic Patient Generation Service

Service for generating realistic synthetic oncology patient data with embeddings for clinical trial matching and portfolio analysis.

## Overview

This service uses LLMs to generate structured cancer patient profiles (PatientCancerExtraction) and converts them to dense narrative text with embeddings optimized for semantic similarity matching.

## Quick Start

### Using the Script (Recommended)

Edit configuration in `scripts/synthetic_patient_generation/generate_cohort.py`:

```python
COHORT_TYPE = "breast cancer"
NUM_PATIENTS = 5
EMBEDDER_TYPE = EmbedderType.OPENAI  # or EmbedderType.MEDEMBED
OUTPUT_NAME = "breast_cohort"
```

Run:
```bash
uv run -m scripts.synthetic_patient_generation.generate_cohort
```

Output: `outputs/synthetic_vae_data/breast_cohort_20251105_143022.jsonl`

### Programmatic Usage

```python
from src.services.synthetic_patient.service import SyntheticPatientService
from src.services.synthetic_patient.embedders.openai import OpenAIEmbedder
from src.config.settings import settings

# Initialize service and embedder
service = SyntheticPatientService(openai_api_key=settings.openai_api_key)
embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model="text-embedding-3-small")

# Generate cohort
cohort_data = await service.generate_patient_cohort(
    prompt="Generate 5 diverse breast cancer patients with varying subtypes and stages",
    n=5,
    embedder=embedder
)

# Save to file
output_path = service.save_cohort_to_jsonl(
    cohort_data=cohort_data,
    cohort_name="breast_cohort",
    output_subdir="synthetic_vae_data"
)
```

## Architecture

### Components

```
src/services/synthetic_patient/
├── embedders/
│   ├── base.py          # Abstract EmbeddingProvider interface
│   ├── openai.py        # OpenAI embeddings (text-embedding-3-small/large)
│   └── medembed.py      # MedEmbed medical-domain embeddings
└── service.py           # Main SyntheticPatientService class
```

### Key Classes

**`SyntheticPatientService`**
- `generate_patient_cohort()` - Generate N patients with embeddings
- `patient_to_dense_narrative()` - Convert structured data → narrative text
- `save_cohort_to_jsonl()` - Save cohort with embeddings to file

**`EmbeddingProvider`** (Abstract)
- `embed(text)` - Generate embedding vector
- `get_dimension()` - Get embedding dimensions
- `get_name()` - Get provider identifier

## Output Format

JSONL file with one patient per line:

```json
{
  "patient_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "structured_data": {...PatientCancerExtraction...},
  "embedding_text": "Patient with metastatic breast cancer...",
  "embedding": [0.123, -0.456, ...],
  "metadata": {
    "prompt": "...",
    "generated_at": "2025-11-05T14:30:22",
    "model_used": "gpt-5",
    "embedder_used": "text-embedding-3-small",
    "embedding_dimension": 1536,
    "batch_size": 5,
    "index_in_batch": 0
  }
}
```

## Swapping Embedders

### OpenAI (Default)
```python
from src.services.synthetic_patient.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key=settings.openai_api_key,
    model="text-embedding-3-small"  # or "text-embedding-3-large"
)
```

### MedEmbed (Medical-Domain)
```python
from src.services.synthetic_patient.embedders.medembed import MedEmbedder

embedder = MedEmbedder(
    model="abhinand/MedEmbed-base-v0.1"  # or small/large variants
)
```

## Features

- **Batch Generation**: Generate multiple diverse patients in one LLM call
- **Dense Narratives**: Converts structured data to paragraph-style text optimized for embeddings
- **Swappable Embedders**: Easy switching between OpenAI and MedEmbed via abstract interface
- **Unique Patient IDs**: UUIDs prevent collisions across generation runs
- **Metadata Tracking**: Full provenance (prompt, timestamp, model, embedder, dimensions)

## Use Cases

- Generate test data for trial matching algorithms
- Create evaluation datasets with known ground truth
- Build patient cohorts for portfolio gap analysis
- Test edge cases and specific molecular profiles
- Clustering analysis for underserved populations

