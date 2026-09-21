"""Central place for every filesystem path this repo touches.

Edit the defaults below, or override any of them with the matching
environment variable, without needing to touch any training/eval script.
"""
import os

# Directory containing the raw MIMIC-CXR-JPG images.
IMAGE_DIR = os.environ.get("ADAPTER_IMAGE_DIR", "/path/to/mimic-cxr-jpg")

# CSV with one row per image (ImagePath, Split, PatientID, Dataset, Race, Sex,
# View, Age, + pathology label columns). See README for the exact schema.
METADATA_CSV = os.environ.get("ADAPTER_METADATA_CSV", "./data/chai_cxr_master.csv")

# Where extracted Rad-DINO CLS embeddings (memmap .dat files) are cached.
EMBEDDING_CACHE_DIR = os.environ.get("ADAPTER_EMBEDDING_CACHE_DIR", "./cache/raddino")

# Where trained model checkpoints are written/read.
CHECKPOINT_DIR = os.environ.get("ADAPTER_CHECKPOINT_DIR", "./checkpoints")

# Where evaluation CSVs and figures are written.
OUTPUT_DIR = os.environ.get("ADAPTER_OUTPUT_DIR", "./outputs")
