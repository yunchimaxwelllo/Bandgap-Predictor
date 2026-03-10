import os
import torch
import numpy as np
from pathlib import Path

# Project Paths 
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CIF_DIR = DATA_DIR / "cifs"
XRD_DIR = DATA_DIR / "xrd"
RESULTS_DIR = ROOT_DIR / "results"

# Ensure directories exist
for p in [DATA_DIR, RAW_DIR, CIF_DIR, XRD_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Global Settings
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS = os.cpu_count() or 4 #1 if kernal keeps dying

# Dataset Paths 
SPLIT_PATH = DATA_DIR / "split.json"

# Domain Constants 
BANDGAP_MIN_EV = 0.5
BANDGAP_MAX_EV = 3.0
TTH_MIN = 10.0
TTH_MAX = 80.0
WAVELENGTH = "CuKa"
FWHM_DEG = 0.1

# For Log Transforms
Y_LOG_MIN = float(np.log1p(BANDGAP_MIN_EV))
Y_LOG_MAX = float(np.log1p(BANDGAP_MAX_EV))