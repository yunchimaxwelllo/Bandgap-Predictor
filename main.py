import argparse
import subprocess
import sys
import os
from pathlib import Path

# Define paths to the notebooks
ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

NB_EXTRACTION = NOTEBOOK_DIR / "01_data_extraction.ipynb"
NB_DL_ABLATION = NOTEBOOK_DIR / "02_deep_learning_ablation.ipynb"
NB_ML_BASELINES = NOTEBOOK_DIR / "03_traditional_ml_baselines.ipynb"
NB_VISUALIZE = NOTEBOOK_DIR / "04_results_visualization.ipynb"

def run_notebook(notebook_path):
    """Executes a Jupyter Notebook headlessly and saves the output."""
    
    print(f"EXECUTING: {notebook_path.name}")
    
    
    if not notebook_path.exists():
        print(f"Error: Could not find {notebook_path}")
        sys.exit(1)

    try:
        # Uses Jupyter's CLI to run the notebook from top to bottom
        subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace", # Overwrites the notebook with the new cell outputs
            str(notebook_path)
        ], check=True)
        print(f"Successfully finished {notebook_path.name}\n")
        
    except subprocess.CalledProcessError as e:
        print(f"Execution failed for {notebook_path.name}")
        print(e)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Bandgap Prediction | CHEMENG 177/277 & MATSCI 166/176 Final Project"
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['extract', 'train_dl', 'train_ml', 'visualize', 'all'],
        help="Which part of the pipeline to run."
    )
    
    args = parser.parse_args()

    if args.mode in ['extract', 'all']:
        run_notebook(NB_EXTRACTION)

    if args.mode in ['train_dl', 'all']:
        run_notebook(NB_DL_ABLATION)

    if args.mode in ['train_ml', 'all']:
        run_notebook(NB_ML_BASELINES)
        
    if args.mode in ['visualize', 'all']:
        run_notebook(NB_VISUALIZE)

    print("Pipeline execution complete. Check the /results folder for your metrics and charts!")

if __name__ == "__main__":
    main()