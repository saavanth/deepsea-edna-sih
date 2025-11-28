# Deepsea eDNA – Taxonomy & Abundance (Baseline v1)

This repo contains the **baseline pipeline** for classifying deep-sea eDNA 18S sequences
into a full taxonomic hierarchy and estimating abundance per taxon.

## 1. Tech Stack (Baseline)

- Python 3.10
- PyTorch (simple CNN baseline)
- scikit-learn
- BLAST+ (only needed to build training data, not for running the model)
- Conda for environment management

## 2. Getting Started

git clone https://github.com/saavanth/deepsea-edna-sih.git
cd deepsea-edna-sih

conda create -n edna-sih python=3.10 -y
conda activate edna-sih

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn tqdm biopython


3. Pipeline Overview (Baseline)
A. Build Training Data (one-time, already done for PR2 subset)

Extract a sample from BLAST DB:

src/pipeline/extract_from_blastdb.py

Run BLASTN against the reference DB:

src/pipeline/run_blastn.py

Convert BLAST hits → training CSV with taxonomy:

src/pipeline/build_training_dataset.py

Attach nucleotide sequences:

src/pipeline/add_sequences_to_training.py

Output: data/pr2_training_6000_with_seq.csv
