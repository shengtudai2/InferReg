# InferReg

## Introduction

InferReg is a GNN-based tool designed for the inference of gene regulatory networks from high-throughput genomic data. It leverages the latest advances in machine learning to provide users with an accurate and efficient way to predict regulatory interactions between transcription factors (TFs) and target genes. 

## Usage

Here's a step-by-step guide on how to use InferReg:

### environment setup

To ensure successful use of this project, which relies on Python3, pandas, PyTorch, and PyTorch Geometric (PyG), it is crucial to properly set up these packages in your local environment. To facilitate a seamless installation process, please refer to the following comprehensive guides tailored for each library:
1. Python3 >= 3.8.
2. pandas. pandas can be easily installed using pip, Python's package installer.
3. PyTorch. To install PyTorch with support for your specific hardware (CPU or GPU) and operating system, visit the official installation page: https://pytorch.org/get-started/locally/
4. PyTorch Geometric (PyG): For installing PyG, a library dedicated to deep learning on irregularly structured data such as graphs, consult its official documentation at: https://pytorch-geometric.readthedocs.io/en/latest/index.html


### Data Preparation

Prepare the following files in the `code/data/raw` directory:

- `gene_expression.csv`: A CSV file containing gene expression data with gene names in rows and sample names in columns.
- `edges.tsv`: A TSV file with edge information, where each row represents a TF and its target genes.
- `pos_edges.tsv`: A TSV file containing positive edge information validated by ChIP-Seq data.

Sample data files(Arabidopsis) are provided in the `https://doi.org/10.5281/zenodo.10960938`

### Data Preprocessing

Navigate to the `code/src/data_preprocess` directory and run `python data_loader.py` to load the input data.

### Model Training

To train the model, execute `python model_fit.py` from the `code` directory.

### Prediction

To predict regulatory interactions, run `python predict.py` from the `code` directory. The predicted network will be saved in the `data/predicted/at_edges.tsv` file.
