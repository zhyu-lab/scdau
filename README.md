# scDAU

A feature-decoupled deep learning method for single-cell cross-modality translation

## Requirements

* Python 3.9+.

# Installation

## Clone repository

First, download scDAU from github and change to the directory:

```bash
git https://github.com/zhyu-lab/scdau
cd scdau
```

## Create conda environment (optional)

Create a new environment named "scdau":

```bash
conda create --name scdau python=3.9
```

Then activate it:

```bash
conda activate scdau
```

## Install requirements

```bash
python -m pip install -r requirements.txt
```

# Usage

## Train the scDAU model

### Step1: Prepare the input data in h5ad format.

We use RNA and ATAC (or ADT) data stored in .h5ad files as input.

### Step 2: Run scDAU

The paired_RNA_ATAC contains code for processing paired RNA and ATAC data. The paired_RNA_ADT contains code for processing paired RNA and ADT data

The arguments to run scDAU are as follows:

| Parameter     | Description                                                | Possible values             |
| ------------- | ---------------------------------------------------------- | --------------------------- |
| --rna_path    | input file containing RNA data                             | Ex: /path/to/RNA.h5ad       |
| --atac_path   | input file containing ATAC(ADT)data                        | Ex: /path/to/ATAC(ADT).h5ad |
| --result_path | a directory to save results                                | Ex: /path/to/results        |
| --ae_epochs   | number of epoches to train the ae                          | Ex: epochs=250              |
| --dae_epochs  | number of epoches to train the dae                         | Ex: epochs=120              |
| --unet_epochs | number of epoches to train the unet                        | Ex: epochs=20               |
| --bl          | weight for dae reconstruction loss                         | Ex: bl=1                    |
| --fold        | determine the batch splits for five-fold cross-validation. | Ex: fold=0                  |
| --seed        | random seed (for reproduction of the results)              | Ex: seed=1                  |

Example:

```bash
python paired_RNA_ATAC.py 
python paired_RNA_ADT.py
```

# Contact

If you have any questions, please contact 12024130911@stu.nxu.edu.cn.