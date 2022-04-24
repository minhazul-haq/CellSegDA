# CellSegDA
This repo holds code for [Adversarial Domain Adaptation for Cell Segmentation](http://proceedings.mlr.press/v121/haq20a.html)

## Usage

### 1. Environment

Run following commands to prepare environment with all dependencies.

```bash
conda env create -f environment.yml
conda activate cellseg-da
```

### 2. Dataset

Please send an email to mohammadminhazu.haq AT mavs.uta.edu to request the datasets.

### 3. Training

#### CellSegUDA
Run following script to train CellSegUDA model with KIRC as source dataset and TNBC as target dataset.

```bash
python train_cellseg_uda.py --source_dataset kirc --target_dataset tnbc
```

#### CellSegSSDA
Run following script to train CellSegSSDA model with KIRC as source dataset, TNBC as target dataset, and 25% labels of target dataset.

```bash
python train_cellseg_ssda.py --source_dataset kirc --target_dataset tnbc --target_label_percentage 25
```

### 4. Prediction and Evaluation

Run following script to predict the segmentation masks on kirc-test images, and then evaluate the predictions.

```bash
python predict_and_evaluate.py --test_dataset kirc --model_path path_to_best_model
```

## Citations

```bibtex
@inproceedings{haq2020adversarial,
  title={Adversarial domain adaptation for cell segmentation},
  author={Haq, Mohammad Minhazul and Huang, Junzhou},
  booktitle={Medical Imaging with Deep Learning},
  pages={277--287},
  year={2020},
  organization={PMLR}
}
```
