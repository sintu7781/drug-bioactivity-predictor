# 🧬 Drug Bioactivity Predictor

A machine learning system for predicting EGFR drug bioactivity from molecular SMILES using RDKit molecular descriptors and Morgan fingerprints.

## Project Overview

![Drug Bioactivity Predictor Architecture](docs/project-overview.png)

This project implements an end-to-end drug bioactivity prediction pipeline:

- ChEMBL data collection and curation
- SMILES validation and preprocessing
- Molecular descriptor calculation
- Morgan fingerprint generation
- Machine learning model training
- Random Forest, Logistic Regression, and XGBoost comparison
- Random and scaffold-based evaluation
- Production inference
- Streamlit web application
- Hugging Face model deployment

---

## 🚀 Live Demo

**Streamlit Application:**  
[Open the Streamlit Application](https://drug-bioactivity-predictor.streamlit.app/)

**Hugging Face Model:**  
[View the Hugging Face Model](https://huggingface.co/prime7781/drug-bioactivity-predictor)

**GitHub Repository:**  
[View Source Code](https://github.com/sintu7781/drug-bioactivity-predictor)

---

## 🎯 Project Objective

The objective is to build a reproducible machine-learning pipeline that predicts whether a compound is likely to exhibit significant activity against the **Epidermal Growth Factor Receptor (EGFR)** based on its molecular structure.

The model performs binary classification:

| Classification | Definition     |
| -------------- | -------------- |
| ACTIVE         | IC50 ≤ 1000 nM |
| INACTIVE       | IC50 > 1000 nM |

Target:

- **Target:** EGFR
- **ChEMBL ID:** CHEMBL203
- **Activity type:** IC50
- **Activity unit:** nM
- **Active threshold:** 1000 nM

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       ChEMBL         │
                    │   Experimental Data  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Download      │
                    │   & Molecule Mapping │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Data Curation       │
                    │                      │
                    │ • IC50 filtering     │
                    │ • Unit normalization │
                    │ • SMILES validation  │
                    │ • Deduplication      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering │
                    │                      │
                    │ • Molecular          │
                    │   descriptors       │
                    │ • Morgan fingerprints│
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Model Training            │
              │                                │
              │ Logistic Regression             │
              │ Random Forest                   │
              │ XGBoost                         │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    │                      │
                    │ Random split         │
                    │ Scaffold split       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Random Forest Model  │
                    │                      │
                    │ ROC-AUC: 0.9512      │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │ Hugging Face Model Hub      │
                 └─────────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Application│
                    │                      │
                    │ SMILES → Prediction  │
                    └──────────────────────┘
```
