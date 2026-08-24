# 🧬 Drug Bioactivity Predictor

An end-to-end machine learning application for predicting **EGFR inhibitor bioactivity** from molecular SMILES strings.

The project uses experimentally measured **IC50 activity data from ChEMBL**, performs molecular data curation and feature engineering with RDKit, trains multiple machine-learning classifiers, evaluates generalization using both random and scaffold splits, and deploys the final Random Forest model through a Streamlit web application.

---

## 🚀 Live Demo

**Streamlit Application:**  
https://drug-bioactivity-predictor.streamlit.app/

**Hugging Face Model:**  
https://huggingface.co/prime7781/drug-bioactivity-predictor

**GitHub Repository:**  
https://github.com/sintu7781/drug-bioactivity-predictor

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
