# 🧬 Drug Bioactivity Predictor

<p align="center">

An end-to-end cheminformatics and machine-learning pipeline for predicting **EGFR bioactivity** from molecular SMILES.

</p>

<p align="center">

**ChEMBL → Data Curation → RDKit → Molecular Features → ML → Scaffold Validation → Hugging Face → Streamlit**

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-red.svg)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)](https://www.rdkit.org/)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](#testing)
[![Ruff](https://img.shields.io/badge/code%20quality-Ruff-purple.svg)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-to%20be%20added-lightgrey.svg)](#license)

</p>

---

## 🌐 Application Preview

![Drug Bioactivity Predictor](docs/streamlit-preview.png)

---

## 📌 Overview

**Drug Bioactivity Predictor** is a research-oriented machine-learning application that predicts whether a chemical compound is likely to be **ACTIVE** or **INACTIVE** against the **Epidermal Growth Factor Receptor (EGFR)** using its molecular structure represented as a SMILES string.

The project combines:

- ChEMBL bioactivity data
- Data curation and quality control
- RDKit cheminformatics
- Molecular descriptors
- Morgan fingerprints
- Classical machine-learning models
- Cross-validation
- Random train/test evaluation
- Bemis-Murcko scaffold validation
- Production inference
- Automated testing
- Code-quality checks
- GitHub Actions CI
- Hugging Face model hosting
- Streamlit deployment

The final selected model is a **Random Forest classifier**.

---

# 🎯 Problem Statement

Drug discovery involves evaluating large chemical libraries against biological targets.

Experimental measurements such as **IC50** are valuable but can be expensive and time-consuming to obtain. Computational models can be used as an initial prioritization mechanism to identify compounds that may warrant further investigation.

This project addresses the following problem:

> **Given a molecular SMILES representation, predict whether the compound is likely to exhibit activity against EGFR based on patterns learned from curated experimental IC50 measurements.**

The prediction is formulated as a binary classification problem.

```text
IC50 <= 1000 nM  → ACTIVE
IC50 > 1000 nM   → INACTIVE
```

## 🌐 Streamlit Application

https://drug-bioactivity-predictor.streamlit.app/
