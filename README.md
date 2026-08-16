# Watershed Hydrological Health & Contaminant Classification Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Natural Resources Engineering](https://img.shields.io/badge/Domain-Natural%20Resources%20%7C%20Hydrology-green.svg)](#)
[![Stack: Scikit--Learn | XGBoost | Pandas](https://img.shields.io/badge/Stack-Scikit--Learn%20%7C%20XGBoost%20%7C%20Pandas-orange.svg)](#)
[![Data: USGS NWIS | EPA WQP](https://img.shields.io/badge/Data-USGS%20NWIS%20%7C%20EPA%20WQP-blueviolet.svg)](#)

An enterprise-grade environmental data science framework that classifies freshwater basin impairment and contamination risks using multi-parameter sensor streams. The platform integrates hydrological parameters (Dissolved Oxygen, pH, Turbidity, Nitrates, Conductivity, Temperature) and benchmarks six supervised classification architectures using Stratified 10-Fold Cross-Validation.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Problem Statement & Background](#problem-statement--background)
- [System Architecture](#system-architecture)
- [Mathematical Methodology](#mathematical-methodology)
- [Data Dictionary & EPA Standards](#data-dictionary--epa-standards)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage](#execution--usage)
- [Benchmark Results & Performance](#benchmark-results--performance)
- [License](#license)

---

## Project Overview

River basin networks and reservoir catchments are vulnerable to agricultural non-point source runoff, industrial effluents, and thermal pollution. Identifying contaminated and impaired reaches requires monitoring complex geochemical parameters where interactions are non-linear.

This project delivers:
1. **Automated Geochemical Telemetry Pipeline:** Standardizes multi-sensor water quality measurements modeled on the **USGS National Water Information System (NWIS)** and **EPA Water Quality Portal (WQP)**.
2. **Multi-Model Classifier Benchmark:** Evaluates six machine learning algorithms (**SVM RBF, XGBoost, Random Forest, K-NN, Logistic Regression, Decision Trees**) with **Stratified 10-Fold Cross-Validation** to minimize false negatives on contaminated water alerts.

---

## Key Features

- **Robust Standardization:** Preprocessing pipelines using `StandardScaler` to handle multi-scale geochemical variables.
- **Stratified 10-Fold Validation:** Preserves minority impairment distribution across validation splits to ensure generalization.
- **Multi-Metric Evaluation:** Comprehensive tracking of Accuracy, Recall (Sensitivity), Precision, F1-Score, and ROC-AUC.
- **EPA Clean Water Act Alignment:** Classification boundaries calibrated against statutory environmental water quality criteria.
- **Modular OOP Architecture:** Clean class design allowing direct plug-in to real-time USGS NWIS REST endpoints.

---

## Problem Statement & Background

Water quality status classification faces several analytical challenges:

1. **Non-Linear Parameter Couplings:** Biochemical Oxygen Demand (BOD) and Dissolved Oxygen ($DO$) have inverse non-linear relationships with temperature. Threshold-based rules fail when multiple borderline parameters compound into ecological hypoxia.
2. **High Cost of False Negatives:** In environmental protection, failing to flag an impaired water body (False Negative) can lead to toxic algal blooms, fish kills, or unsafe drinking water. Thus, model selection must optimize for **Recall** and **F1-Score** rather than raw accuracy.

---

## System Architecture

```text
  ┌─────────────────────────────────┐        ┌──────────────────────────────────┐
  │   USGS NWIS In-Situ Sensors     │        │      EPA Water Quality Portal    │
  │  (pH, DO, Turbidity, Temp)      │        │  (Nitrate, Electrical Cond.)     │
  └────────────────┬────────────────┘        └─────────────────┬────────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Data Preprocessing & Robust Scaling    │
                   │           z = (x - mu) / sigma            │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Exploratory Geochemical Diagnostics     │
                   │      (Pearson & Spearman Covariance)      │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Stratified 10-Fold Split Generator      │
                   │       (Preserves Impairment Ratio)        │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │     Multi-Classifier Benchmark Suite      │
                   │  ├─ Support Vector Machine (RBF Kernel)   │
                   │  ├─ XGBoost Classifier                    │
                   │  ├─ Random Forest Classifier              │
                   │  ├─ K-Nearest Neighbors                   │
                   │  ├─ Logistic Regression                   │
                   │  └─ Decision Tree Classifier              │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   ROC-AUC & Recall Optimization Matrix    │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Production Pipeline Deployment          │
                   └───────────────────────────────────────────┘
