# Fraudulent Job Posting Detection - MLOps Project

## Project Overview
An end-to-end MLOps solution for detecting fraudulent job postings using machine learning. The project aims to build and deploy an ML pipeline for real-time fraud detection to prevent scams and protect job seekers.

## Problem Statement
Scammers create fake job listings to gain access to personal information and money. Common red flags include:
- Vague job descriptions and requirements
- Missing company information

## Dataset Characteristics
- Total entries: ~17,300 job postings
- Legitimate postings: 16,500
- Fraudulent postings: ~800
- Class imbalance ratio: 20:1

### Key Features
- Employment type distribution
  - Full-time (~500 fraud cases)
  - Part-time (75 fraud cases)
  - Contract (50 fraud cases)
  - Temporary (<10 fraud cases)

- Experience level analysis
  - Entry-level (180 fraud cases)
  - Mid-Senior level (110 fraud cases)
  - Executive positions (minimal cases)

## Technical Implementation

### Feature Store Pipeline
1. **Data Cleaning & Feature Engineering**
   - Encoding nominal variables (one-hot encoding)
   - Encoding ordinal variables (label encoding)
   - Text cleaning (lowercase, HTML tag removal, URL removal)
   - Stopword removal and lemmatization

2. **Text Processing**
   - TF-IDF implementation
   - SVD for dimension reduction
   - Feature store using Feast

### Model Training
- Platform: H2O AutoML with MLflow
- Best Model: XGBoost
- Performance Metrics:
  - Precision: 97%
  - Recall: 100%
  - Macro F1-Score: 82%

### Deployment & Monitoring
- MLflow Model Serve for deployment
- Evidently AI integration for monitoring
- Drift detection capabilities

## Future Improvements
- Implement pretrained models for online inference storage
- Incorporate SMOTE for imbalanced data handling
- Migrate deployment to cloud infrastructure
