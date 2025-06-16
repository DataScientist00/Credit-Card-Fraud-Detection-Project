# Credit Card Fraud Detection

## Watch the Video 📺

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Video-red?logo=youtube&logoColor=white&style=for-the-badge)](https://youtu.be/RCwazMVEIXU)

## Overview
This project is focused on building a machine learning model to detect fraudulent credit card transactions. The goal is to minimize financial losses for cardholders and financial institutions by accurately classifying transactions as fraudulent or legitimate.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Features](#features)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction
Credit card fraud is a significant threat in today's digital world. This project leverages machine learning techniques to build a robust fraud detection system. By analyzing transaction patterns and using predictive analytics, the model classifies transactions as either fraudulent or legitimate.

### Objectives
- Detect fraudulent credit card transactions with high accuracy.
- Minimize false positives to avoid impacting genuine users.
- Provide an easy-to-use framework for testing and deploying fraud detection models.

---

## Dataset
### Source
The dataset used in this project is the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Description
- Contains transactions made by European cardholders in September 2013.
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Features:
  - 28 anonymized numerical features (`V1` to `V28`).
  - `Time`: Time elapsed between the first transaction and the current one.
  - `Amount`: Transaction amount.
  - `Class`: Label (1 for fraud, 0 for legitimate).

---

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning algorithms and evaluation metrics.
- **Matplotlib / Seaborn**: Data visualization.
- **Jupyter Notebook**: Development and experimentation.

---

## Installation
### Prerequisites
Ensure Python (>= 3.8) is installed on your system.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset and place it in the `data/` directory.

---

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `fraud_detection.ipynb` to:
   - Load and preprocess the data.
   - Train and evaluate machine learning models.
   - Visualize results.
---

## Features
- **Data Preprocessing**:
  - Handle class imbalance using techniques like SMOTE or undersampling.
  - Normalize numerical features.
- **Model Training**:
  - Algorithms: Logistic Regression, Random Forest, Decision Tree
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion matrix visualization.
- **Deployment Ready**:
  - Save and load models using joblib for real-world applications.

---

## Results
- **Model Performance**:
  - Regression(SMOTE): `99.92%`
  - Decision Tree(SMOTE): `99.89%`
  - Random Forest(SMOTE): `99.94%`

---

## Contributing
We welcome contributions to improve this project! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a pull request.

---

## Acknowledgements
- Thanks to [MLG-ULB](https://www.ulb.ac.be/) for providing the dataset.
- Inspired by open-source contributions and Kaggle kernels.

---

## Contact
For questions or support, please reach out:
- **Email**: nikzmishra@gmail.com
- **Youtube**: [Youtube Channel](https://www.youtube.com/@NeuralArc00/videos)
