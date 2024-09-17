# K-Nearest Neighbors on Vertebral Column Data Set

## Project Overview

This project focuses on applying various distance metrics and K-Nearest Neighbors (KNN) techniques to classify patients' vertebral column conditions using a biomedical dataset. The dataset includes biomechanical attributes derived from the pelvis and lumbar spine. The task is to perform binary classification, distinguishing between **Normal** (NO=0) and **Abnormal** (AB=1) cases.

The project explores different distance metrics, KNN configurations, and voting strategies, while evaluating the impact of training set size on classification accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [How to Run](#how-to-run)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [KNN Classification](#knn-classification)
- [Distance Metrics](#distance-metrics)
- [Weighted Voting](#weighted-voting)
- [Results](#results)
- [License](#license)

## Dataset

The Vertebral Column Data Set was built by Dr. Henrique da Mota during his medical residency in Lyon, France. The dataset contains six biomechanical attributes of the pelvis and lumbar spine:

- Pelvic incidence
- Pelvic tilt
- Lumbar lordosis angle
- Sacral slope
- Pelvic radius
- Grade of spondylolisthesis

The dataset includes class labels representing conditions:
- DH (Disk Hernia)
- SL (Spondylolisthesis)
- NO (Normal)
- AB (Abnormal)

For this project, we focus on a binary classification: NO=0 and AB=1.

## Features

1. **Pre-processing & Exploratory Data Analysis**:
    - Scatterplots of independent variables with color representing the binary classes (0 or 1).
    - Boxplots of independent variables, categorized by classes.
    - Training and test split: The first 70 rows of Class 0 and the first 140 rows of Class 1 are used for training, with the remainder as the test set.

2. **KNN Classification**:
    - **KNN with Euclidean metric**: Applied on the dataset with majority voting.
    - **Train and test error analysis**: Train and test errors are plotted for varying values of k from {208, 205, ..., 1}.
    - **Confusion matrix**: Metrics such as True Positive Rate, True Negative Rate, Precision, and F1-score are calculated for the best k.
    - **Learning curve**: The effect of training set size on test error is plotted.

3. **Distance Metrics**:
    - KNN is tested with different distance metrics:
      - **Manhattan Distance (Minkowski with p=1)**
      - **Minkowski Distance** with varying values of log10(p) from {0.1, 0.2, ..., 1}.
      - **Chebyshev Distance** (Minkowski with p→∞).
      - **Mahalanobis Distance**

4. **Weighted Voting**:
    - KNN classification is also tested with a weighted voting system, where the influence of each neighbor is inversely proportional to its distance from the query point. The test errors for weighted voting are reported using Euclidean, Manhattan, and Chebyshev distances.

## How to Run

### Requirements

- Python 3.x
- Jupyter Notebook
- Required Python libraries (can be installed via `pip`):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Instructions

1. Clone the repository and navigate to the project folder.
2. Open the Jupyter Notebook (`Huang_Bor-Sheng_HW1.ipynb`).
3. Run the cells sequentially to execute the analysis and view results.

## Exploratory Data Analysis

Scatterplots and boxplots were generated to visualize the relationship between biomechanical attributes and the binary classification (Normal vs. Abnormal). These plots reveal potential patterns and differences in the data, aiding in feature selection and model building.

## KNN Classification

The K-Nearest Neighbors classifier was implemented with the following steps:
1. **Euclidean Distance Metric**: A KNN model was trained, and the train/test errors were plotted for different values of k. The optimal value of k was determined by the lowest test error.
2. **Performance Metrics**: Confusion matrix, True Positive Rate, True Negative Rate, Precision, and F1-score were computed for the optimal k value.

## Distance Metrics

The KNN model was further evaluated with various distance metrics:
- **Manhattan Distance**
- **Minkowski Distance with different log10(p) values**
- **Chebyshev Distance**
- **Mahalanobis Distance**

The test errors for the best k were recorded in a table for comparison across distance metrics.

## Weighted Voting

The KNN model was modified to use weighted voting, where closer neighbors have more influence. The best test errors were reported for weighted voting using different distance metrics.

## Results

- The lowest training error rate achieved was `X%`.
- The learning curve showed that increasing the training set size reduced test error rates up to a certain point.
- The optimal distance metric and k value for this dataset were found to be [insert best metric and k].

## License

This project is intended for academic purposes and is based on the DSCI 552 course material.
