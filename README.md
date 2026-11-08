# ml-assignment-2
Machine Learning Course Assigment 2 - BITS Pilani
# Machine Learning Classification & Evaluation Dashboard

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five distinct Machine Learning classification algorithms on a multi-feature dataset, and deploy an interactive web dashboard via Streamlit Community Cloud to evaluate test datasets on pre-trained models.

## b. Dataset Description
- **Dataset Name**: Synthetic Binary Classification Dataset (13-Feature Standard Tabular Set)
- **Number of Instances**: 1,000 samples (800 train / 200 test split)
- **Number of Features**: 13 numeric continuous features (10 informative, 3 redundant)
- **Target Variable**: Binary class labels (0 or 1)

## c. GitHub Repository Link
`https://github.com/stirumal16/ml-assignment-2`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8850 | 0.9420 | 0.8725 | 0.9000 | 0.8861 | 0.7702 |
| **Decision Tree** | 0.8350 | 0.8350 | 0.8235 | 0.8500 | 0.8364 | 0.6702 |
| **KNN** | 0.8700 | 0.9250 | 0.8641 | 0.8800 | 0.8721 | 0.7401 |
| **Naive Bayes** | 0.8600 | 0.9310 | 0.8447 | 0.8800 | 0.8627 | 0.7205 |
| **Support Vector Machine** | 0.8950 | 0.9510 | 0.8835 | 0.9100 | 0.8966 | 0.7903 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Strong linear baseline performance with fast inference and high AUC (0.9420), showing good linear separability in scaled features. |
| **Decision Tree** | Showed slight variance sensitivity and lower AUC (0.8350) compared to probabilistic models due to axis-aligned decision boundaries. |
| **KNN** | Good non-linear decision boundary capture; sensitive to scaling, reaching competitive F1 (0.8721). |
| **Naive Bayes** | Performs reliably with fast prediction speed; minor performance hit due to feature correlation assumptions. |
| **Support Vector Machine** | Superior performance across all metrics (Accuracy: 0.8950, MCC: 0.7903) due to RBF kernel effective margin maximization. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the highest F1 score (0.8966), AUC (0.9510), and MCC (0.7903). |