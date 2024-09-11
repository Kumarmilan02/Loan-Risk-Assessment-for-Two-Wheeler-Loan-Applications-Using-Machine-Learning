# Loan Application Prediction: Machine Learning Project

This project aims to predict whether a loan application will be accepted or rejected based on applicant data. The dataset contains various features, including demographic, financial, and verification details of loan applicants.

## Table of Contents

- [Project Overview](#project-overview)
- [Approach Taken](#approach-taken)
- [Data Preprocessing](#data-preprocessing)
- [Modeling and Evaluation](#modeling-and-evaluation)
  - [Base Model](#base-model)
  - [Fine-Tuning & Optimization](#fine-tuning--optimization)
- [Ensemble Methods](#ensemble-methods)
- [Results & Insights](#results--insights)
- [Conclusion](#conclusion)
- [How to Run the Code](#how-to-run-the-code)

---

## Project Overview

The objective of this project is to build a predictive model to determine the **Application Status** (Approved/Declined) of two-wheeler loan applications based on a set of features, including the applicant's demographic and financial information. The project includes preprocessing, multiple machine learning models, model optimization, and ensemble methods for better accuracy.

---

## Approach Taken

### 1. **Data Preprocessing**
The first step involved cleaning and preprocessing the data:
- Handled missing values.
- Categorical features were encoded using **LabelEncoder**.
- Continuous variables were normalized.
- Data was split into training and test sets with a ratio of 80:20.

The preprocessing pipeline was designed to ensure consistent transformation of both the training and test datasets.

### 2. **Modeling**
We trained various machine learning models and compared their performance using key evaluation metrics such as **accuracy, precision, recall, F1-score, log loss, and ROC-AUC**.

#### Models used:
1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. k-Nearest Neighbors (k-NN)
4. Decision Tree Classifier
5. Gradient Boosting Classifier
6. Logistic Regression
7. XGBoost Classifier
8. Naive Bayes
9. Perceptron
10. Stochastic Gradient Descent (SGD)
11. Multi-layer Perceptron (MLP)

---

## Data Preprocessing

### **Steps Followed:**
1. **Feature Encoding:** Categorical features were encoded using **LabelEncoder** for simplicity.
2. **Scaling:** We applied normalization to continuous numerical features for better convergence of the models.
3. **Train/Test Split:** We split the dataset into an 80% training set and a 20% test set.
4. **Pipeline Setup:** A preprocessing pipeline was built using scikit-learn to ensure consistent transformation of both train and test data.

---

## Modeling and Evaluation

### **Base Model**
We started with a **Multi-layer Perceptron (MLP)** as the base model for the task. The model was trained using a standard feedforward neural network architecture. After the initial model evaluation, the following metrics were recorded:

#### **Performance on Train Data**
- **Accuracy:** 99.96%
- **Precision:** 94%
- **Recall:** 87%
- **F1 Score:** 90%

#### **Performance on Test Data**
- **Accuracy:** 84.45%
- **Precision:** 85%
- **Recall:** 78%
- **F1 Score:** 81%

### **Optimization Techniques**
We performed **Grid Search** and **Bayesian Optimization** to fine-tune the hyperparameters of the base MLP model.

Key parameters tuned:
1. **Learning Rate**
2. **Hidden Layer Structure** (number of layers, neurons per layer)
3. **Regularization Parameters** (L2 penalty)
4. **Batch Size and Epochs**

The best configuration was chosen based on cross-validation scores, and further evaluation was performed on both train and test data.

---

### **Ensemble Methods**
Several ensemble techniques were explored to improve model performance by combining predictions from multiple models:

1. **Bagging**: This method reduces variance by averaging predictions from multiple models. A **BaggingClassifier** with Decision Trees was used.
2. **Boosting**: Models were sequentially trained using **Gradient Boosting Machines (GBM)**, **LightGBM**, and **CatBoost** to focus on difficult cases in the dataset.
3. **Stacking**: Predictions from multiple base models were used as input for another model (meta-learner) to improve accuracy.
4. **Voting Classifier**: Combined predictions of the top-performing models using a majority voting approach.

---

## Results & Insights

### **Performance Comparison of Models**

| Model                            | Train Accuracy | Test Accuracy |
|-----------------------------------|----------------|---------------|
| Random Forest Classifier          | 99.99%         | 83.88%        |
| Support Vector Machine (SVM)      | 87.28%         | 82.89%        |
| k-Nearest Neighbors (k-NN)        | 85.25%         | 75.43%        |
| Decision Tree Classifier          | 99.99%         | 83.77%        |
| Gradient Boosting Classifier      | 84.39%         | 82.63%        |
| Logistic Regression               | 94.03%         | 85.60%        |
| XGBoost Classifier                | 92.77%         | 85.08%        |
| Naive Bayes                       | 88.59%         | 78.51%        |
| Perceptron                        | 93.66%         | 84.71%        |
| Stochastic Gradient Descent (SGD) | 93.79%         | 85.86%        |
| Multi-layer Perceptron (MLP)      | 99.96%         | 84.45%        |

### **Insights**
1. **Random Forest Classifier** and **Decision Tree Classifier** achieved near-perfect accuracy on the train set, suggesting potential overfitting. However, their performance on the test set was around 83.88%.
2. **Logistic Regression** and **XGBoost** demonstrated strong generalization with test accuracies of 85.60% and 85.08%, respectively.
3. **Support Vector Machine (SVM)** and **Gradient Boosting Classifier** performed well, with balanced accuracy between train and test sets.
4. **Ensemble Methods** provided marginal improvements in accuracy but helped reduce the variance in predictions.
5. **MLP** showed strong performance as a base model, and after fine-tuning, it provided comparable results to XGBoost and Logistic Regression.

---

## Conclusion

- **Best Models**: Logistic Regression, XGBoost, and fine-tuned MLP demonstrated strong performance on the test dataset.
- **Ensemble Techniques**: Stacking and Boosting techniques provided robust solutions by combining different model predictions.
- **Next Steps**: Fine-tuning and validation of ensemble methods with a focus on reducing overfitting in Random Forest and Decision Tree models.

---

## How to Run the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/loan-application-prediction.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main script**:
   ```bash
   python main.py
   ```
4. **View the results**:
   Predictions and model evaluations will be saved in the `results/` directory.
