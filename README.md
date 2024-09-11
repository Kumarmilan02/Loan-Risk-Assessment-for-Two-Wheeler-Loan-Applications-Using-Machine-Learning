# Loan Application Prediction: Machine Learning Project

This project aims to predict whether a loan application will be **accepted** or **rejected** based on applicant data. The dataset contains features such as demographic, financial, and verification details of the applicants. The project involves various machine learning models, ensemble techniques, and optimization strategies to improve prediction accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Approach Taken](#approach-taken)
- [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Handling Missing Values](#handling-missing-values)
  - [Feature Encoding](#feature-encoding)
- [Modeling and Evaluation](#modeling-and-evaluation)
  - [Base Model](#base-model)
  - [Optimization Techniques](#optimization-techniques)
  - [Ensemble Methods](#ensemble-methods)
- [Results & Insights](#results--insights)
- [Conclusion](#conclusion)
- [How to Run the Code](#how-to-run-the-code)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

The objective of this project is to build a predictive model that can determine the **Application Status** (Approved/Declined) of two-wheeler loan applications using a set of features. These features include demographic and financial details of loan applicants, along with information such as verification statuses. By using various machine learning models, we aim to identify patterns in the data that distinguish between approved and declined applications.

---

## Approach Taken

### 1. **Data Understanding and Exploration**
We started by exploring the dataset to understand the structure and features available. The data includes labeled entries for training (`Assignment_Train.csv`), while the test dataset (`Assignment_Test.csv`) lacks the target variable **Application Status**.

We also consulted the `Assignment_FeatureDictionary.xlsx` file, which provides a detailed description of each variable to understand their meaning and data types.

### 2. **Data Preprocessing**

#### **Feature Engineering**
Some important variables in the dataset include:
- **HDB BRANCH STATE**: Location-based information.
- **AADHAR VERIFIED**: Verification status of the applicant.
- **CIBIL Score**: Creditworthiness score of the applicant.
- **EMPLOY CONSTITUTION**: Type of employment of the applicant.

We did not create new features, but considered how existing ones would impact the model.

#### **Handling Missing Values**
- **Categorical Features**: Missing values in categorical features were imputed using the most frequent value.
- **Numerical Features**: Missing values in numerical columns, like `CIBIL Score` and `AGE`, were handled using mean imputation.

#### **Feature Encoding**
We encoded categorical variables such as **HDB BRANCH STATE** and **MARITAL STATUS** using **LabelEncoder** from scikit-learn to convert them into numerical values.

#### **Train/Test Split**
We split the `Assignment_Train.csv` dataset into 80% training data and 20% validation data to evaluate the models before testing them on the actual test set.

---

## Modeling and Evaluation

### **Base Model**
We initially chose a **Multi-layer Perceptron (MLP)** as the base model, utilizing a feedforward neural network architecture with the following configuration:
- **Input Layer**: Number of features = 15
- **Hidden Layers**: Two layers with ReLU activation
- **Output Layer**: Two classes (Approved/Declined)

#### **Performance on Train Data**
- **Accuracy**: 99.96%
- **Precision**: 94%
- **Recall**: 87%
- **F1 Score**: 90%

#### **Performance on Test Data**
- **Accuracy**: 84.45%
- **Precision**: 85%
- **Recall**: 78%
- **F1 Score**: 81%

### **Optimization Techniques**
To improve model performance, we used:
1. **Grid Search**: To systematically search for the best hyperparameters.
2. **Bayesian Optimization**: For more efficient hyperparameter tuning.

We focused on tuning the following hyperparameters:
- **Learning Rate**
- **Number of Hidden Layers and Neurons**
- **L2 Regularization (for preventing overfitting)**
- **Batch Size and Epochs**

After optimization, the MLP model demonstrated more stable results with balanced accuracy and precision.

### **Ensemble Methods**
To improve predictions, we explored the following ensemble techniques:

1. **Bagging (Bootstrap Aggregating)**:
   - Used a **BaggingClassifier** with Decision Trees to reduce variance in the model.
   - Ensemble models generated more stable predictions than individual classifiers.

2. **Boosting**:
   - Implemented **Gradient Boosting Machines (GBM)**, **LightGBM**, and **CatBoost** to sequentially correct errors of weaker models.

3. **Stacking**:
   - Combined predictions from multiple base models (e.g., SVM, Logistic Regression, and XGBoost) using a meta-learner for better overall predictions.

4. **Voting Classifier**:
   - Combined models using both hard voting and soft voting to aggregate predictions based on majority or probability estimates.

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
- **Random Forest** and **Decision Tree** classifiers had near-perfect accuracy on the training data, which may indicate overfitting, but their test performance was reasonable.
- **Logistic Regression**, **XGBoost**, and **MLP** models had strong generalization with accuracies exceeding 85% on test data.
- **Ensemble techniques** such as **Stacking** and **Voting Classifier** improved test accuracy and provided stable predictions.

---

## Conclusion

- **Best Performing Models**: Logistic Regression, XGBoost, and fine-tuned MLP showed the most reliable performance on the test dataset.
- **Ensemble Methods**: Techniques like Boosting and Stacking provided more robust and accurate predictions by leveraging multiple models.
- **Next Steps**: Further refinement of Random Forest and Decision Tree models to reduce overfitting is recommended.

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
   The predictions and model evaluations will be saved in the `results/` directory.

---

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Your contributions are welcome! We appreciate your contributions and will review pull requests promptly.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For any questions or suggestions, please reach out to:

GitHub: Kumarmilan02  
Feel free to contact me for any inquiries or discussions related to this project.
