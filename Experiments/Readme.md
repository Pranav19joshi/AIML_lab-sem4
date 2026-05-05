# Applied Machine Learning — Laboratory File
**School of Computer Science, UPES, Dehradun**
B.TECH. — IV Semester | Jan – May 2026

**Submitted By:** Pranav Joshi | SAP ID: 590012855 | Batch: 19

---

## Index

| S.No | Experiment | Date |
|------|-----------|------|
| 1 | Data Preprocessing | 27-01-26 |
| 2 | Predicting House Price using Linear Regression | 03-02-26 |
| 3 | Predicting Stock Price using Regression | 10-02-26 |
| 4 | Predicting Customer Churn Rate using Logistic Regression | 10-02-26 |
| 5 | Spam Detection (Classification and Comparative Analysis) | 24-02-26 |
| 6 | Credit Risk Assessment and Comparative Analysis | 24-02-26 |
| 7 | Anomaly Detection and Comparative Analysis | 24-02-26 |
| 8 | Student Performance Level Analysis | 17-02-26 |
| 9 | Physiological Signal Classification | 24-02-26 |
| 10 | Iris Classification | 24-02-26 |
| 11 | Prediction using Bagging Classifier | 31-02-26 |
| 12 | Regularization (Ridge and Lasso) | 31-02-26 |

---

## Experiment 1

### Aim
Text pre-processing in IPL dataset using Python libraries.

---

## Experiment 2

### Predicting Housing Prices

**Aim:** Develop a regression model to predict house prices based on features like location, size, and amenities.

### Problem Statement

The determination of real estate market value is a complex process influenced by a diverse interplay of structural, locational, and economic variables. Historically, property valuation has relied heavily on manual appraisals and the subjective expertise of agents, which can often be inconsistent, time-consuming, or influenced by human bias. As the real estate market grows in complexity, there is an increasing demand for objective, data-driven solutions that can provide rapid and accurate valuations without the variance found in manual estimations.

This project addresses these inefficiencies by seeking to automate the valuation process through the identification of intricate patterns within historical housing data. Given a dataset containing 120 detailed property records, the primary objective is to develop a robust regression model capable of transforming property attributes into precise market estimates. By leveraging machine learning, we aim to move away from subjective "best guess" valuations toward a reliable, evidence-based framework that ensures transparency and mathematical consistency for both buyers and sellers.

**Input Features:**
- **Location:** The neighborhood type (Rural, Urban, Suburban).
- **Area_sqft:** The total floor area of the house in square feet.
- **Bedrooms/Bathrooms:** The count of essential living spaces.
- **Balcony/Parking:** Additional amenities available.
- **Age_Years:** The age of the property (to account for depreciation or vintage value).
- **Furnished:** Status of the interior furnishing (Yes/No).

**Target Variable:**
- **Price:** The estimated market value of the house (in thousands or millions of units).

### Model Pipeline

#### 1. Data Understanding
- **Target Variable (y):** The target is to calculate the price of the house, which represents the numerical value we want the model to learn to predict.
- **Input Features:**
  - **Numerical:** Area_sqft, Bedrooms, Bathrooms, Age_Years, Parking, and Balconies.
  - **Categorical:** Location (Urban, Suburban, Rural) and Furnished (Yes, No).
- In the code, we will use Pandas functions like `pd.read_csv()` to load the dataset and `df.info()` to verify that each row represents an individual house record.

#### 2. Data Preprocessing
- **Handling Categorical Variables:** Since machine learning models require numerical input, we will convert text-based categories like "Urban" or "Suburban" into numbers. We will use One-Hot Encoding with help of `pandas.get_dummies()` to create binary columns like `Location_Urban` and `Furnished_Yes`, which avoids introducing false ordinal relationships.
- **Feature and Target Split:** We separate the data into X which will contain all the input columns and y which will have the output column which will be the price column.

#### 3. Train–Test Split
We use the `train_test_split()` function to divide the data into a Training set (80%) to learn the relationships and a Test set (20%) to evaluate performance.

#### 4. Regression Model & Training
- **Model Selection:** We choose Linear Regression as the mathematical model to learn how each feature contributes to the final price.
- **Training Process:** Using the `model.fit()` function, the model calculates the Coefficients (weights) that minimize prediction error. This is achieved by minimizing the error on the training data to find the best fit line.

#### 5. Predictions
- Once trained, the model is applied to the unseen test data to generate price estimates.
- We use the `model.predict()` function on the test features `X_test` to see how well the model generalizes to new information.
- We plot the actual prices against the predicted prices to visually confirm if the model is correctly following the market's upward or downward trends.

#### 6. Evaluate the Model
1. We assess the model's accuracy using Mean Squared Error (MSE) and R-Square via the `sklearn.metrics` library.
2. A lower MSE indicates a better fit as it penalizes large errors strongly, while the R² score tells us how much of the price variance is explained by our input features.

#### 7. Saving the Model
- To use this model in future applications without retraining, we save it as a file.
- We use the Pickle library (`pickle.dump()`) to serialize the trained model and save it as a `.pkl` file. This allows us to unpickle or load the model later using `pickle.load()` for instant predictions.

### Key Takeaways
- The linear regression model successfully learned the relationship between housing features such as location, area, amenities, and age with the house price, producing stable and consistent predictions.
- The model achieved very low mean squared error and a high R² score, indicating strong predictive performance. This high accuracy is expected since the dataset is synthetic and well-structured.
- One-hot encoding of categorical variables and standardization of numerical features helped ensure that all input features contributed fairly to the regression model.
- Saving both the trained model and the scaler ensures the model can be reliably reused for future predictions on new housing data.

---

## Experiment 3

### Stock Price Prediction

**Aim:** Develop a time series prediction model to forecast stock prices.

### Problem Overview

The stock market is a dynamic environment where price determination is driven by a complex interplay of historical trends, market sentiment, and macroeconomic shifts. Traditional manual forecasting methods often struggle to maintain consistency, as human analysts can be biased or overwhelmed by the sheer volume of sequential data points. This project aims to address these challenges by developing an automated valuation system that identifies hidden temporal patterns within historical stock records.

By analyzing a dataset of stock records, the task is to build a robust regression model that can predict future market values based on past performance. Moving away from subjective estimates, this project implements a data-driven framework where the model learns from historical "lags" to forecast future outcomes. This ensures that the forecasting process is grounded in empirical evidence, providing a consistent and transparent tool for investors and analysts to anticipate market movements without the pitfalls of manual estimation.

### Model Pipeline

#### 1. Understand the Dataset
- **Temporal Data:** Each row represents a specific point in time, tracking the movement of various stocks over a sequential period.
- **Target Variable (y):** Price — The continuous value we want the model to forecast for the current day.
- **Input Features (X):**
  - **Numerical (Lags):** We use the prices from the previous 5 days (lag1 to lag5) as our primary predictors.
  - **Categorical:** The Stock ticker (e.g., AAPL) identifying which specific company we are forecasting.
- **Excluded Features:** We intentionally exclude same-day Open, High, Low, Close, and Volume to prevent the model from cheating and to ensure it relies only on historical trends.

#### 2. Data Preprocessing
- **Chronological Alignment & Encoding:**
  - Machine learning models require numerical data and proper ordering, so we convert the Date column using `pd.to_datetime()` and sort the records by time.
  - We use One-Hot Encoding (via `pd.get_dummies()`) to convert stock names into binary numbers, allowing the model to distinguish between different stocks without assuming one is "higher" than another.
- **Lag Feature Engineering:**
  - To capture the "Time Series" behavior, we create 5 new columns using a `shift()` operation. These Lag Features allow the model to look at the prices from the last 5 days to predict today's value.
- **Feature-Target Split:**
  - We separate the data into X (the lags and stock dummies) and Y (the target Price) to define our forecasting function.

#### 3. Train–Test Split
- Because this is time-series data, we do not shuffle the records. We must train on the "past" to predict the "future".
- We take the first 80% of the historical timeline for the Training set and reserve the final 20% for the Test set to evaluate true forecasting accuracy.

#### 4. Choose and Train the Model
- We choose a Linear Regression model to identify the mathematical trend line that connects past prices to future values.
- The model uses the `.fit()` function to calculate weights (coefficients) for each lag. It learns exactly how much weight to give to "yesterday's price" versus "the price 5 days ago" to minimize the prediction error.

#### 5. Predictions
- The trained model is applied to the unseen test set using the `.predict()` function.
- We plot the actual prices against the predicted prices to visually confirm if the model is correctly following the market's upward or downward trends.

#### 6. Evaluate the Model
- We evaluate the performance using Mean Absolute Error (MAE) and R-Squared (R²).
- A lower MAE indicates the predicted prices are very close to reality, while the R² score shows how much of the stock's volatility is explained by its own history.

| Metric | Value |
|--------|-------|
| MAE | 1.27668 |
| R² | 1.00 |

#### 7. Save the Model
- To use the forecaster in the future without retraining on old data, we save the completed model.
- We use the Pickle library (`pickle.dump()`) to serialize the trained Linear Regression model into a `.pkl` file. This allows for instant loading and real-time forecasting in future sessions.

---

## Experiment 4

### Customer Churn Prediction

**Aim:** Customer Churn Prediction: Develop a model to predict customer churn in a subscription-based business.

### Problem Statement

Customer retention is a major challenge for service-based businesses, as acquisition costs often exceed retention costs. Manual tracking typically fails to identify the subtle behavioral patterns that signal a "churn" event. This project aims to automate the identification of at-risk customers by analyzing a dataset of 200 records including historical usage and contract data.

Unlike prior regression tasks, this is a Classification problem focused on predicting a discrete category: Churn (1) or No Churn (0). Implementing this data-driven framework allows the business to transition from reactive to proactive retention strategies. By targeting interventions toward the users most likely to depart, the business can optimize marketing resources and stabilize long-term revenue.

### Project Pipeline

#### 1. Data Understanding
The dataset consists of 200 customer records with 11 attributes.
- **Target Variable (y):** Churn.
- **Numerical Features:** Age, Tenure_Months, MonthlyCharges, TotalCharges, and SupportTickets.
- **Categorical Features:** Gender, SubscriptionType, PaymentMethod, and ContractType.

#### 2. Data Preprocessing
Raw data is cleaned and transformed to make it compatible with the Logistic Regression algorithm.
- **Feature Selection:** The `CustomerID` column is dropped using `df.drop()` as it is a unique identifier with no predictive value.
- **Cleaning:** Rows with missing values are removed via `df.dropna()`.
- **Categorical Encoding:** Text-based features are converted into numerical binary vectors using One-Hot Encoding (`pd.get_dummies()`).

#### 3. Feature-Target & Train-Test Split
The processed data is separated into features (X) and the target (y), then divided to ensure a fair evaluation.
- **Ratio:** 80% of the data is used for training, and 20% is reserved for testing.
- **Test Sample:** This results in 40 unseen records used to validate the model's accuracy.

#### 4. Choose and Train the Model
Logistic Regression is selected for its simplicity and interpretability in binary tasks.
- **Implementation:** The model is initialized with `max_iter=1000` and trained using the `.fit()` method.

#### 5. Make Predictions
The trained model applies its learned weights to the test set to flag potential churn risks.
- The `.predict()` function is used on `X_test`.
- An array of binary predictions (e.g., `array([0, 0, 1...])`), assigning a churn status to each customer in the test group.

#### 6. Evaluate the Model
The performance is measured by comparing the model's guesses against the actual outcomes.
- **Accuracy:** 75% (The model correctly identified 30 out of 40 customers).
- **Classification Report:**
  - Class 0 (Stayed): Precision and Recall are high at approximately 85%.
  - Class 1 (Churned): Precision and Recall are low at 29%, indicating the model struggles to catch the minority churn class.

| Accuracy | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 0.75 | 0.85 | 0.85 | 0.85 |

#### 7. Save the Model
The model is preserved so it can be deployed immediately in a business dashboard without retraining.
- We use the Pickle library (`pickle.dump()`) to serialize the model object.
- A permanent file named `customerchurn` is saved for instant future use.

---

## Experiment 5

### Spam Detection

**Aim:** Spam Email Detection: Build a spam email filter using text classification algorithms and perform comparative analysis. (Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree, Random Forest)

### Problem Statement

Spam detection is essential for maintaining email security and user productivity, as malicious or unsolicited emails can lead to security breaches or cluttered inboxes. Manual filtering is impossible given the volume of communication, necessitating automated classification models. This project aims to build a robust spam filter by performing a comparative analysis of five text classification algorithms: Naïve Bayes, Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest. By evaluating these models, the goal is to identify which algorithm most accurately distinguishes between "Spam" and "Ham" (legitimate) emails, enabling proactive security and improved resource management.

### Pipeline

#### 1. Problem Definition
The primary goal of this experiment is to develop a robust automated spam filter using advanced text classification techniques.
- **Classification Task:** The model must determine if an incoming email is Spam (1) or Ham (0).
- **Objective:** Perform a detailed comparative analysis across five distinct algorithms — Naïve Bayes, Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest — to identify the most reliable architecture for real-world deployment.

#### 2. Data Understanding
- **Dataset Content:** A collection of thousands of email text bodies labeled by human moderators as either spam or legitimate (ham).
- **Target (y):** Binary labels where 1 represents malicious or unsolicited "Spam" and 0 represents legitimate "Ham."
- **Input Features (X):** The raw text content of the emails, including the subject lines and message bodies.

#### 3. Data Preprocessing
Since machine learning models cannot interpret raw text, the data is put through a rigorous cleaning and transformation process.
- **Text Cleaning:** Stripping out punctuation, special characters, and numbers to reduce noise and focus only on significant vocabulary.
- **Stop-word Removal:** Eliminating frequently used words (e.g., "and", "the", "is") that appear in almost every email and carry little to no predictive value.
- **Vectorization:** Utilizing TF-IDF (Term Frequency-Inverse Document Frequency) to transform words into numerical vectors. This method balances how often a word appears in one email against how common it is across the entire dataset.

#### 4. Train-Test Split
To ensure the filter works on emails it has never seen before, the dataset is strategically divided.
- **Data Allocation:** The cleaned dataset is split into a Training set (80%) to build the models and a Test set (20%) to validate them.
- **Purpose:** This division prevents "overfitting" and allows us to measure how accurately the filter will perform in a live inbox environment.

#### 5. Model Selection and Training
Five different classification models are initialized and trained on the same training data to ensure a fair comparison:
1. **Naïve Bayes:** A probabilistic classifier based on Bayes' Theorem, highly efficient for high-dimensional text data.
2. **Logistic Regression:** A linear model that estimates the probability of an email belonging to the spam class based on the frequency of specific words.
3. **Support Vector Machine (SVM):** Finds the optimal hyperplane (boundary) that maximizes the distance between the spam and ham clusters.
4. **Decision Tree:** Uses a tree-like structure of word-based logical decisions to categorize each message.
5. **Random Forest:** An ensemble method that constructs a multitude of decision trees and outputs the class that is the mode of the individual trees, significantly improving stability.

#### 6. Predictions
Once training is complete, each of the five models is tasked with classifying the 40 unseen emails in the test set.
- The `.predict()` function is called for each algorithm, generating five separate arrays of binary predictions.
- These predictions are compared directly against the actual ground-truth labels to pinpoint errors such as "False Positives" (blocking a real email) or "False Negatives" (letting spam through).

#### 7. Model Comparison and Evaluation
The performance of each model is measured using Accuracy, Precision, and Recall. Based on the experimental outcomes:

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Naive Bayes | 0.976 | 1.0 | 0.825 |
| Logistic Regression | 0.967 | 1.0 | 0.758 |
| SVM | 0.984 | 1.0 | 0.88 |
| Decision Tree | 0.968 | 0.9 | 0.859 |
| Random Forest | 0.979 | 1.0 | 0.83 |

- **Accuracy:** Correct prediction rate of the model. SVM achieved the best accuracy (98.7%).
- **Precision:** Measures how many predicted positives are actually correct. Logistic Regression, Naive Bayes, SVM, and Random Forest scored 1.
- **Recall:** Measures how many actual positives are correctly identified. SVM achieved the highest recall (88%).

Overall SVM is the best model among all as it scored highest accuracy and recall as well as perfect precision.

#### 8. Model Saving
After identifying the superior architecture, the final step is ensuring the model is persistent.
- **Implementation:** We will save all the trained models using the Pickle library (`pickle.dump()`) to serialize the model objects into permanent files.
- **Final Output:** The generated `.pkl` files allow for immediate deployment in an email system for real-time filtering without the need for retraining.

---

## Experiment 6

### Credit Risk Assessment

**Aim:** Build a credit scoring model to assess the creditworthiness of applicants using historical financial data and perform comparative analysis (Logistic Regression, Random Forest, XGBoost).

### Problem Definition

Automated credit scoring is vital for financial institutions to mitigate potential losses and optimize lending decisions. Because the volume of applications makes manual review inefficient, we require robust classification models to predict applicant creditworthiness. This project aims to build a reliable credit filter by performing a comparative analysis of three algorithms: Logistic Regression, Random Forest, and Gradient Boosting. By evaluating these models, we seek to identify the architecture that most accurately categorizes "High," "Medium," and "Low" risk applicants, ensuring proactive risk management and improved financial stability.

### Pipeline

#### 1. Problem Definition
The primary goal of our experiment is to develop a robust automated credit scoring model using advanced classification techniques:
- **Classification Task:** We must determine the creditworthiness of an applicant, categorizing them into risk levels such as High, Medium, or Low.
- **Objective:** We perform a detailed comparative analysis across three distinct algorithms — Logistic Regression, Random Forest, and Gradient Boosting — to identify the most reliable architecture for financial deployment.

#### 2. Data Understanding
- **Dataset Content:** We utilize a collection of 500 samples of applicant profiles including historical financial health indicators.
- **Input Features (X):** We analyze raw data including age, annual income, employment years, credit score, loan amount, and late payments.
- **Target (y):** We use categorical labels representing "Credit Risk" (High, Medium, or Low).

#### 3. Data Preprocessing
Since financial data requires specific cleaning, we put it through a rigorous transformation process:
- **Feature Selection:** We drop non-predictive identifiers, such as `Applicant_ID`, to focus on significant data.
- **Label Encoding:** We utilize a `LabelEncoder` to transform categorical risk labels into numerical vectors.
- **Standardization:** We apply a `StandardScaler` to normalize numerical features like income and loan amounts, preventing large values from dominating the model logic.

#### 4. Train-Test Split
To ensure our filter works on applicants it has never seen before, we strategically divide the dataset:
- **Data Allocation:** We split the cleaned dataset into a Training set (80%) to build our models and a Test set (20%) to validate them.
- **Purpose:** We use this division to prevent "overfitting" and allow us to measure how accurately the filter will perform in a live financial environment.

#### 5. Model Selection and Training
We initialize and train three different classification models on the same data to ensure a fair comparison:
- **Logistic Regression:** We use this as a baseline linear model to estimate risk probabilities based on weighted input features.
- **Random Forest:** We implement this ensemble method to construct multiple decision trees, improving stability and capturing non-linear relationships.
- **Gradient Boosting:** We task this model with correcting errors of previous trees sequentially to achieve high-fidelity predictions.

#### 6. Predictions
- **Execution:** Once training is complete, we task each model with classifying the 100 unseen samples in our test set.
- **Output Generation:** We generate binary and multi-class prediction arrays for each algorithm to prepare for the evaluation phase.

#### 7. Model Evaluation and Comparison
The performance of each model is measured using Accuracy, Precision, Recall, and F1. Based on the experimental outcomes:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.85 | 0.81 | 0.76 | 0.79 |
| Random Forest | 0.98 | 0.98 | 0.87 | 0.91 |
| Gradient Boosting | 1.00 | 1.00 | 1.00 | 1.00 |

We determine that Gradient Boosting is the best model among all tested as it scored the highest across all critical financial metrics and showed perfect classification capability on the test set.

#### 8. Model Saving
The final step is ensuring our models are persistent for real-time filtering:
- We save all trained models, along with the scaler and label encoder, using the Pickle library to serialize the objects into permanent `.pkl` files.
- We generate these files to allow for immediate deployment in a banking system without the need for retraining.

---

## Experiment 7

### Anomaly Detection

**Aim:** Implement an anomaly detection system for detecting outliers in data (e.g., fraud detection) and perform comparative analysis. (Isolation Forest, Local Outlier Factor, One-Class SVM)

### Problem

Anomaly detection is critical for identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data. Because the volume of transactions makes manual inspection for fraud or errors impossible, we require automated unsupervised models to detect outliers. This project aims to build a robust detection system by performing a comparative analysis of three algorithms: Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM. By evaluating these models, we seek to identify the architecture that most accurately distinguishes between "Normal" data and "Anomalies," ensuring proactive security and data integrity.

### Pipeline

#### 1. Problem Definition
The primary goal of our experiment is to develop a robust automated anomaly detection system using advanced unsupervised learning techniques:
- **Classification Task:** We must determine if a data point is Normal (1) or an Anomaly (-1).
- **Objective:** We perform a detailed comparative analysis across three distinct algorithms — Isolation Forest, Local Outlier Factor, and One-Class SVM — to identify the most reliable architecture for outlier detection.

#### 2. Data Understanding
- **Dataset Content:** We utilize a collection of 500 samples consisting of features like Transaction Amount, Frequency, and Time.
- **Input Features (X):** We analyze numerical data representing behavioral patterns (e.g., Feature_1, Feature_2).
- **Target (y):** We use labels where 1 represents a "Normal" observation and -1 represents an "Anomaly" (Outlier).

#### 3. Data Preprocessing
Since anomaly detection models are sensitive to the scale of data, we put it through a rigorous transformation process:
- **Cleaning:** We remove non-predictive columns and ensure the data is formatted for unsupervised learning.
- **Standardization:** We apply a `StandardScaler` to normalize numerical features, ensuring that features with larger ranges (like transaction amounts) do not disproportionately influence the distance-based models like LOF and SVM.

#### 4. Train-Test Split
To evaluate how well the system identifies new outliers, we strategically divide the dataset:
- **Data Allocation:** We split the dataset into a Training set (80%) to establish the "normal" baseline and a Test set (20%) to validate detection accuracy.
- **Purpose:** This division allows us to measure the model's ability to generalize and identify anomalies in unseen data.

#### 5. Model Selection and Training
We initialize and train three different anomaly detection models on the same data to ensure a fair comparison:
- **Isolation Forest:** We use this tree-based model that explicitly isolates anomalies by randomly selecting a feature and a split value.
- **Local Outlier Factor (LOF):** We implement this density-based method that identifies outliers by comparing the local density of a point to its neighbors.
- **One-Class SVM:** We task this model with learning a decision boundary that encompasses the "normal" data points in a high-dimensional space.

#### 6. Predictions
- **Execution:** Once training is complete, we task each model with classifying the 100 unseen samples in our test set.
- **Output Generation:** We generate prediction arrays where each algorithm flags points as either 1 (Normal) or -1 (Anomaly).

#### 7. Model Evaluation and Comparison
The performance of each model is measured using Accuracy, Precision, Recall, and F1-Score. Based on the experimental outcomes:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Isolation Forest | 0.96 | 0.7 | 0.7 | 0.7 |
| Local Outlier Factor | 0.88 | 0 | 0 | 0 |
| One-Class SVM | 0.93 | 0.41 | 0.43 | 0.42 |

We determine that Isolation Forest is the best model among all tested as it achieved the highest accuracy and balanced precision and recall, making it the most effective at isolating outliers without misclassifying normal data.

#### 8. Model Saving
The final step is ensuring our models are persistent for real-time monitoring:
- **Implementation:** We save all trained models, along with the anomaly scaler, using the Pickle library to serialize the objects into permanent `.pkl` files.
- **Final Output:** We generate these files to allow for immediate deployment in a fraud detection system for real-time filtering without the need for retraining.

---

## Experiment 8

### Student Performance Level Analysis

**Aim:** Implement Multiclass Classification models for students Performance Level analysis and perform comparative analysis. (Random Forest, Decision Tree, Multinomial Logistic Regression, XGBoost, K-Nearest Neighbors)

### Pipeline

#### 1. Problem Definition
The objective of this experiment is to develop an automated system to predict a student's **Performance Level** based on various academic and behavioral metrics.
- **Classification Task:** This is a multiclass classification problem where students are categorized into distinct levels (e.g., 0, 1, or 2).
- **Objective:** Perform a comparative analysis across five distinct algorithms — Multinomial Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (XGBoost/GBM), and K-Nearest Neighbors (KNN) — to determine which model best predicts academic outcomes.

#### 2. Data Understanding
- **Dataset Content:** A collection of 500 student samples with academic indicators.
- **Input Features (X):**
  - **Numerical:** Study Hours, Attendance Percentage, Assignment Score, Internal Marks.
  - **Categorical:** Participation (Low/Medium/High), Internet Access (Yes/No), Previous Grade (A/B/C).
- **Target (y):** Performance_Level (Multiclass labels representing different tiers of academic achievement).

#### 3. Data Preprocessing
To prepare the dataset for multiclass classification, a rigorous transformation process was applied to ensure the data is in a format compatible with both linear and distance-based algorithms:
- **Feature-Target Separation:** After encoding, the dataset was divided into Input Features (X) and Target (y): the Performance_Level label.
- **Standardization:** We applied `StandardScaler` to normalize the numerical features.
- **Label Encoding:** We perform label encoding on the categorical data to convert it into numerical data.

#### 4. Train-Test Split
- **Data Allocation:** The dataset was partitioned into a Training set (80%) to train the models and a Test set (20%) to evaluate their performance on unseen data.
- **Random State:** A fixed `random_state=42` was used to ensure the results are reproducible across different runs.

#### 5. Model Selection and Training
Five different classification architectures were initialized and trained:
1. **Multinomial Logistic Regression:** A linear model adapted for multiclass settings using the softmax function.
2. **Decision Tree:** A non-linear model that splits data based on feature thresholds to create a tree-like decision structure.
3. **Random Forest:** An ensemble of decision trees that reduces overfitting by averaging multiple "votes."
4. **Gradient Boosting:** An iterative ensemble technique that builds new trees to correct the errors made by previous ones.
5. **K-Nearest Neighbors (KNN):** A distance-based model that classifies points based on the majority label of their nearest neighbors.

#### 6. Predictions
- Testing data is used to do prediction in the trained model.
- The predicted output is stored to further use in model evaluation.

#### 7. Model Evaluation and Comparison
The models were evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score** (using 'macro' averaging to account for class distribution) as well as AUC ROC curve.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9700 | 0.6474 | 0.6599 | 0.6533 |
| Decision Tree | 0.9000 | 0.9320 | 0.9320 | 0.9320 |
| Random Forest | 0.9400 | 0.6269 | 0.6395 | 0.6331 |
| Gradient Boosting | 0.9500 | 0.9661 | 0.9660 | 0.9660 |
| K-Nearest Neighbors | 0.7900 | 0.5272 | 0.5374 | 0.5321 |

**ROC-AUC Curve Analysis:**

| Model | Class 0 | Class 1 | Class 2 |
|-------|---------|---------|---------|
| Logistic Regression | 1.00 | 0.96 | 0.86 |
| Decision Tree | 0.9 | 0.90 | 1.00 |
| Random Forest | 1.00 | 0.98 | 1.00 |
| Gradient Boosting | 0.99 | 0.99 | 1.00 |
| K-Nearest Neighbors | 0.92 | 0.89 | 0.50 |

**Conclusion:** While Logistic Regression achieved the highest raw accuracy, **Gradient Boosting** is the most robust model for this dataset, as it provides the best balance between high accuracy and superior Precision/Recall/F1-Scores.

#### 8. Model Saving
To make the system deployable for real-time student analysis:
- **Implementation:** All five trained models and the StandardScaler were serialized into `.pkl` files using the Pickle library.
- **Final Output:** These files allow the prediction system to be reloaded in a production environment (like a school dashboard) without retraining the models.

---

## Experiment 9

### Physiological Signal Classification

**Aim:** Implement a classification model to distinguish between normal and abnormal physiological signals using extracted signal features. (Perform comparative analysis on different ML models)

### Pipeline

#### 1. Problem Definition
The primary goal of this experiment is to automate the detection of abnormal heart rhythms (arrhythmias) which is critical for early diagnosis of cardiovascular diseases:
- **Classification Task:** Categorize heartbeat samples into "Normal" (0) and "Abnormal" (1).
- **Objective:** Perform a comparative analysis using a Decision Tree, Random Forest, and SVM, and ultimately combine them into a Voting Classifier to improve overall prediction stability and accuracy.

#### 2. Data Understanding
- **Dataset Content:** The dataset consists of ECG recordings from the PTB Diagnostic Datasets, containing thousands of heartbeat samples.
- **Input Features (X):** 187 numerical features representing the normalized intensity of the ECG signal over a single heartbeat period.
- **Target (y):** Binary labels where 0 represents a Normal heartbeat and 1 represents an Abnormal heartbeat.

#### 3. Data Preprocessing
To prepare the medical signal data for the machine learning models:
- **Data Integration:** Combined separate CSV files for normal and abnormal cases into a single unified dataframe.
- **Feature Extraction:** Separated the signal data (first 187 columns) from the ground truth labels (last column).
- **Class Labeling:** Explicitly assigned numerical values (0 and 1) to distinguish between the two health states.

#### 4. Train-Test Split
To validate the model's diagnostic reliability on unseen patients:
- **Data Allocation:** The combined dataset was split into a Training set (80%) to teach the models and a Test set (20%) for final validation.
- **Random State:** A fixed seed (42) was used to ensure the results are reproducible across different runs.

#### 5. Model Selection and Training
Four distinct configurations were trained to compare individual performance vs. ensemble performance:
- **Decision Tree:** A baseline model using 'entropy' as the split criterion with a maximum depth of 10.
- **Random Forest:** An ensemble of 100 trees to reduce variance and improve accuracy through bagging.
- **Support Vector Machine (SVM):** A high-dimensional classifier used to find the optimal hyperplane between normal and abnormal signals.
- **Voting Classifier (Soft Voting):** A meta-classifier that aggregates the probability predictions of the three models above to make a final "consensus" decision.

#### 6. Predictions
- Each trained model was used to predict the labels for the unseen samples in the test set.
- The predicted data is further used in evaluation of the model.

#### 7. Model Evaluation and Comparison
The performance was evaluated using Precision, Recall, and F1-Score, specifically focusing on the model's ability to identify abnormal cases correctly.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 0.8952 | 0.9355 | 0.9182 | 0.9268 |
| Random Forest | 0.9705 | 0.9701 | 0.9895 | 0.9797 |
| SVM | 0.9024 | 0.9139 | 0.9548 | 0.9339 |
| Voting Ensemble | 0.9443 | 0.9567 | 0.9667 | 0.9617 |

**AUC-ROC Curve:**

| Decision Tree | Random Forest | SVM | Voting Ensemble |
|---------------|---------------|-----|-----------------|
| 0.93 | 0.99 | 0.94 | 0.98 |

#### 8. Model Saving
The final step ensures the diagnostic tool is ready for clinical application:
- **Serialization:** The trained Voting Classifier was saved into a `.pkl` file using the Pickle library.
- **Deployment Ready:** This file allows medical software to load the model and analyze live ECG streams without needing to retrain the algorithms.

---

## Experiment 10

### Iris Classification

**Aim:** Iris Flower Classification: Use the Iris dataset to build a classification model that predicts the species of iris flowers. [Dataset: Load dataset from sk-learn]

### Pipeline

#### 1. Problem Definition
The primary goal of this experiment is to develop a robust automated classification system to identify biological species based on physical measurements:
- **Classification Task:** We must determine if a data point belongs to the **Setosa**, **Versicolor**, or **Virginica** class.
- **Objective:** We perform a detailed comparative analysis across three distinct algorithms — Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression — to identify the most accurate architecture for multiclass classification.

#### 2. Data Understanding
- **Dataset Content:** We utilize the classic Iris dataset consisting of 150 samples with four physical attributes: sepal length, sepal width, petal length, and petal width.
- **Input Features (X):** We analyze numerical data representing the four flower measurements.
- **Target (y):** We use labels where 0, 1, and 2 represent the three specific species of iris.

#### 3. Data Preprocessing
To ensure the data is suitable for algorithmic processing, we perform the following:
- **Data Formatting:** We separate the dataset into a feature matrix (X) and a target vector (y).
- **Cleanliness:** As the Scikit-Learn iris dataset is a standard benchmark, the data is verified to be clean and correctly formatted for supervised learning.

#### 4. Train-Test Split
To evaluate how well the system identifies species in new samples, we strategically divide the dataset:
- **Data Allocation:** We split the dataset into a Training set (80%) to allow the models to learn the patterns and a Test set (20%) to validate accuracy.
- **Reproducibility:** We use a `random_state` of 42 to ensure that the data split remains consistent across different experimental runs.

#### 5. Model Selection and Training
We initialize and train three different classification models on the same data to ensure a fair comparison:
- **Decision Tree Classifier:** We use this tree-based model that breaks down the dataset into smaller subsets while at the same time an associated decision tree is incrementally developed.
- **K-Nearest Neighbors (KNN):** We implement this instance-based method that classifies a point based on the majority class of its nearest neighbors in the feature space.
- **Logistic Regression:** We task this linear model with learning the probability of a sample belonging to each of the three species classes.

#### 6. Predictions
- **Execution:** Once training is complete, we task each model with classifying the 30 unseen samples in our test set.
- **Output Generation:** We generate prediction arrays where each algorithm flags points as either class 0, 1, or 2.

#### 7. Model Evaluation and Comparison
The performance of each model is measured using a Confusion Matrix for visual verification and a Classification Report containing Precision, Recall, and F1-Score. Based on the experimental outcomes:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |
| KNN | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |

As the data is very simple and got perfect scores in all the models, the AUC-ROC of all the models was also 1.00.

#### 8. Model Saving
The final step is ensuring our models are persistent for future deployment:
- **Implementation:** We save the top-performing trained model (Logistic Regression) using the Pickle library to serialize the object into a permanent `.pkl` file.
- **Final Output:** We generate the file `logistic_iris.pkl` to allow for immediate deployment in an automated classification system without the need for retraining.

---

## Experiment 11

### Diabetes Prediction using Bagging Classifier

**Aim:** To predict whether a person has diabetes based on features such as blood pressure, skin thickness, age, etc., using the bagging ensemble technique. Also perform comparative analysis among the bagging classifier, random forest, and the decision tree classifier.

### Pipeline

#### 1. Problem Definition
The goal of this experiment is to implement an automated classification system to predict student diabetes outcomes based on diagnostic metrics.
- **Classification Task:** This is a binary classification problem where samples are categorized into two distinct classes (Outcome 0 or 1).
- **Objective:** Perform a comparative analysis between a single Decision Tree and ensemble methods, specifically Random Forest and Bagging, to determine which architecture offers the most reliable predictive performance.

#### 2. Data Understanding
The experiment utilizes a diabetes dataset containing several medical indicator features:
- **Input Features (X):** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
- **Target (y):** Outcome (Binary labels where 1 indicates the presence of diabetes and 0 indicates absence).

#### 3. Data Preprocessing
To prepare the data for training, the following cleaning and transformation steps were executed:
- **Missing Value Handling:** Columns containing invalid zero values (Glucose, BloodPressure, SkinThickness, Insulin, BMI) were identified. These zeros were replaced with NaN and subsequently imputed using the median value of each column.
- **Feature-Target Separation:** The dataset was divided into input features (X) and the target variable (y).
- **Standardization:** The `StandardScaler` was applied to normalize the feature set, ensuring that variables with different scales do not bias the algorithms.

#### 4. Train-Test Split
- **Data Allocation:** The dataset was partitioned into a Training set (80%) to build the models and a Test set (20%) to validate performance on unseen data.
- **Random State:** A fixed `random_state=42` was used to ensure reproducibility of the data split and model results.

#### 5. Model Selection and Training
Three distinct classification architectures were initialized and trained:
- **Decision Tree:** A base non-linear model that creates decision rules based on feature thresholds.
- **Random Forest:** An ensemble technique that builds multiple decision trees on different data subsets and averages their results to improve accuracy and control overfitting.
- **Bagging Classifier:** An ensemble method that uses a `DecisionTreeClassifier` as the base estimator, training 200 different versions of the model on bootstrapped samples to reduce variance.
- All three models are trained on the training data.

#### 6. Predictions
- The standardized test set was processed by all three trained models to generate predicted labels for the samples.
- The models transformed medical input patterns into binary classifications (0 or 1), which were then compared against the actual ground-truth outcomes from the test set.

#### 7. Model Evaluation and Comparison
The models were evaluated using a suite of classification metrics and visualization tools:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Decision Tree | 0.71 | 0.59 | 0.61 | 0.60 |
| Random Forest | 0.74 | 0.63 | 0.65 | 0.64 |
| Bagging (Decision Tree) | 0.77 | 0.67 | 0.70 | 0.69 |

**AUC-ROC Score:**

| Decision Tree | Random Forest | Bagging |
|---------------|---------------|---------|
| 0.69 | 0.83 | 0.83 |

By looking at the results we can conclude that the Bagging classifier outperforms Random Forest and Decision Tree in all the metrics.

#### 8. Model Saving
To ensure the models are ready for deployment or future use without retraining:
- **Serialization:** All three trained models (Decision Tree, Random Forest, and Bagging Classifier) were exported into `.pkl` files using the pickle library.
- **Utility:** These files allow for the immediate reloading of the predictive system into a production environment for real-time analysis.

---

## Experiment 12

### Regularization

**Aim:** Implement L1 (Lasso) and L2 (Ridge) regularization on the given Melbourne House Price Dataset.

### Pipeline

#### 1. Problem Definition
The primary goal is to develop a predictive model for real estate valuation using advanced regression techniques:
- **Regression Task:** We aim to predict the continuous numerical value of house prices based on various structural and locational features.
- **Objective:** We perform a comparative analysis between Standard Linear Regression, Lasso (L1), and Ridge (L2) regularization to determine which approach best handles multicollinearity and prevents model over-complexity.

#### 2. Data Understanding
- **Dataset Content:** We utilize the Melbourne House Price dataset, which contains records of real estate transactions.
- **Input Features (X):**
  - **Numerical:** Rooms, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Propertycount.
  - **Categorical:** Suburb, Address, Type, Method, SellerG, Date, CouncilArea, Regionname.
- **Target (y):** Price (The market value of the property in AUD).

#### 3. Data Preprocessing
To ensure the dataset is optimized for regularization, a rigorous transformation process is applied:
- **Handling Missing Values:** We address null entries in features like 'Car', 'BuildingArea', and 'YearBuilt' using zero-filling or mean/median imputation.
- **Categorical Encoding:** We apply One-Hot Encoding (via `get_dummies`) to transform non-numeric features into binary columns compatible with regression algorithms.
- **Feature-Target Separation:** The dataset is divided into the feature matrix (X) and the target variable (y).

#### 4. Train-Test Split
- **Data Allocation:** The dataset is partitioned into a Training set (80%) to fit the model and a Test set (20%) to evaluate its performance on unseen data.
- **Random State:** A fixed `random_state` is used to ensure the results and data splits are reproducible across different runs.

#### 5. Model Selection and Training
We initialize and train three different regression architectures to observe the impact of penalties:
- **Linear Regression:** A baseline model that calculates coefficients without any regularization constraints.
- **Lasso Regression (L1):** A model that adds an absolute value penalty to the loss function, which can shrink some coefficients to exactly zero, performing automatic feature selection.
- **Ridge Regression (L2):** A model that adds a squared magnitude penalty to the loss function, which helps manage multicollinearity by shrinking coefficients but not setting them to zero.

#### 6. Predictions
- **Execution:** The trained models (Linear, Lasso, and Ridge) are used to predict house prices for the properties in the test set.
- **Storage:** Predicted values are stored to calculate error metrics and compare against the actual house price.

#### 7. Model Evaluation and Comparison
The models are evaluated using the R² score (Coefficient of Determination) to measure how well the models generalize:

| Linear Regression | Ridge | Lasso |
|-------------------|-------|-------|
| 0.6426 | 0.6529 | 0.6530 |

#### 8. Model Saving
- **Implementation:** The optimized regularized model and any preprocessing objects are serialized into `.pkl` files using the **Pickle** library.
- **Final Output:** These files allow the prediction system to be reloaded for real-time house price estimations without needing to retrain the model.

---
  <h2 id="mentor">Mentor</h2>
  <p><strong>Dr. Sahinur Rahman Laskar</strong><br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
