# Applied Machine Learning — Mini Project
**School of Computer Science, UPES, Dehradun**
B.Tech IV Semester | Jan – May 2026

**Submitted By:** Pranav Joshi | SAP ID: 590012855 | Batch: 19

---

# Project 1: Automobile Price Prediction

**Dataset:** `automobiles.csv` (UCI Automobile Dataset)
**Target Variable:** `price` (continuous numeric — USD)
**Problem Type:** Regression

---

## 1. Problem Definition

The goal of this project is to build a machine learning regression model that can accurately predict the price of an automobile given its technical and physical specifications. Unlike classification tasks, the output here is a continuous numeric value (price in USD), making it a pure regression problem.

| Parameter | Detail |
|---|---|
| Target Variable (y) | `price` — the listed selling price of the car in USD |
| Nature of Target | Continuous numeric; ranges from ~$5,000 to ~$45,000 in this dataset |

---

## 2. Data Understanding

The dataset is the UCI Automobile Dataset loaded from `automobiles.csv`. Each row represents one car model. The dataset contains **26 columns** in total, including categorical specs (make, body-style, drive-wheels) and numerical measurements (engine-size, horsepower, mpg). The target column `price` appears in the last column.

**Numerical Features (X):**
`normalized-losses`, `wheel-base`, `length`, `width`, `height`, `curb-weight`, `engine-size`, `bore`, `stroke`, `compression-ratio`, `horsepower`, `peak-rpm`, `city-mpg`, `highway-mpg`

**Categorical Features (X):**
`make`, `fuel-type`, `aspiration`, `num-of-doors`, `body-style`, `drive-wheels`, `engine-location`, `engine-type`, `num-of-cylinders`, `fuel-system`

**Target Variable (y):** `price`

---

## 3. Data Pre-Processing

Raw data from `automobiles.csv` contains missing values encoded as `'?'` strings, mixed numeric/text columns, and categorical features that must be converted before any ML algorithm can process them. The preprocessing pipeline performs the following steps in strict order:

### Step 3a: Replace `'?'` with NaN
The raw CSV uses `'?'` as a placeholder for missing values. Replacing with `NaN` allows pandas/sklearn to detect and handle them properly.

```python
df.replace('?', np.nan)
```

### Step 3b: Drop Rows with Missing Price
`price` is the target variable — a row without a known price cannot be used for supervised learning and is removed entirely.

```python
df.dropna(subset=['price'])
df['price'] = pd.to_numeric(df['price'])
```

### Step 3c: Cast Numerical Columns to Float
14 columns (`engine-size`, `horsepower`, `bore`, etc.) are stored as `object` dtype due to the `'?'` contamination. Explicit casting enables median imputation in the next step.

```python
for col in real_num_cols:
    df[col] = pd.to_numeric(df[col])
```

### Step 3d: Median Imputation for Numerical Missing Values
Remaining `NaN` values in numerical columns are filled using the **median** (not mean) because median is robust to outliers — automobile specs like horsepower can have extreme values that would distort mean imputation.

```python
imputer = SimpleImputer(strategy='median')
df[real_num_cols] = imputer.fit_transform(df[real_num_cols])
```

### Step 3e: One-Hot Encoding of Categorical Features
Categorical columns (`make`, `body-style`, `fuel-type`, `drive-wheels`, `engine-type`, etc.) are converted into binary (0/1) indicator columns using One-Hot Encoding. `drop_first=True` avoids the dummy-variable trap (multicollinearity).

```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

### Step 3f: Feature Scaling — StandardScaler
All numerical features are standardised to zero-mean and unit-variance. This is critical for regularised models (Ridge, Lasso) because they penalise coefficients — without scaling, features with large ranges dominate the penalty term.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 4. Splitting

After preprocessing, the full feature matrix `X_scaled` and target vector `y` are split into training and testing subsets using sklearn's `train_test_split`.

| Split Parameter | Value | Reasoning |
|---|---|---|
| `test_size` | 0.2 (20%) | Standard hold-out ratio. Leaves enough data for reliable evaluation without shrinking the training set too much. |
| Training samples | ~160 rows | 80% of the cleaned dataset used to fit all regression models. |
| Testing samples | ~40 rows | 20% completely unseen during training — used only for final evaluation. |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

## 5. Model Training

Five regression algorithms are trained and compared. Each model was chosen for a specific reason based on the nature of the automobile dataset:

### 1. Linear Regression
Acts as a benchmark assuming linear price-feature relationships; no hyperparameters required for simple performance validation.

```python
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
```

### 2. Ridge Regression (L2 Regularisation)
Prevents overfitting in datasets with many dummy variables by using L2 penalties to shrink coefficients.

```python
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
```

### 3. Lasso Regression (L1 Regularisation)
Uses L1 penalties for automatic feature selection, zeroing out irrelevant features to create a sparser model.

```python
model_lasso = Lasso(alpha=1.0, max_iter=10000)
model_lasso.fit(X_train, y_train)
```

### 4. Decision Tree Regressor
Handles non-linear price impacts across feature regions without requiring data scaling for original splits.

```python
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
```

### 5. Random Forest Regressor
Ensemble of 100 trees that reduces variance, captures complex feature interactions, and provides robustness against price outliers.

```python
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
```

---

## 6. Model Testing and Evaluation

After training, every model predicts prices on the held-out test set and is evaluated using three complementary methods: numeric metrics, visual actual-vs-predicted scatter plots, and a final side-by-side comparison chart.

**Evaluation Metrics Explained:**

| Metric | Explanation |
|---|---|
| R² Score | A statistical measure that represents the proportion of the variance for a dependent variable that's explained by the independent variables in the model. |
| MSE (Mean Squared Error) | The average of the squared differences between the actual values and the predicted values. |

**Final Comparison:**

| Model | R² | MSE | RMSE |
|---|---|---|---|
| Linear Regression | 0.8750 | 14,007,858 | 3,742 |
| Ridge | 0.9304 | 7,794,482 | 2,791 |
| Lasso | 0.8831 | 13,094,871 | 3,618 |
| Decision Tree | 0.9217 | 8,770,303 | 2,961 |
| Random Forest | **0.9685** | **3,533,105** | **1,879** |

---

## 7. Conclusion

- **Data preprocessing was the most critical step.** The dataset contained `'?'` as missing values and mixed data types across 26 columns, requiring type conversion, imputation, and label encoding before any model could be trained.
- **Random Forest significantly outperformed all other models**, achieving an R² of 0.9685 and an RMSE of ~1880 — far better than plain Linear Regression (R² = 0.875, RMSE ~3743), confirming that ensemble methods capture the complex, non-linear interactions in car pricing data.
- **Regularization improved over basic linear regression but has limits.** Ridge (R² = 0.930) outperformed both Linear and Lasso, showing that L2 regularization helps control overfitting on this high-dimensional data; however, neither could rival tree-based models.
- **Feature scaling is essential for regularized models.** Ridge and Lasso depend on uniform penalization across features, making `StandardScaler` a prerequisite; tree-based models are inherently scale-invariant yet still benefited from a consistent pipeline.

---
---

# Project 2: Health Risk Level Prediction

**Dataset:** `Health_Risk_Dataset.csv`
**Target Variable:** `Risk_Level` (4 classes: Low, Medium, High, Normal)
**Problem Type:** Multi-class Classification

---

## 1. Problem Definition

The goal of this project is to build a machine learning model that can classify a patient's health risk level based on their vital signs and clinical indicators. The output is one of four discrete categories — Low, Medium, High, or Normal — making this a **multi-class classification problem**, not a regression problem.

| Parameter | Detail |
|---|---|
| Target Variable (y) | `Risk_Level` — 4 classes: Low / Medium / High / Normal |
| Nature of Target | Categorical, ordinal; encoded as integers via `LabelEncoder` (0, 1, 2, 3) |

---

## 2. Data Understanding

The dataset is loaded from `Health_Risk_Dataset.csv`. Each row represents one patient record. The dataset includes a non-informative ID column, one mixed-type categorical column (`Consciousness`), several numerical vital sign measurements, and the target column `Risk_Level`. A correlation matrix is computed to understand feature relationships before modelling begins.

**Numerical Features (X):**
- `Respiratory_Rate`
- `Oxygen_Saturation`
- `O2_Scale`
- `Systolic_BP`
- `Heart_Rate`
- `Temperature`
- `On_Oxygen`

**Categorical Features (X):**
- `Consciousness` — Alert (A) / Responds to Pain (P) / Unresponsive (U)

**Target Variable (y):**
- `Risk_Level` — Normal / Low / Medium / High

**Class Distribution:**

| Class | Count |
|---|---|
| Medium | 306 |
| High | 279 |
| Low | 255 |
| Normal | 160 |

---

## 3. Data Pre-Processing

The raw dataset requires three targeted preprocessing steps before any model can be trained.

### Step 3a: Drop `Patient_ID` Column
`Patient_ID` is a unique row identifier with no predictive meaning. Including it would cause the model to memorise IDs instead of learning clinical patterns — a classic form of data leakage.

```python
df.drop(columns=['Patient_ID'])
```

### Step 3b: Label Encode `Consciousness`
`Consciousness` has 3 ordinal states (Alert < Confused < Unresponsive). `LabelEncoder` maps these to 0, 1, 2, preserving the inherent clinical order. One-Hot Encoding is avoided here because the ordinal relationship is medically meaningful.

```python
le_cons = LabelEncoder()
df['Consciousness'] = le_cons.fit_transform(df['Consciousness'])
```

### Step 3c: Label Encode Target
The 4-class target (Low, Medium, High, Normal) must be integer-encoded for sklearn classifiers. `le_target.classes_` is saved so original class names can be restored in classification reports and confusion matrix labels.

```python
le_target = LabelEncoder()
df['Risk_Level'] = le_target.fit_transform(df['Risk_Level'])
```

### Step 3d: Feature Scaling
Vital signs vary in magnitude; without standardization, larger values dominate Logistic Regression. The scaler is fit **only on `X_train`** to prevent data leakage and ensure model integrity.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

---

## 4. Splitting

The preprocessed feature matrix `X` and target vector `y` are split before scaling is applied, ensuring the scaler learns only from training statistics.

| Split Parameter | Value | Reasoning |
|---|---|---|
| `test_size` | 0.2 (20%) | Standard 80/20 split; enough test samples for reliable metric estimation across all 4 risk classes. |
| Train size | ~80% rows | Used to fit all models. Scaler is also fitted only on this subset. |
| Test size | ~20% rows | Completely held-out. Used only for `predict()` and metric calculation. |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

> **Note:** `stratify=y` is used to preserve the class distribution across the train/test split, which is critical given the imbalanced class counts.

---

## 5. Model Training

Six classification models are trained, spanning from simple linear classifiers to ensemble and boosting methods.

### 1. Logistic Regression (Baseline — No Penalty)
An unregularized multinomial baseline using Softmax to model four classes simultaneously. Serves as a benchmark; if regularization provides no improvement, the features are already well-separated.

```python
m1 = LogisticRegression(
    penalty=None, multi_class='multinomial',
    solver='lbfgs', max_iter=10000, random_state=42
)
m1.fit(X_train_scaled, y_train)
```

### 2. Logistic Regression — L1 Penalty
Using a SAGA solver, this L1-regularized model (`C=0.1`) performs feature selection by zeroing out redundant coefficients. Ideal for health data where overlapping vital signs often provide duplicate information.

```python
m2 = LogisticRegression(
    penalty='l1', solver='saga', multi_class='multinomial',
    max_iter=5000, C=0.1, random_state=42
)
m2.fit(X_train_scaled, y_train)
```

### 3. Logistic Regression — L2 Penalty
This L2-regularized model (`C=0.01`) shrinks coefficients toward zero without eliminating them. Used to determine if keeping all features at small weights outperforms the selection-heavy L1 approach.

```python
m3 = LogisticRegression(
    penalty='l2', solver='lbfgs', multi_class='multinomial',
    max_iter=1000, C=0.01, random_state=42
)
m3.fit(X_train_scaled, y_train)
```

### 4. Decision Tree Classifier
A rule-based classifier using a max depth of 5 to ensure medical interpretability and prevent overfitting, mirroring clinical diagnostic logic.

```python
m4 = DecisionTreeClassifier(max_depth=5, random_state=42)
m4.fit(X_train_scaled, y_train)
```

### 5. Bagging Classifier
By averaging 50 diverse trees, this ensemble reduces variance and stabilizes predictions, balancing computational speed with robust, reliable accuracy.

```python
m5 = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50, random_state=42
)
m5.fit(X_train_scaled, y_train)
```

### 6. XGBoost Classifier
This gradient boosting model builds trees sequentially, with each new tree correcting errors from previous ones. It offers high performance and efficiency through advanced regularization and optimized computation.

```python
m_xgb = XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric='mlogloss'
)
m_xgb.fit(X_train_scaled, y_train)
```

---

## 6. Model Testing and Evaluation

Each model is evaluated on the held-out test set using four complementary methods: a full classification report, per-class confusion matrices, a comparative bar chart, and macro-averaged ROC curves with AUC scores.

**Evaluation Metrics Explained:**

| Metric | Explanation |
|---|---|
| Precision (macro avg) | Macro average precision measures correct class predictions, emphasizing low false positives. |
| Recall (macro avg) | Recall measures the fraction of true cases correctly identified, which is vital for catching all critical patients. |
| F1-Score (macro avg) | The harmonic mean of precision and recall, balancing both metrics. Used here to ensure the model doesn't ignore minority risk classes. |
| ROC + AUC | Evaluates each class binary, averaging the results. Using probability scores, it compares all models to determine discrimination excellence. |

**Final Results — Ranked by Macro F1-Score:**

| Rank | Model | Precision | Recall | F1-Score | AUC | Accuracy |
|---|---|---|---|---|---|---|
| 1 | XGBoost | 0.9612 | 0.9563 | 0.9586 | 0.9989 | 0.955 |
| 2 | Logistic Regression | 0.9614 | 0.9501 | 0.9546 | 0.9964 | 0.955 |
| 3 | Logit L1 | 0.9450 | 0.9378 | 0.9408 | 0.9959 | 0.940 |
| 4 | Logit L2 | 0.9253 | 0.8972 | 0.9078 | 0.9808 | 0.910 |
| 5 | Bagging DT | 0.9145 | 0.8994 | 0.9059 | 0.9870 | 0.900 |
| 6 | Decision Tree | 0.9060 | 0.8916 | 0.8978 | 0.9615 | 0.890 |

---

## 7. Conclusion

1. **Stratified splitting ensured fair class representation.** With 4 imbalanced classes — Medium (306), High (279), Low (255), and Normal (160) — using `stratify=y` preserved each class's proportion in the test set, making macro F1 and AUC scores reliable across all risk groups, especially the minority Normal class.

2. **XGBoost emerged as the best overall model**, achieving the highest macro F1 (0.959) and AUC (0.999), along with 95.5% accuracy — demonstrating that gradient boosting captures complex, non-linear interactions between vital signs far better than linear or shallow tree-based models.

3. **Logistic Regression variants revealed clear regularisation tradeoffs.** Vanilla LR (F1 = 0.955) outperformed L1 (0.941) and L2 (0.908), showing that aggressive regularisation hurt performance here — L2 at `C=0.01` notably degraded recall for Low and Normal classes by over-shrinking informative feature weights.

4. **Bagging improved on a single Decision Tree but couldn't close the gap with linear models.** The single Decision Tree (F1 = 0.898) was the weakest model; 50 bootstrapped trees in the `BaggingClassifier` raised the macro F1 to 0.906, confirming variance reduction — but both lagged behind even L2 Logistic Regression.

5. **High AUC across all models confirmed strong probabilistic separation of risk levels.** Every model scored above 0.96 AUC, meaning all models rank patients reliably by risk — making threshold tuning at deployment a viable strategy to further optimise recall for the Critical and High risk groups in a hospital triage context.

---


  <h2 id="mentor">Mentor</h2>
  <p><strong>Dr. Sahinur Rahman Laskar</strong><br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
