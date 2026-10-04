# Team 10 ML Project — Wine Quality Classification

## Project Objective

This project uses a **Random Forest Classifier** to predict wine quality and compares three evaluation strategies:

1. Single 80/20 train-test split
2. 30 repeated random 80/20 splits
3. 5-fold stratified cross-validation

The goal is to study the model's **accuracy, stability, generalization, and computational cost** under different evaluation methods.

---

## 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
```

### `import pandas as pd`
- **What:** Imports Pandas.
- **Purpose:** Handles datasets in DataFrame/table form.
- **Use:** Reading the CSV, selecting columns, creating result tables, and calculating statistics.

### `import numpy as np`
- **What:** Imports NumPy.
- **Purpose:** Numerical operations.
- **Use:** Operations such as `np.nan` and `np.arange()`.

### `import matplotlib.pyplot as plt`
- **What:** Imports Matplotlib plotting functions.
- **Purpose:** Creates graphs.
- **Use:** Accuracy, performance-comparison, and computational-time graphs.

### `import time`
- **What:** Imports Python's time module.
- **Purpose:** Measures execution time.
- **Use:** Compares how long each evaluation method takes.

---

## 2. Machine-Learning Imports

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
```

### `train_test_split`
Divides the data into training and testing portions.

```text
Dataset
   |
   +---- 80% Training
   |
   +---- 20% Testing
```

### `StratifiedKFold`
Used for 5-fold cross-validation while maintaining a similar class distribution across folds.

---

```python
from sklearn.ensemble import RandomForestClassifier
```

Imports the Random Forest classification algorithm.

```text
Tree 1
Tree 2
Tree 3
...
Tree 100
   |
Random Forest
   |
Prediction
```

---

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

Imports evaluation metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | How many predicted samples were actually correct |
| Recall | How many actual samples were correctly identified |
| F1 Score | Balance between precision and recall |

---

## 3. Loading and Cleaning the Dataset

```python
df = pd.read_csv("winequality-white.csv", sep=";")
```

Reads the wine-quality CSV file into a Pandas DataFrame called `df`.

`sep=";"` is used because the dataset uses semicolons to separate columns.

```python
print("Original Dataset Shape:", df.shape)
```

Displays the number of rows and columns.

### Duplicate detection

```python
duplicate_count = df.duplicated().sum()
```

- `duplicated()` checks for duplicate rows.
- `sum()` counts them.

### Missing-value detection

```python
missing_rows = df.isnull().any(axis=1).sum()
```

Checks whether rows contain missing values and counts those rows.

### Remove duplicates

```python
df = df.drop_duplicates()
```

Removes duplicate records.

### Remove missing rows

```python
df = df.dropna()
```

Removes rows containing missing values.

### Final shape

```python
print("Final Dataset Shape:", df.shape)
```

Shows the remaining dataset size after cleaning.

---

## 4. Selecting Features and Target

```python
selected_features = [
    "alcohol",
    "volatile acidity",
    "density",
    "chlorides",
    "sulphates",
    "citric acid"
]
```

These are the six selected input features.

```text
Alcohol
Volatile Acidity
Density
Chlorides
Sulphates
Citric Acid
        |
        v
Random Forest
        |
        v
Wine Quality
```

### Creating `X`

```python
X = df[selected_features]
```

`X` contains the input features.

### Creating `y`

```python
y = df["quality"]
```

`y` contains the target/output variable.

```text
X = Features / Inputs
y = Target / Output
```

---

## 5. Creating the Random Forest Model

```python
def create_model():
```

Creates a function so the same Random Forest configuration can be created repeatedly.

```python
return RandomForestClassifier(
```

Creates and returns a Random Forest classifier.

```python
n_estimators=100,
```

Creates a Random Forest with **100 decision trees**.

```python
random_state=42
```

Fixes the random seed so the experiment is reproducible.

---

## 6. Metrics Function

```python
def calculate_metrics(y_test, y_pred):
```

Creates a function that calculates metrics from actual test values and model predictions.

### Accuracy

```python
"Accuracy": accuracy_score(y_test, y_pred),
```

Calculates the proportion of correct predictions.

### Precision

```python
precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
```

Calculates weighted precision across the wine-quality classes.

- `average="weighted"` accounts for the number of samples in each class.
- `zero_division=0` prevents an error when a class has no predicted samples.

### Recall

```python
recall_score(...)
```

Measures how many actual samples were successfully identified.

### F1 Score

```python
f1_score(...)
```

Combines precision and recall into one score.

---

## 7. Single 80/20 Train-Test Split

```python
start_time = time.perf_counter()
```

Starts a timer to measure execution time.

### Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Creates:

```text
100% Dataset
     |
     +---- 80% Training
     |
     +---- 20% Testing
```

- `X` = input features
- `y` = target
- `test_size=0.20` = 20% test data
- `random_state=42` = reproducible split
- `stratify=y` = maintains approximately similar target-class distribution

### Train

```python
model = create_model()
```

Creates a Random Forest.

```python
model.fit(X_train, y_train)
```

Trains the model using the training data.

### Predict

```python
y_train_pred = model.predict(X_train)
```

Predicts the training data.

```python
y_test_pred = model.predict(X_test)
```

Predicts unseen test data.

### Training accuracy

```python
single_train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)
```

Calculates training accuracy.

### Test metrics

```python
single_results = calculate_metrics(
    y_test,
    y_test_pred
)
```

Calculates test Accuracy, Precision, Recall, and F1 Score.

### Time

```python
single_time = time.perf_counter() - start_time
```

Calculates the execution time.

### Generalization gap

```python
single_train_accuracy - single_results["Accuracy"]
```

Compares training accuracy with test accuracy.

A large gap can indicate possible overfitting.

---

## 8. 30 Repeated Random 80/20 Splits

```python
N_SPLITS = 30
```

Specifies 30 repeated experiments.

### Why repeat?
One random split may give a lucky or unlucky result. Repeating the split shows how much the result changes.

```python
random_results = []
```

Creates an empty list for storing results.

```python
start_time = time.perf_counter()
```

Starts timing.

### Loop

```python
for random_state in range(1, N_SPLITS + 1):
```

Runs the experiment 30 times using random states 1 through 30.

For every iteration, the code:

1. Splits the data.
2. Creates a new Random Forest.
3. Trains it.
4. Predicts training and testing data.
5. Calculates metrics.
6. Stores the result.

The split uses:

```python
random_state=random_state
```

so each iteration can use a different random train-test arrangement.

### Store results

```python
random_results.append({
```

Stores values such as:

- Split
- Accuracy
- Precision
- Recall
- F1 Score
- Training Accuracy
- Generalization Gap

The generalization gap is:

```python
train_accuracy - metrics["Accuracy"]
```

---

## 9. Random-Split Results DataFrame

```python
random_df = pd.DataFrame(random_results)
```

Converts the 30 results into a table.

Example:

| Split | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| 30 | ... | ... | ... | ... |

### Mean

```python
random_df[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].mean()
```

Calculates average performance over the 30 splits.

### Standard deviation

```python
.std()
```

Measures how much the results vary.

- Small standard deviation → more stable results.
- Large standard deviation → more variation between splits.

### Minimum and maximum

```python
.min()
.max()
```

Find the lowest and highest observed performance.

### Range

```text
Maximum - Minimum
```

Shows the difference between the best and worst observed performance.

---

## 10. 5-Fold Stratified Cross-Validation

```python
kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Creates 5 stratified folds.

Conceptually:

```text
Dataset
-------------------------
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Each round uses four folds for training and one fold for testing.

```text
Round 1:
Train = 2,3,4,5
Test  = 1

Round 2:
Train = 1,3,4,5
Test  = 2

...

Round 5:
Train = 1,2,3,4
Test  = 5
```

- `n_splits=5` → five folds.
- `shuffle=True` → shuffles before creating folds.
- `random_state=42` → reproducible shuffling.

---

## 11. Cross-Validation Loop

```python
for fold, (train_index, test_index) in enumerate(
    kf.split(X, y),
    start=1
):
```

Generates training and testing indexes for each fold.

`start=1` gives fold numbers 1–5.

### Select data

```python
X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]
```

Uses the generated indexes to select training and testing rows.

### Train and predict

```python
model = create_model()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

Creates, trains, and evaluates a new Random Forest for the current fold.

### Store metrics

```python
cv_results.append({
```

Stores:

- Accuracy
- Precision
- Recall
- F1 Score
- Training Accuracy
- Generalization Gap

---

## 12. Cross-Validation Results

```python
cv_df = pd.DataFrame(cv_results)
```

Converts the five fold results into a DataFrame.

Statistics such as:

```python
.mean()
.std()
.min()
.max()
```

can then be calculated.

---

## 13. Final Comparison

The project compares:

```text
Single 80/20 Split
30 Random 80/20 Splits
5-Fold Cross-Validation
```

The comparison considers:

- Mean Accuracy
- Standard Deviation
- Precision
- Recall
- F1 Score
- Computational Time

### Mean Accuracy

Compares:

```text
Single split accuracy
        vs
Average of 30 random splits
        vs
Average of 5 CV folds
```

### Standard Deviation

Shows the variation in repeated results.

For a single split, standard deviation is not available because there is only one result. This can be represented by:

```python
np.nan
```

---

## 14. Accuracy Graph

```python
plt.figure(figsize=(9, 5))
```

Creates the plotting area.

```python
plt.plot(
    random_df["Split"],
    random_df["Accuracy"],
    marker="o"
)
```

Plots accuracy for each of the 30 random splits.

### Purpose

Shows whether accuracy is stable or changes significantly across different random splits.

### Mean line

```python
plt.axhline(
    random_df["Accuracy"].mean(),
    linestyle="--",
    label="Mean Accuracy"
)
```

Adds a horizontal line showing the mean accuracy.

Other plotting commands:

```python
plt.title(...)
plt.xlabel(...)
plt.ylabel(...)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()
```

These respectively:

- Add a title
- Label the X-axis
- Label the Y-axis
- Set accuracy range from 0 to 1
- Add grid lines
- Show the legend
- Display the graph

---

## 15. Performance Comparison Graph

```python
methods = [
    "Single Split",
    "30 Random Splits",
    "5-Fold CV"
]
```

Stores the names of the three methods.

```python
mean_accuracy = [...]
```

Stores accuracy values.

```python
mean_f1 = [...]
```

Stores F1 values.

```python
x = np.arange(len(methods))
```

Creates numerical positions for the three methods.

```python
width = 0.35
```

Controls bar width.

### Accuracy bars

```python
plt.bar(
    x - width/2,
    mean_accuracy,
    width,
    label="Accuracy"
)
```

Creates accuracy bars.

### F1 bars

```python
plt.bar(
    x + width/2,
    mean_f1,
    width,
    label="F1 Score"
)
```

Creates F1 bars beside the accuracy bars.

### Purpose

Makes it easy to compare Accuracy and F1 Score for the three evaluation methods.

---

## 16. Computational-Time Graph

```python
times = [
    single_time,
    random_time,
    cv_time
]
```

Stores the execution times.

```python
plt.bar(
    methods,
    times
)
```

Creates a bar graph comparing computational cost.

### Purpose

Shows which evaluation method requires more or less computation.

---

## 17. Final Printed Summary

### Single Split

```python
print(
    f"Single Split Test Accuracy       : "
    f"{single_results['Accuracy']:.4f}"
)
```

Prints the test accuracy from the single 80/20 split.

### Repeated Random Splits

```python
print(
    f"Repeated Split Mean Test Accuracy: "
    f"{random_df['Accuracy'].mean():.4f}"
)
```

Prints the average accuracy over 30 random splits.

### 5-Fold CV

```python
print(
    f"5-Fold CV Mean Test Accuracy     : "
    f"{cv_df['Accuracy'].mean():.4f}"
)
```

Prints the average accuracy over the five folds.

---

# 18. Complete Project Working

```text
                 WINE DATASET
                      |
                      v
                Data Cleaning
                      |
          +-----------+-----------+
          |                       |
    Remove Duplicates       Remove Missing Values
          |                       |
          +-----------+-----------+
                      |
                      v
              Select Features
                      |
          +-----------+-----------+
          |           |           |
       Alcohol    Density      etc.
          |           |           |
          +-----------+-----------+
                      |
                      v
              Target = Quality
                      |
                      v
             Random Forest Model
                 100 Trees
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Single      30 Random    5-Fold
       80/20       80/20        Cross-
       Split       Splits       Validation
          |           |           |
          +-----------+-----------+
                      |
                      v
              Evaluate Model
                      |
          +-----------+-----------+
          |           |           |
       Accuracy   Precision     Recall
                      |
                      v
                   F1 Score
                      |
                      v
              Compare Methods
                      |
          +-----------+-----------+
          |                       |
    Performance             Computational
    Comparison                   Cost
          |
          v
      Final Results
```

---

# 19. Main Purpose of the Project

> **We use a Random Forest classifier to predict wine quality and compare single train-test splitting, repeated random splitting, and 5-fold cross-validation to study the accuracy, stability, generalization, and computational cost of model evaluation.**
