# Wine Quality Prediction with Iterative Feature Selection

An exploratory machine-learning project using the UCI White Wine Quality dataset. This version extends an earlier classification-versus-regression study by adding a validation-based, iterative feature-removal procedure using permutation importance.

The original notebook implementation is preserved: model-search functions, feature-selection logic, evaluation functions, parameter grids, and executable code have not been rewritten. The notebook has been organized with explanatory headings and documentation.

## Project question

Can physicochemical measurements predict the sensory quality score of white wine, and does iterative removal of low-importance features improve predictive performance?

The target is an ordered, integer-valued quality score. The project explores both:

- **Classification:** predict the discrete quality category directly.
- **Regression:** predict a numerical score and evaluate whether it falls within a tolerance of the true score.

## Dataset

- **Source:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Dataset used:** White wine
- **Observations:** 4,898
- **Input features:** 11 numerical physicochemical properties
- **Target:** sensory quality score

The notebook expects the extracted file at:

```text
data/winequality-white.csv
```

The data folder in this repository includes a README explaining the expected file location. The CSV itself is not included. Download the white-wine data from UCI and place it at the path above.

## Repository structure

```text
wine-quality-feature-selection/
├── data/
│   └── README.md
├── docs/
│   └── trexquant_presentation.pdf
├── notebooks/
│   └── wine_quality_feature_selection.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Workflow

### 1. Data split

The code uses a three-way split:

- 70% training data for model fitting and hyperparameter tuning.
- 15% validation data for permutation-based feature selection.
- 15% held-out test data for final evaluation.

The first split is stratified by the quality target. The second split is performed using the original `train_test_split` call without an explicit `stratify` argument.

### 2. Preprocessing

The shared numerical pipeline applies:

1. Median imputation.
2. Standard scaling.

The preprocessing pipeline is included inside each estimator pipeline, so scaling and imputation are fitted within each cross-validation training fold.

### 3. Models

The notebook evaluates six model configurations:

| Task | Model | Grid-search scoring |
|---|---|---|
| Classification | Logistic Regression | Accuracy |
| Classification | Random Forest Classifier | Accuracy |
| Classification | SVM with RBF kernel | Accuracy |
| Regression | Ridge Regression | Accuracy within tolerance |
| Regression | Random Forest Regressor | Accuracy within tolerance |
| Regression | SVR with RBF kernel | Accuracy within tolerance |

For the supplied regression scorer, a prediction is counted as correct when:

\[
|y_{\mathrm{true}}-y_{\mathrm{pred}}| \leq T,
\]

with default tolerance \(T=0.5\). The prediction is clipped to the interval 1–9 before the tolerance comparison. This is the metric used for regression hyperparameter selection and final regression accuracy in the notebook; the regression grid searches do not optimize \(R^2\).

### 4. Iterative feature removal

The `feature_selection` function starts with all 11 features and repeatedly:

1. Fits the selected model-search function on the current training feature set.
2. Computes permutation importance on the validation data.
3. Identifies the feature with the smallest mean permutation importance.
4. Removes that feature and refits the model.
5. Evaluates the reduced feature set on validation data.
6. Retains the best-performing validation feature set.

The implementation uses 10 permutation repeats and a minimum of five features. Its stopping condition is governed by the decline counter in the original code. This is a greedy, one-feature-at-a-time procedure; it does not exhaustively search all possible feature subsets. The function returns the importance DataFrame computed in its final loop iteration; this table is not guaranteed to correspond to the feature subset retained as the best validation-scoring subset.

### 5. Final test

After feature selection, `fnl_tst` combines the training and validation data, clones and refits the selected grid-search object on the combined data, and predicts the held-out test set. The test set is not used by the feature-selection function.

## Results from the supplied notebook

The table below transcribes the saved outputs in the attached notebook. These are the notebook's recorded results, not fresh executions performed while packaging the repository.

| Model | Task | Test metric | Selected features |
|---|---|---:|---:|
| Ridge Regression | Regression | 52.79% within \(T=0.5\) | 11 |
| SVR (RBF) | Regression | 66.12% within \(T=0.5\) | 11 |
| Random Forest Regressor | Regression | **67.76% within \(T=0.5\)** | **6** |
| Logistic Regression | Classification | 54.69% accuracy | 9 |
| SVM (RBF) | Classification | 62.31% accuracy | 6 |
| Random Forest Classifier | Classification | 65.17% accuracy | 10 |

The saved dummy-model benchmark outputs are:

| Baseline | Test metric |
|---|---:|
| Dummy Regressor (mean) | 46% within \(T=0.5\) |
| Dummy Classifier (most frequent) | 46% accuracy |

**Main observation:** Random Forest Regression achieved the strongest test-set result among the models in the supplied notebook, with approximately 67.76% of predictions within \(T=0.5\), using six selected features. The conclusion is consistent with the accompanying presentation: Random Forest Regression performs strongly in this experiment, while classification is also a viable formulation for the ordered target.

The accompanying presentation contains results from a different run (for example, it reports 69.39% for Random Forest Regression). Since the user notes that numerical values changed on rerunning the code, this README uses the saved notebook outputs as the current code-associated results. The presentation is included as supplementary context rather than as the source of the table above.

## Feature-importance interpretation

Permutation importance measures how much the chosen validation score changes when the values of one feature are shuffled. A larger positive importance indicates greater reliance of the fitted model on that feature for predictive performance on the validation set.

Permutation importance is model-agnostic, but it should not be interpreted as a causal effect or an intrinsic measure of scientific importance. Correlated features can share predictive information, which can affect their individual permutation-importance scores.

## Limitations and possible extensions

- **Class imbalance:** quality scores 3 and 9 are rare in this dataset. The saved Random Forest regression confusion matrix shows that the model struggles to predict these minority classes.
- **Greedy feature selection:** removing one least-important feature at a time can miss useful feature combinations and interactions.
- **Validation-score variability:** feature selection uses a single validation split. Repeated splits or repeated cross-validation could help assess the stability of selected features and model performance.
- **Metric choice:** regression is optimized using accuracy within \(T=0.5\), not a conventional regression loss or \(R^2\). Other tolerances and metrics could lead to different model choices.
- **Model coverage:** the notebook explores Logistic Regression, Ridge Regression, Random Forest, SVC, and SVR. It is not an exhaustive comparison of possible algorithms.
- **Permutation importance:** rankings can be affected by correlated input variables and the particular validation sample.

Potential next steps include repeated-split stability analysis, direct comparison of alternative tolerance values, more careful treatment of class imbalance, and a broader study of feature interactions.

## Reproducing the notebook

Python 3.10 or newer is recommended. The notebook uses scikit-learn components including `TunedThresholdClassifierCV` and `TargetEncoder` in its import cell, so use scikit-learn 1.5 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `winequality-white.csv` in `wine+quality/`, launch Jupyter, and open:

```text
notebooks/wine_quality_feature_selection.ipynb
```

Run the notebook from top to bottom.

## Supplementary presentation

The `docs/trexquant_presentation.pdf` file contains the accompanying presentation, including the workflow diagram, model comparison, Random Forest regression confusion matrix, and validation-set permutation-importance plot.

## Acknowledgement

The Wine Quality dataset is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). The original dataset study is Cortez et al., *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems 47 (2009), 547–553, DOI: 10.1016/j.dss.2009.05.016.
