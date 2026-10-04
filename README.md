# Applied-AI-Project1-churn
### Week 2: Building ML Models

In Week 2, different machine learning models were developed and evaluated for customer churn prediction. A baseline model was first created by predicting that every customer would stay. This baseline achieved an accuracy of 73.5%. However, because approximately 26.5% of the customers are churners, the baseline completely failed to identify churners, resulting in a recall of 0%. This shows that accuracy alone is not sufficient for evaluating a churn prediction model.

Logistic Regression and Random Forest were the best-performing models, with both achieving a test AUC of approximately 0.842. At the default probability threshold of 0.5, Logistic Regression achieved 80.7% accuracy, 65.8% precision, and 56.7% recall. Random Forest achieved 80.7% accuracy, 67.3% precision, and 52.9% recall. These results show that both models were able to identify a significant portion of customers who were likely to churn.

The Random Forest permutation-importance analysis identified the main factors associated with churn prediction. The three most important features were Tenure, TotalCharges, and Contract_Two year. Other important features included InternetService_Fiber optic and Contract_One year. These features provide useful information about which customer characteristics contribute most to the model's predictions.

The probability threshold was also analyzed because the cost of missing a potential churner is higher than the cost of making an unnecessary retention offer. The empirical business-cost analysis selected a threshold of 0.15, while the theoretical threshold was approximately 0.14. This calculation was based on an assumed cost of PKR 6,000 for a missed churner, or false negative, and PKR 1,000 for an unnecessary retention offer, or false positive. Therefore, a lower threshold can be used to identify more potential churners.

The threshold analysis demonstrated an important trade-off between recall and precision. For example, at a threshold of 0.2, churn recall increased to 85.6%, compared with 56.7% at the default threshold of 0.5. However, precision decreased from 65.8% to 46.7%. This means that lowering the threshold allows the model to identify more potential churners, but it also results in more customers being incorrectly classified as likely to churn.

Class imbalance was another important consideration. The balanced Logistic Regression model increased recall from 56.7% to 78.1%, while precision decreased from 65.8% to 50.5%. This demonstrates the trade-off between identifying a larger number of churners and generating more false-positive predictions.

Feature engineering was also performed by creating four new features: n_services, is_new, charge_per_mo, and price_jump. However, these engineered features did not improve the Random Forest model's performance. The AUC changed only slightly from 0.8422 before feature engineering to 0.8420 after feature engineering. Therefore, the additional features did not provide a meaningful improvement in predictive performance.

A Decision Tree model was also examined to understand the effect of tree depth and overfitting. Increasing the tree depth improved training accuracy, but after a certain point, the test accuracy decreased. For example, an unrestricted Decision Tree achieved 99.8% training accuracy but only 74.2% test accuracy, which indicates overfitting. In comparison, a depth-5 Decision Tree achieved 80.1% training accuracy and 79.4% test accuracy, showing a smaller gap between training and test performance.

The main lesson from Week 2 is that accuracy alone is not enough when evaluating a customer churn prediction model. Precision, recall, F1-score, AUC, class imbalance, probability threshold selection, feature importance, overfitting, and business costs must also be considered. The model should therefore be evaluated not only according to its statistical performance but also according to the practical business objective of identifying customers who are likely to churn.

# Week 3: Model Optimization and Unsupervised Learning

## Dataset and Preprocessing
The project uses the **Telco Customer Churn dataset** containing:

- Total customers = **7,043**
- Original features = **21 columns**
- Features after one-hot encoding = **30**
- Training samples = **5,634**
- Test samples = **1,409**
- Churn rate in training data = **26.5%**
- Churn rate in test data = **26.5%**

The target variable was converted into binary form:

\[
y =
\begin{cases}
1, & \text{if Churn = Yes}\\
0, & \text{if Churn = No}
\end{cases}
\]

The data was split using an 80:20 stratified train-test split.

---

## Model Performance

The baseline model always predicts the majority class.

**Baseline accuracy:**

\[
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

\[
Accuracy = \frac{1035}{1409} \approx 0.735
\]

Therefore:

**Baseline accuracy = 73.5%**

### Logistic Regression

The Logistic Regression model achieved:

- Accuracy = **80.7%**
- Precision = **65.8%**
- Recall = **56.7%**
- F1-score = **60.9%**
- Test AUC = approximately **0.842**

The Logistic Regression probability is calculated using the sigmoid function:

\[
P(Y=1|X)=\frac{1}{1+e^{-z}}
\]

where

\[
z=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n
\]

For one test customer, the notebook obtained:

\[
z=-3.070
\]

Therefore:

\[
P=\frac{1}{1+e^{-(-3.070)}}\approx0.0444
\]

So the predicted probability of churn for that customer was approximately:

\[
\boxed{4.44\%}
\]

---

## Random Forest Optimization

The Random Forest model used:

- Number of trees = **300**
- `max_features = sqrt`
- `min_samples_leaf = 5`
- OOB evaluation enabled
- `random_state = 42`

Results:

- OOB accuracy = **80.3%**
- Test accuracy = **80.7%**
- Test AUC = **0.842**

The Random Forest achieved approximately the same AUC as Logistic Regression.

### Feature Importance

The most important features according to permutation importance were:

1. **Tenure** = 0.0404
2. **TotalCharges** = 0.0206
3. **Contract_Two year** = 0.0164
4. **InternetService_Fiber optic** = 0.0132
5. **Contract_One year** = 0.0059

This indicates that customer tenure and billing/contract-related variables are important for predicting churn.

---

## Confusion Matrix Calculation

For Logistic Regression, the notebook obtained:

- True Negative (TN) = **925**
- False Positive (FP) = **110**
- False Negative (FN) = **162**
- True Positive (TP) = **212**

### Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
Accuracy=\frac{212+925}{212+925+110+162}
\]

\[
Accuracy=\frac{1137}{1409}=0.807
\]

\[
\boxed{Accuracy=80.7\%}
\]

### Precision

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Precision=\frac{212}{212+110}
=\frac{212}{322}
\approx0.658
\]

\[
\boxed{Precision=65.8\%}
\]

### Recall

\[
Recall=\frac{TP}{TP+FN}
\]

\[
Recall=\frac{212}{212+162}
=\frac{212}{374}
\approx0.567
\]

\[
\boxed{Recall=56.7\%}
\]

### F1-score

\[
F1=2\frac{Precision\times Recall}{Precision+Recall}
\]

\[
F1=2\frac{(0.658)(0.567)}{0.658+0.567}
\approx0.609
\]

\[
\boxed{F1=60.9\%}
\]

---

## Class-Balanced Logistic Regression

Because churn is the minority class, a balanced Logistic Regression model was also tested.

Results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.7% | 65.8% | 56.7% | 60.9% |
| Balanced Logistic Regression | 73.9% | 50.5% | 78.1% | 61.3% |

The balanced model increased recall from:

\[
56.7\% \rightarrow 78.1\%
\]

but reduced precision and overall accuracy.

This shows the trade-off between identifying more churn customers and producing fewer false alarms.

---

## Feature Engineering

Additional features were created:

### Number of Services

\[
n\_services=\sum_{i=1}^{7} I(Service_i=Yes)
\]

where \(I(\cdot)\) is an indicator function.

### New Customer Indicator

\[
is\_new=
\begin{cases}
1,& tenure\leq12\\
0,& tenure>12
\end{cases}
\]

### Charge Per Month

\[
charge\_per\_mo=
\frac{TotalCharges}{\max(tenure,1)}
\]

The `max(tenure,1)` operation prevents division by zero.

### Price Jump

\[
price\_jump=MonthlyCharges-charge\_per\_mo
\]

However, after feature engineering, the Random Forest AUC changed only slightly:

\[
AUC_{before}=0.8422
\]

\[
AUC_{after}=0.8420
\]

Change:

\[
\Delta AUC=0.8420-0.8422=-0.0002
\]

Thus, the additional engineered features did **not improve the Random Forest model** on this test split.

---

## Week 3 Required Metrics

### Split-to-split accuracy across 20 seeds

**Not available in the uploaded notebook.**

The notebook uses a single train-test split:

\[
random\_state=42
\]

Therefore, a minimum-to-maximum accuracy range across 20 seeds cannot be calculated from the current notebook.

**Current available test accuracy: 80.7%.**

---

### 5-fold Cross-Validation AUC

The uploaded notebook does **not implement 5-fold cross-validation**.

Therefore:

- LR CV AUC = **Not calculated**
- RF CV AUC = **Not calculated**
- XGBoost CV AUC = **Not calculated**

The available single test-split AUC values are:

\[
AUC_{LR}\approx0.842
\]

\[
AUC_{RF}=0.842
\]

---

### Hyperparameter Tuning

The uploaded notebook does not contain GridSearchCV or RandomizedSearchCV.

The Random Forest parameters actually used were:

\[
n\_estimators=300
\]

\[
max\_features=\text{'sqrt'}
\]

\[
min\_samples\_leaf=5
\]

\[
random\_state=42
\]

Therefore:

- Best RF parameters from grid search = **Not calculated**
- Grid Search time = **Not calculated**
- Random Search time = **Not calculated**

---

### Final Model Test AUC

The Random Forest achieved:

\[
\boxed{Test\ AUC=0.842}
\]

The feature-engineered Random Forest achieved:

\[
AUC=0.8420
\]

Therefore, the original Random Forest was retained as the better of the two tested RF versions.

---

## Customer Segmentation

Customer segmentation using K-Means was **not implemented in the uploaded notebook**.

Therefore:

\[
k=\text{Not calculated}
\]

and customer segment names/churn percentages cannot be reported without performing the clustering analysis.

---

## PCA

PCA was **not implemented in the uploaded notebook**.

The dataset contains **30 encoded features**, but the notebook does not calculate how many principal components explain 90% of the variance.

Therefore:

\[
n\text{ components for 90\% variance}=\text{Not calculated}
\]

The PCA calculation would normally use:

\[
ExplainedVarianceRatio_i=
\frac{\lambda_i}{\sum_{j=1}^{30}\lambda_j}
\]

and the required number of components \(n\) would be the smallest value satisfying:

\[
\sum_{i=1}^{n} ExplainedVarianceRatio_i \geq 0.90
\]

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 73.5% | 0.0% | 0.0% | 0.0% | 0.500 |
| Logistic Regression | 80.7% | 65.8% | 56.7% | 60.9% | 0.842 |
| Balanced LR | 73.9% | 50.5% | 78.1% | 61.3% | 0.841 |
| Decision Tree | 79.6% | 63.2% | 55.1% | 58.9% | 0.829 |
| Random Forest | **80.7%** | **67.3%** | 52.9% | 59.3% | **0.842** |

---

## Biggest Lesson

**The biggest lesson is that model optimization should be based on the appropriate evaluation metric: Random Forest and Logistic Regression achieved similar AUC (≈0.842), while class balancing greatly improved churn recall from 56.7% to 78.1% at the cost of lower precision and accuracy.**
