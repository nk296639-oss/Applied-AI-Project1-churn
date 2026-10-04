# Applied-AI-Project1-churn
### Week 2: Building ML Models

In Week 2, different machine learning models were developed and evaluated for customer churn prediction. A baseline model was first created by predicting that every customer would stay. This baseline achieved an accuracy of 73.5%. However, because approximately 26.5% of the customers are churners, the baseline completely failed to identify churners, resulting in a recall of 0%. This shows that accuracy alone is not sufficient for evaluating a churn prediction model.

Logistic Regression and Random Forest were the best-performing models, with both achieving a test AUC of approximately 0.842. At the default probability threshold of 0.5, Logistic Regression achieved 80.7% accuracy, 65.8% precision, and 56.7% recall. Random Forest achieved 80.7% accuracy, 67.3% precision, and 52.9% recall. These results show that both models were able to identify a significant portion of customers who were likely to churn.

The Random Forest permutation-importance analysis identified the main factors associated with churn prediction. The three most important features were Tenure, TotalCharges, and Contract_Two year. Other important features included InternetService_Fiber optic and Contract_One year. These features provide useful information about which customer characteristics contribute most to the model's predictions.

The probability threshold was also analyzed because the cost of missing a potential churner is higher than the cost of making an unnecessary retention offer. The empirical business-cost analysis selected a threshold of 0.15, while the theoretical threshold was approximately 0.14. This calculation was based on an assumed cost of PKR 6,000 for a missed churner, or false negative, and PKR 1,000 for an unnecessary retention offer, or false positive. Therefore, a lower threshold can be used to identify more potential churners.

The threshold analysis demonstrated an important trade-off between recall and precision. For example, at a threshold of 0.2, churn recall increased to 85.6%, compared with 56.7% at the default threshold of 0.5. However, precision decreased from 65.8% to 46.7%. This means that lowering the threshold allows the model to identify more potential churners, but it also results in more customers being incorrectly classified as likely to churn.

Class imbalance was another important consideration. The balanced Logistic Regression model increased recall from 56.7% to 78.1%, while precision decreased from 65.8% to 50.5%. This demonstrates the trade-off between identifying a larger number of churners and generating more false-positive predictions.

Feature engineering was also performed by creating four new features: n_services, is_new, charge_per_mo, and price_jump. However, these engineered features did not improve the Random Forest model's performance. The AUC changed only slightly from 0.8422 before feature engineering to 0.8420 after feature engineering. Therefore, the additional features did not provide a meaningful improvement in predictive performance.

A Decision Tree model was also examined to understand the effect of tree depth and overfitting. Increasing the tree depth mproved training accuracy, but after a certain point, the test accuracy decreased. For example, an unrestricted Decision Tree achieved 99.8% training accuracy but only 74.2% test accuracy, which indicates overfitting. In comparison, a depth-5 Decision Tree achieved 80.1% training accuracy and 79.4% test accuracy, showing a smaller gap between training and test performance.

The main lesson from Week 2 is that accuracy alone is not enough when evaluating a customer churn prediction model. Precision, recall, F1-score, AUC, class imbalance, probability threshold selection, feature importance, overfitting, and business costs must also be considered. The model should therefore be evaluated not only according to its statistical performance but also according to the practical business objective of identifying customers who are likely to churn.





## Week 3: Model Optimization and Unsupervised Learning

### Setup
- Dataset: Telco Customer Churn, $N = 7043$ rows, $d = 30$ features after one-hot encoding
- Split: 80/20 stratified, `random_state=42` ($n_{train} = 5634$, $n_{test} = 1409$), churn rate $\approx 0.265$ in both
- Baseline (always predict "Stay"):

$$\text{Acc}_{base} = \frac{1035}{1409} = 0.735$$

---

### Results

**1. Split-to-split accuracy range across 20 seeds:** [min] to [max]

$$\bar{a} = \frac{1}{20}\sum_{i=1}^{20} a_i, \qquad s = \sqrt{\frac{1}{19}\sum_{i=1}^{20}\left(a_i - \bar{a}\right)^2}$$

(single-seed reference, seed 42: LR $= 0.807$, RF $= 0.807$)

**2. 5-fold CV AUC:** LR [X ± s], RF [X ± s], XGBoost [X ± s]

$$\overline{\text{AUC}}_{CV} = \frac{1}{K}\sum_{k=1}^{K}\text{AUC}_k,\quad K = 5, \qquad s_{CV} = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}\left(\text{AUC}_k - \overline{\text{AUC}}_{CV}\right)^2}$$

(single test split, seed 42: LR $= 0.842$, RF $= 0.842$)

**3. Tuning:** best RF params [..]; grid vs random search time [..]

$$\text{fits}_{grid} = \left(\prod_{j} |V_j|\right) \times K, \qquad \text{fits}_{random} = n_{iter} \times K$$

(current RF: `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`)

**4. Test AUC of final model (used once):** [X]

**5. Customer segments (k = [k]):** [name 1, churn %], [name 2, churn %], ...

$$J = \sum_{i=1}^{N}\left\lVert \mathbf{x}_i - \boldsymbol{\mu}_{c(i)} \right\rVert^2, \qquad \boldsymbol{\mu}_c = \frac{1}{|C_c|}\sum_{i \in C_c}\mathbf{x}_i$$

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}, \qquad \text{churn}_c = \frac{\sum_{i \in C_c} y_i}{|C_c|} \times 100\%$$

where $a(i)$ is the mean distance to points in the same cluster and $b(i)$ the mean distance to the nearest other cluster.

**6. PCA:** [n] of 30 components explain 90% of the variance

$$\Sigma = \frac{1}{N-1}Z^{\top}Z, \qquad \Sigma\,\mathbf{v}_j = \lambda_j \mathbf{v}_j, \qquad \text{EVR}_j = \frac{\lambda_j}{\sum_{k=1}^{30}\lambda_k}$$



**7. Biggest lesson:** [one sentence]

---

### Calculations from my notebook

#### Model comparison (test set, $n = 1409$)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| LR balanced | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (depth 5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | 0.807 | 0.673 | 0.529 | 0.593 | 0.842 |

#### Logistic regression

$$z = b + \sum_{j=1}^{30} w_j x_j, \qquad P(\text{churn}\mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\ln\frac{p}{1-p} = z, \qquad \text{OR}_j = e^{w_j}$$

Check on test row 0 ($z = -3.070$):

$$\sigma(-3.070) = \frac{1}{1 + e^{3.070}} = \frac{1}{1 + 21.54} = 0.0444$$

This matches sklearn's `predict_proba` (0.0444).

Odds ratios (per +1 standard deviation of the scaled feature):

$$\text{OR}_{tenure} = e^{-1.220} = 0.295 \;\Rightarrow\; (1 - 0.295) = 70.5\%\ \text{lower churn odds}$$

$$\text{OR}_{fiber} = e^{0.779} = 2.179, \qquad \text{OR}_{2\text{-year}} = e^{-0.589} = 0.555$$

#### Confusion matrix: TN = 925, FP = 110, FN = 162, TP = 212

$$\text{Accuracy} = \frac{TP + TN}{n} = \frac{212 + 925}{1409} = 0.807$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{212}{322} = 0.658$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{212}{374} = 0.567$$

$$F_1 = \frac{2PR}{P + R} = \frac{2(0.658)(0.567)}{0.658 + 0.567} = 0.609$$

#### ROC / AUC

$$\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}, \qquad \text{AUC} = \int_0^1 \text{TPR}\; d(\text{FPR}) = 0.842$$

#### Business cost and threshold

$$\text{Cost}(t) = FN(t)\cdot C_{FN} + FP(t)\cdot C_{FP}, \qquad C_{FN} = 6000,\; C_{FP} = 1000\ \text{PKR}$$

Flag a customer when the expected cost of ignoring them exceeds the cost of the offer:

$$p \cdot C_{FN} > (1 - p)\cdot C_{FP} \;\Longrightarrow\; p > t^{*} = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1000}{7000} = 0.143$$

The empirical best threshold was $0.15$, close to the theoretical $t^{*}$.

| Threshold | FN | FP | Total cost (PKR) |
|---|---|---|---|
| Never flag anyone | 374 | 0 | $374 \times 6000 = 2{,}244{,}000$ |
| 0.5 | 162 | 110 | $972{,}000 + 110{,}000 = 1{,}082{,}000$ |
| 0.4 | 124 | 191 | $744{,}000 + 191{,}000 = 935{,}000$ |
| 0.3 | 92 | 262 | $552{,}000 + 262{,}000 = 814{,}000$ |
| 0.2 | 54 | 365 | $324{,}000 + 365{,}000 = 689{,}000$ |


#### Decision tree: overfitting

Split criterion (Gini impurity) and information gain:

$$G = 1 - \sum_{k}p_k^{2}, \qquad \Delta G = G_{parent} - \frac{n_L}{n}G_L - \frac{n_R}{n}G_R$$

$$\text{Gap} = \text{Acc}_{train} - \text{Acc}_{test}$$

- Depth 6 (best test accuracy): $\text{Gap} = 0.807 - 0.797 = 0.010$
- Unlimited depth: $\text{Gap} = 0.998 - 0.742 = 0.256$ (overfitting)

#### Random forest

$$\hat{p}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B}\hat{p}_b(\mathbf{x}), \qquad B = 300$$

$$\text{PI}_j = \text{AUC}_{base} - \overline{\text{AUC}}_{\text{permuted } j}$$

- OOB accuracy $= 0.803$, test accuracy $= 0.807$, test AUC $= 0.842$
- Top permutation importance (AUC drop): tenure $0.0404$, TotalCharges $0.0206$, two-year contract $0.0164$

#### Feature engineering

$$\Delta\text{AUC} = 0.8420 - 0.8422 = -0.0002 \;\Rightarrow\; \text{no improvement}$$

#### Class weighting

`class_weight='balanced'` uses

$$w_c = \frac{n}{2\, n_c}$$

$$\Delta R = 0.781 - 0.567 = +0.214, \qquad \Delta P = 0.505 - 0.658 = -0.153, \qquad \Delta F_1 = 0.613 - 0.609 = +0.004$$
**7.

Biggest lesson:** A single test score can mislead: my 80.7% accuracy was only 7.2 points above a baseline that always predicts "Stay" (73.5%), so I now judge models by cross-validated AUC and by business cost, not one accuracy number.
