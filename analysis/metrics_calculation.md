# Metrics Calculation

This document explains how the performance metrics in research paper ( Table 2 and Table 3 ) are calculated based on the output data provided from the model evaluations. We will check if the numbers in the tables match those derived from the output data and point out any discrepancies.

---

## Table 2: Performance on Unbalanced Data

The models evaluated on the unbalanced dataset are:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Random Forest
- BERT (Unbalanced)

### Logistic Regression on Unbalanced Data

**Output Extract:**

```
Training Logistic Regression on unbalanced data...
Accuracy: 0.67
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.95      0.62      0.75        29
   changetype_core       0.62      0.95      0.75       199
   changetype_file       0.67      0.56      0.61        36
    changetype_ftp       0.50      0.05      0.10        19
   changetype_jdbc       0.86      0.25      0.39        24
    changetype_jms       1.00      0.29      0.45        24
   changetype_mail       0.75      0.18      0.29        17
  changetype_redis       1.00      0.45      0.62        22
changetype_tcp_udp       0.82      0.45      0.58        31

          accuracy                           0.67       401
         macro avg       0.80      0.42      0.50       401
      weighted avg       0.72      0.67      0.63       401
```

**Metrics Calculation:**

- **Accuracy:** 0.67 (from `accuracy                           0.67`)
- **Precision:** 0.72 (from `weighted avg       0.72`)
- **Recall:** 0.67 (from `weighted avg           0.67`)
- **F1-Score:** 0.63 (from `weighted avg          0.63`)

**Comparison with Table 2:**

- **Table Values:**

  | Model                | Accuracy | Precision | Recall | F1-Score |
  |----------------------|----------|-----------|--------|----------|
  | Logistic Regression  |   0.67   |    0.72   |  0.67  |   0.63   |

- **Result:** The values match the output.

---

### Linear SVM on Unbalanced Data

**Output Extract:**

```
Training Linear SVM on unbalanced data...
Accuracy: 0.80
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.79      0.79      0.79        29
   changetype_core       0.77      0.93      0.84       199
   changetype_file       0.76      0.69      0.72        36
    changetype_ftp       0.80      0.42      0.55        19
   changetype_jdbc       0.82      0.58      0.68        24
    changetype_jms       0.93      0.54      0.68        24
   changetype_mail       0.89      0.47      0.62        17
  changetype_redis       1.00      0.82      0.90        22
changetype_tcp_udp       0.81      0.81      0.81        31

          accuracy                           0.80       401
         macro avg       0.84      0.67      0.73       401
      weighted avg       0.81      0.80      0.79       401
```

**Metrics Calculation:**

- **Accuracy:** 0.80
- **Precision:** 0.81 (from `weighted avg       0.81`)
- **Recall:** 0.80 (from `weighted avg           0.80`)
- **F1-Score:** 0.79 (from `weighted avg          0.79`)

**Comparison with Table 2:**

- **Table Values:**

  | Model       | Accuracy | Precision | Recall | F1-Score |
  |-------------|----------|-----------|--------|----------|
  | Linear SVM  |   0.80   |    0.81   |  0.80  |   0.79   |

- **Result:** The values match the output.

---

### Multinomial Naive Bayes on Unbalanced Data

**Output Extract:**

```
Training Multinomial NB on unbalanced data...
Accuracy: 0.50
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       1.00      0.03      0.07        29
   changetype_core       0.50      1.00      0.66       199
   changetype_file       0.00      0.00      0.00        36
    changetype_ftp       0.00      0.00      0.00        19
   changetype_jdbc       0.00      0.00      0.00        24
    changetype_jms       0.00      0.00      0.00        24
   changetype_mail       0.00      0.00      0.00        17
  changetype_redis       0.00      0.00      0.00        22
changetype_tcp_udp       0.00      0.00      0.00        31

          accuracy                           0.50       401
         macro avg       0.17      0.11      0.08       401
      weighted avg       0.32      0.50      0.33       401
```

**Metrics Calculation:**

- **Accuracy:** 0.50
- **Precision:** 0.32
- **Recall:** 0.50
- **F1-Score:** 0.33

**Comparison with Table 2:**

- **Table Values:**

  | Model            | Accuracy | Precision | Recall | F1-Score |
  |------------------|----------|-----------|--------|----------|
  | Multinomial NB   |   0.50   |    0.32   |  0.50  |   0.33   |

- **Result:** The values match the output.

---

### Random Forest on Unbalanced Data

**Output Extract:**

```
Training Random Forest on unbalanced data...
Accuracy: 0.76
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.77      0.69      0.73        29
   changetype_core       0.73      0.92      0.82       199
   changetype_file       0.68      0.64      0.66        36
    changetype_ftp       0.75      0.16      0.26        19
   changetype_jdbc       0.77      0.42      0.54        24
    changetype_jms       0.90      0.75      0.82        24
   changetype_mail       0.91      0.59      0.71        17
  changetype_redis       0.94      0.68      0.79        22
changetype_tcp_udp       0.85      0.71      0.77        31

          accuracy                           0.76       401
         macro avg       0.81      0.62      0.68       401
      weighted avg       0.77      0.76      0.74       401
```

**Metrics Calculation:**

- **Accuracy:** 0.76
- **Precision:** 0.77
- **Recall:** 0.76
- **F1-Score:** 0.74

**Comparison with Table 2:**

- **Table Values:**

  | Model          | Accuracy | Precision | Recall | F1-Score |
  |----------------|----------|-----------|--------|----------|
  | Random Forest  |   0.76   |    0.77   |  0.76  |   0.74   |

- **Result:** The values match the output.

---

### BERT (Unbalanced)

**Output Extract (Final Epoch):**

```
Epoch 4/4 for unbalanced data
...
Accuracy on unbalanced data: 0.85
Classification Report:
...
          accuracy                           0.85       401
         macro avg       0.81      0.83      0.82       401
      weighted avg       0.86      0.85      0.85       401
```

**Metrics Calculation:**

- **Accuracy:** 0.85
- **Precision:** 0.86
- **Recall:** 0.85
- **F1-Score:** 0.85

**Comparison with Table 2:**

- **Table Values:**

  | Model             | Accuracy | Precision | Recall | F1-Score |
  |-------------------|----------|-----------|--------|----------|
  | BERT (Unbalanced) |   0.85   |    0.86   |  0.85  |   0.85   |

- **Result:** The values match the output.

---

## Table 3: Performance on Balanced Data

The models evaluated after applying SMOTE and oversampling are:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Random Forest
- BERT (Balanced)

### Logistic Regression on Balanced Data

**Output Extract:**

```
Training Logistic Regression on balanced data...
Accuracy: 0.79
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.68      0.86      0.76        29
   changetype_core       0.84      0.86      0.85       199
   changetype_file       0.76      0.72      0.74        36
    changetype_ftp       0.59      0.53      0.56        19
   changetype_jdbc       0.65      0.54      0.59        24
    changetype_jms       0.89      0.71      0.79        24
   changetype_mail       0.91      0.59      0.71        17
  changetype_redis       0.72      0.82      0.77        22
changetype_tcp_udp       0.77      0.87      0.82        31

          accuracy                           0.79       401
         macro avg       0.76      0.72      0.73       401
      weighted avg       0.79      0.79      0.79       401
```

**Metrics Calculation:**

- **Accuracy:** 0.79
- **Precision:** 0.79
- **Recall:** 0.79
- **F1-Score:** 0.79

**Comparison with Table 3:**

- **Table Values:**

  | Model               | Accuracy | Precision | Recall | F1-Score |
  |---------------------|----------|-----------|--------|----------|
  | Logistic Regression |   0.79   |    0.79   |  0.79  |   0.79   |

- **Result:** The values match the output.

---

### Linear SVM on Balanced Data

**Output Extract:**

```
Training Linear SVM on balanced data...
Accuracy: 0.79
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.69      0.83      0.75        29
   changetype_core       0.83      0.86      0.85       199
   changetype_file       0.71      0.69      0.70        36
    changetype_ftp       0.67      0.53      0.59        19
   changetype_jdbc       0.70      0.67      0.68        24
    changetype_jms       0.90      0.75      0.82        24
   changetype_mail       0.90      0.53      0.67        17
  changetype_redis       0.82      0.82      0.82        22
changetype_tcp_udp       0.75      0.87      0.81        31

          accuracy                           0.79       401
         macro avg       0.77      0.73      0.74       401
      weighted avg       0.80      0.79      0.79       401
```

**Metrics Calculation:**

- **Accuracy:** 0.79
- **Precision:** 0.80
- **Recall:** 0.79
- **F1-Score:** 0.79

**Comparison with Table 3:**

- **Table Values:**

  | Model      | Accuracy | Precision | Recall | F1-Score |
  |------------|----------|-----------|--------|----------|
  | Linear SVM |   0.79   |    0.80   |  0.79  |   0.79   |

- **Result:** The values match the output.

---

### Multinomial Naive Bayes on Balanced Data

**Output Extract:**

```
Training Multinomial NB on balanced data...
Accuracy: 0.72
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.59      0.93      0.72        29
   changetype_core       0.92      0.68      0.78       199
   changetype_file       0.61      0.64      0.62        36
    changetype_ftp       0.50      0.58      0.54        19
   changetype_jdbc       0.58      0.75      0.65        24
    changetype_jms       0.57      0.71      0.63        24
   changetype_mail       0.85      0.65      0.73        17
  changetype_redis       0.55      0.73      0.63        22
changetype_tcp_udp       0.64      0.90      0.75        31

          accuracy                           0.72       401
         macro avg       0.64      0.73      0.67       401
      weighted avg       0.76      0.72      0.72       401
```

**Metrics Calculation:**

- **Accuracy:** 0.72
- **Precision:** 0.76
- **Recall:** 0.72
- **F1-Score:** 0.72

**Comparison with Table 3:**

- **Table Values:**

  | Model           | Accuracy | Precision | Recall | F1-Score |
  |-----------------|----------|-----------|--------|----------|
  | Multinomial NB  |   0.72   |    0.76   |  0.72  |   0.72   |

- **Result:** The values match the output.

---

### Random Forest on Balanced Data

**Output Extract:**

```
Training Random Forest on balanced data...
Accuracy: 0.79
Classification Report:
                      precision    recall  f1-score   support

  changetype_build       0.86      0.66      0.75        29
   changetype_core       0.76      0.93      0.83       199
   changetype_file       0.72      0.72      0.72        36
    changetype_ftp       0.80      0.42      0.55        19
   changetype_jdbc       0.71      0.42      0.53        24
    changetype_jms       0.95      0.75      0.84        24
   changetype_mail       0.92      0.65      0.76        17
  changetype_redis       0.94      0.68      0.79        22
changetype_tcp_udp       0.89      0.77      0.83        31

          accuracy                           0.79       401
         macro avg       0.84      0.67      0.73       401
      weighted avg       0.80      0.79      0.78       401
```

**Metrics Calculation:**

- **Accuracy:** 0.79
- **Precision:** 0.80
- **Recall:** 0.79
- **F1-Score:** 0.78

**Comparison with Table 3:**

- **Table Values:**

  | Model          | Accuracy | Precision | Recall | F1-Score |
  |----------------|----------|-----------|--------|----------|
  | Random Forest  |   0.79   |    0.80   |  0.79  |   0.78   |

- **Result:** The values match the output.

---

### BERT (Balanced)

**Output Extract (Final Epoch):**

```
Epoch 4/4 for balanced data
...
Accuracy on balanced data: 0.88
Classification Report:
...
          accuracy                           0.88       401
         macro avg       0.85      0.85      0.84       401
      weighted avg       0.88      0.88      0.88       401
```

**Metrics Calculation:**

- **Accuracy:** 0.88
- **Precision:** 0.88
- **Recall:** 0.88
- **F1-Score:** 0.88

**Comparison with Table 3:**

- **Table Values:**

  | Model            | Accuracy | Precision | Recall | F1-Score |
  |------------------|----------|-----------|--------|----------|
  | BERT (Balanced)  |   0.88   |    0.88   |  0.88  |   0.88   |

- **Result:** The values match the output.

---



---

**Key Observations:**

- **Best Performance:** BERT consistently outperforms traditional machine learning models on both unbalanced and balanced datasets.
- **Impact of Balancing Data:**
  - Balancing the dataset using SMOTE significantly improves the performance of traditional models.
  - Multinomial Naive Bayes showed the most considerable improvement, increasing accuracy from 50% to 72% after balancing.
- **BERT's Robustness:** BERT maintains high performance even with imbalanced data due to its deep learning architecture and contextual understanding.
- **Computational Trade-offs:** While BERT provides superior accuracy, it requires more computational resources compared to traditional models.

---

**Recommendation:** For applications where computational resources permit, incorporating BERT into the defect assignment process can greatly enhance prediction accuracy. Balancing datasets is crucial for traditional models to perform effectively.