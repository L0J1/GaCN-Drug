import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. Read and Prepare Data
df = pd.read_csv('D:/1234.txt', sep='\t', header=None, names=['PC1', 'PC2', 'lab'])

# Map labels to 0 to N-1
unique_labels = sorted(df['lab'].unique())
label_map = {label: i for i, label in enumerate(unique_labels)}
df['lab_mapped'] = df['lab'].map(label_map)

# 2. Feature Scaling
scaler = StandardScaler()
X = df[['PC1', 'PC2']]
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split
y = df['lab_mapped']
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=1, stratify=y
)

# 4. Model Training
model = OneVsRestClassifier(SVC(kernel='rbf', gamma='scale', C=1000, probability=True))
model.fit(X_train, y_train)

# 5. Prediction Probabilities
y_score = model.predict_proba(X_test)

# 6. Label Binarization
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# 7. Compute ROC Curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 8. Save Micro-Average ROC Curve Data
with open('roc_data_micro_average.txt', 'w') as file:
    file.write(f'FPR\tTPR\tAUC={roc_auc_micro}\n')
    for fp, tp in zip(fpr_micro, tpr_micro):
        file.write(f'{fp}\t{tp}\n')

# 9. Plot ROC Curves
plt.figure()
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot individual class ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()