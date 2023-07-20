import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


merger_path = 'src\\data\\mergers.xlsx'

mergers = pd.DataFrame()
with pd.ExcelFile(merger_path) as xls:
    for sheet_name in xls.sheet_names:
        sheet_df = xls.parse(sheet_name)
        mergers = pd.concat([mergers, sheet_df], ignore_index=True)

nonmerger_path = 'src\\data\\nonmergers.xlsx'

nonmergers = pd.DataFrame()
with pd.ExcelFile(nonmerger_path) as xls:
    for sheet_name in xls.sheet_names:
        sheet_df2 = xls.parse(sheet_name)
        nonmergers = pd.concat([nonmergers, sheet_df2], ignore_index=True)

data = pd.concat([mergers, nonmergers], ignore_index=True)


data = data.drop(columns=[
    'AD-3',
    'AD-2',
    'AD-1',
    'AD-30',
    'Announcement Date',
    'Company RIC'
])
data = data.dropna()


predictors = data.columns.tolist()
predictors.remove('Label')

X = data[predictors]
y = data['Label']

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()