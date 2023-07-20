import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def read_excel(path):
    df = pd.DataFrame()
    with pd.ExcelFile(path) as xl:
        for sheet_name in xl.sheet_names:
            sheet_df = xl.parse(sheet_name)
            df = pd.concat([df, sheet_df], ignore_index=True)
    return df

def calculate_metrics(y_true, y_pred):
    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # find confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)

    return accuracy, false_positive_rate, false_negative_rate

merger_path = 'src\\data\\mergers.xlsx'
nonmerger_path = 'src\\data\\nonmergers.xlsx'

mergers = read_excel(merger_path)
nonmergers = read_excel(nonmerger_path)

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
model = LogisticRegression(max_iter=300)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc, fp, fn = calculate_metrics(y_test, y_pred)
print(acc, fp, fn)