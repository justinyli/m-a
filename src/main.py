import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_excel(path):
    df = pd.DataFrame()
    with pd.ExcelFile(path) as xl:
        for sheet_name in xl.sheet_names:
            sheet_df = xl.parse(sheet_name)
            df = pd.concat([df, sheet_df], ignore_index=True)
    return df

def run_model(model, name, X_train, X_test, y_train, y_test):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
    # this exists for basically every regression
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc, fp, fn = calculate_metrics(y_test, y_pred)
    print(f"Model: {name}")
    print("Accuracy:", acc)
    print("False Positive Rate:", fp)
    print("False Negative Rate:", fn)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
    y_prob = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_prob, name)
    
    plot_confusion_matrix(y_test, y_pred, name)
    
    if hasattr(model, 'coef_'):
        # https://scikit-learn.org/stable/glossary.html#term-coef_
        coefficients = model.coef_[0]
    else:
        # models without coef_
        # https://scikit-learn.org/stable/glossary.html#term-feature_importances_
        coefficients = model.feature_importances_
    plot_feature_importance(coefficients, predictors, name)
    
    return None

def calculate_metrics(y_true, y_pred):
    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # find confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)

    return accuracy, false_positive_rate, false_negative_rate

def plot_roc_curve(y_true, y_prob, name):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return None

def plot_confusion_matrix(y_true, y_pred, name):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn-metrics-confusion-matrix
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    return None

def plot_feature_importance(coefficients, predictors, name):
    coef_df = pd.DataFrame({'Variable': predictors, 'Coefficient': coefficients})
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Variable', data=coef_df, palette='coolwarm')
    plt.title(f'Feature Importance - {name}')
    plt.xlabel('Coefficient')
    plt.ylabel('Variable')
    plt.show()
    
    return None


if __name__ == '__main__':
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # model = LogisticRegression(max_iter=300)

    # run_model(model, 'Logistic', X_train, X_test, y_train, y_test)


    # Number of clusters to create
    num_clusters = 2

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=None)
    data['Cluster'] = kmeans.fit_predict(X)

    # Train logistic regression model for each cluster
    for cluster_id in range(num_clusters):
        cluster_data = data[data['Cluster'] == cluster_id].drop(columns='Cluster')
        X_train, X_test, y_train, y_test = train_test_split(cluster_data[predictors], cluster_data['Label'], test_size=0.2, random_state=None)

        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate metrics for each cluster
        accuracy, false_positive_rate, false_negative_rate = calculate_metrics(y_test, y_pred)

        print(f"Cluster {cluster_id}:")
        print("Accuracy:", accuracy)
        print("False Positive Rate:", false_positive_rate)
        print("False Negative Rate:", false_negative_rate)
        plot_roc_curve(y_test, y_pred, f'Cluster {cluster_id}:')