from main import *

if __name__ == '__main__':
    data_path = 'src\\data\\data2.csv'
    data = pd.read_csv(data_path)
    
    data = data.drop(columns=[
        'company_id'
    ])
    
    predictors = data.columns.tolist()
    predictors.remove('is_acquired')
    
    X = data[predictors]
    y = data['is_acquired']
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    model = LogisticRegression(max_iter=300)

    run_model(model, 'Logistic', X_train, X_test, y_train, y_test)