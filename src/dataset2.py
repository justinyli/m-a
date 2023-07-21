from main import *

def preprocess(data):
    # https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
    # https://datascience.stackexchange.com/questions/27957/why-do-we-need-to-discard-one-dummy-variable/27993#27993
    data = pd.get_dummies(data, columns=['category_code', 'country_code', 'state_code'], drop_first=True)

    # Convert boolean features to binary values (0 and 1)
    data['ipo'] = data['ipo'].astype(int)
    
    # try dropping if company is closed or not
    data['is_closed'] = data['is_closed'].astype(int)
    # data = data.drop(columns=['is_closed'])
    
    data['is_acquired'] = data['is_acquired'].astype(int)
    
    return data

if __name__ == '__main__':
    data_path = 'src\\data\\data2.csv'
    data = pd.read_csv(data_path)
    
    data = data.drop(columns=[
        'products_number',
        'company_id',
        'acquired_companies',
        'mba_degree',
        'phd_degree',
        'ms_degree',
        'other_degree',
    ])
    
    data = preprocess(data)
    
    na_count_per_row = data.isna().sum(axis=1)
    # Total number of rows with NaN values
    total_rows_with_na = len(na_count_per_row[na_count_per_row > 0])
    print("Total rows with NaN values:", total_rows_with_na)
    print(len(data))
    data = data.dropna()
    print(len(data))
    df2 = len(data[data["is_acquired"]==True])
    print('Num Acquired: ' + str(df2))
    
    predictors = data.columns.tolist()
    predictors.remove('is_acquired')
    
    X = data[predictors]
    y = data['is_acquired']
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    model = RandomForestClassifier()

    run_model(model, 'Logistic', X_train, X_test, y_train, y_test, predictors)