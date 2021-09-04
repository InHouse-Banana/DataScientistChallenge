import argparse
import itertools
import numpy as np

import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_score\
                            , recall_score, roc_auc_score

def try_func(func, y_data, predictions):
    try:
        return func(y_data,predictions).round(2)
    except Exception as E:
        print(E)
        return np.nan

def calc_metrics(model, x_data, y_data):

    """
    Calculates model metrics: acc, f1, precision, recall, roc_auc, using x_data (features) and y_data (targets).
    It prints the values and returns them into a list.

    Args:
        model ([type]): xgboost model
        x_data ([type]): pandas dataframe array containing features
        y_data ([type]): pandas dataframe array containing targets

    Returns:
        list([acc, f1, precision, recall, roc_auc])
    """

    shape_x, shape_y = (x_data.shape, y_data.shape)
    print('Data shapes X:', shape_x, 'Y:', shape_y)
    predictions = model.predict(x_data) 

    acc  = try_func(accuracy_score, y_data, predictions)
    f1   = try_func(f1_score, y_data, predictions)
    prec = try_func(precision_score, y_data, predictions)
    rec  = try_func(recall_score, y_data, predictions)
    roc  = try_func(roc_auc_score, y_data, predictions)
    
    print('Metrics model -', 'accuracy:', acc, 'f1:', f1, 'precision:', prec, 'recall:', rec, 'roc_auc:', roc) 
    return [acc, f1, prec, rec, roc, shape_x, shape_y]

     

if __name__ == "__main__":

    """
    Reads a csv file, and generates featueres and calculates metrics: acc, f1, precision, recall, roc_auc
    It prints the values and returns them into a list.

    Optional Arguments: 
    --source-file SOURCE_FILE
                        file containing the features and targets to calculate metrics
    --model_file MODEL_FILE
                        model file location that will be tested
    --enc_file ENC_FILE   encoder file locations to one-hot features

    Output:
        Prints [acc, f1, prec, rec, roc_auc, shape_x, shape_y]
    """
    
    # Parsing arguments 
    parser = argparse.ArgumentParser(description='Parameters for XGBoost Prediction')
    parser.add_argument('--source-file', dest = 'source_file', action = 'store', type = str
                        , default='dataset/data.csv'
                        , help='file containing the features and targets to calculate metrics')
    parser.add_argument('--model_file', dest = 'model_file', action = 'store', type = str
                        , default='model_artifacts/xgb_tuned.pkl'
                        , help='model file location that will be tested')
    parser.add_argument('--enc_file', dest = 'enc_file', action = 'store', type = str
                        , default='model_artifacts/encoder.pkl'
                        , help='encoder file location, that one-hot encode features')      
                                                                        
    args        = parser.parse_args()
    source_file = args.source_file
    model_file  = args.model_file
    enc_file    = args.enc_file
                    
    # Load model
    #model_file = "model_artifacts/xgb_tuned.pkl"
    print('Loading model from:', model_file)
    trained_xgb_tuned = pickle.load(open(model_file, "rb"))

    #Load encoder
    #enc_file = "model_artifacts/encoder.pkl"
    print('Loading encoder from:', enc_file)
    enc = pickle.load(open(enc_file, "rb"))

    #source_file = 'data_sources/2016-09-19_79351_training.csv'
    print('Reading file:', source_file)

    cols_to_drop = [ 'PAY_' + str(x) for x in range(0, 7)] + ['ID']
    clean_data   = pd.read_csv(source_file).drop(columns = cols_to_drop)

    SEX_dict       = {1 : 'male', 2 : 'female'}
    EDUCATION_dict = {1 : 'graduate_school', 2 : 'university', 3: 'high_school'\
                    , 4 : 'education_others', 5 : 'unknown', 6 : 'unknown'}
    MARRIAGE_dict  = {1 : 'married', 2 : 'single', 3 : 'marriage_others'}
    #default_payment_next_month_dict = {0 : 'no', 1 : 'yes'}

    clean_data['SEX']       = clean_data['SEX'].replace(SEX_dict)
    clean_data['EDUCATION'] = clean_data['EDUCATION'].replace(EDUCATION_dict)
    clean_data['MARRIAGE']  = clean_data['MARRIAGE'].replace(MARRIAGE_dict)
    #clean_data['default.payment.next.month']  = clean_data['default.payment.next.month'].replace(default_payment_next_month_dict)

    # Drop categories not-used
    clean_data = clean_data[clean_data['EDUCATION'] != 0]
    clean_data = clean_data[clean_data['MARRIAGE'] != 0]

    clean_data = clean_data[clean_data['EDUCATION'] != 'education_others']
    clean_data = clean_data[clean_data['MARRIAGE'] != 'marriage_others']

    # Create BAL_AMT feature
    for month in range(1,7):
        bill_col = 'BILL_AMT' + str(month) 
        pay_col  = 'PAY_AMT' + str(month) 
        bal_col  = 'BAL_AMT' + str(month) 
        clean_data[bal_col] = clean_data[bill_col] - clean_data[pay_col]
        clean_data = clean_data.drop(columns = [bill_col, pay_col])


    training_data = clean_data.copy()
    cat_features  = training_data.select_dtypes(include=["object"]).columns.values
    print('cat_features:', cat_features)

    numerical_features = ['LIMIT_BAL', 'AGE', 'BAL_AMT1'\
                        ,'BAL_AMT2', 'BAL_AMT3', 'BAL_AMT4', 'BAL_AMT5', 'BAL_AMT6']
    print('numerical_features:', numerical_features)

    target = ['default.payment.next.month']
    print('target:', target)


   # Preparing to one-hot-encode
    x = training_data[list(cat_features) + list(numerical_features)]
    y = training_data[target]

    # One-Hot-Encoding
    x_onehot = enc.fit_transform(x[cat_features]).toarray()

    enc_categories = list(itertools.chain(*np.array(enc.categories_, dtype=object)))
    x_onehot       = pd.DataFrame(x_onehot, columns = enc_categories).astype(str)

    # Features and targets
    features_targets         = pd.concat([x_onehot.reset_index(drop=True), x[numerical_features].reset_index(drop=True), y.reset_index(drop=True)]
                                        , axis = 1)
    features_targets.columns = features_targets.columns.str.lower()

    x = features_targets.drop( columns = features_targets.columns[-1:])
    y = features_targets.drop( columns = features_targets.columns[:-1])

    print("Metrics XGB model")
    print('Shape source features:', x.shape, 'targets:', y.shape)
    metrics = calc_metrics(trained_xgb_tuned, x.values, y.values)    


