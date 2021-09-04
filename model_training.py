import argparse
import datetime
import itertools
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_score\
                            , recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

import xgboost as xgb

SEED = 42

def model_tuning(xgb_model, XGB_Options, X_train, Y_train):
    """
    Args:
        xgb_model ([type]): xgboost model
        XGB_Options ([type]): options for sklearn GridSearchCV 
        X_train ([type]): features to train on
        Y_train ([type]): targets to train on

    Returns:
        [GSCV]: sklearn gridsearch cv
    """
    start = datetime.datetime.now()
    print('Starting with low learning rate and tuning: max_depth, min_child_weight, n_estimators')

    params = {  
        "learning_rate":     [0.1],
        "max_depth":         XGB_Options['max_depth'], 
        "min_child_weight":  XGB_Options['min_child_weight'], 
        "n_estimators":      XGB_Options['n_estimators'], 

        "colsample_bytree":  [0.8], 
        "subsample":         [0.8],
        "gamma":             [0],
    }

    GSCV = GridSearchCV(xgb_model, 
                        params,
                        cv                 = XGB_Options['cv'],
                        scoring            = XGB_Options['scoring'], 
                        n_jobs             = XGB_Options['n_jobs'], 
                        verbose            = XGB_Options['verbose'], 
                        return_train_score = True)

    GSCV.fit(X_train, Y_train)
    end = datetime.datetime.now()

    print('Time to fit', (end-start))
    print('best_params_:', GSCV.best_params_)#, 
    print('best_score_:',  GSCV.best_score_)

    print('Tuning: gamma')
    start = datetime.datetime.now()
    params = {  
        "learning_rate":    [0.1], 
        "max_depth":        [GSCV.best_params_['max_depth']],
        "min_child_weight": [GSCV.best_params_['min_child_weight']],
        "n_estimators":     [GSCV.best_params_['n_estimators']],

        "colsample_bytree": [0.8], 
        "subsample":        [0.8],
        "gamma":            XGB_Options['gamma'],

    }

    GSCV = GridSearchCV(xgb_model, 
                        params,
                        cv                 = XGB_Options['cv'],
                        scoring            = XGB_Options['scoring'], 
                        n_jobs             = XGB_Options['n_jobs'],
                        verbose            = XGB_Options['verbose'], 
                        return_train_score = True)

    GSCV.fit(X_train, Y_train)
    end = datetime.datetime.now()

    print('Time to fit xgb', (end-start))
    print('best_params_:', GSCV.best_params_)#, 
    print('best_score_:', GSCV.best_score_)


    print('Tuning: colsample_bytree, subsample')
    start = datetime.datetime.now()

    params = {  
        "learning_rate":    [0.1], 
        "max_depth":        [GSCV.best_params_['max_depth']],
        "min_child_weight": [GSCV.best_params_['min_child_weight']],
        "n_estimators":     [GSCV.best_params_['n_estimators']],
        "gamma":            [GSCV.best_params_['gamma']],

        "colsample_bytree": XGB_Options['colsample_bytree'],
        "subsample":        XGB_Options['subsample'],

    }

    GSCV = GridSearchCV(xgb_model, 
                        params,
                        cv                 = XGB_Options['cv'],
                        scoring            = XGB_Options['scoring'], 
                        n_jobs             = XGB_Options['n_jobs'],
                        verbose            = XGB_Options['verbose'], 
                        return_train_score = True)

    GSCV.fit(X_train, Y_train)
    end = datetime.datetime.now()

    print('Time to fit', (end-start))
    print('best_params_:', GSCV.best_params_) 
    print('best_score_:', GSCV.best_score_)

    print('Tuning: reg_alpha, reg_lambda')
    start = datetime.datetime.now()
    params = {  
        "learning_rate":    [0.1], 
        "max_depth":        [GSCV.best_params_['max_depth']],
        "min_child_weight": [GSCV.best_params_['min_child_weight']],
        "n_estimators":     [GSCV.best_params_['n_estimators']],
        "gamma":            [GSCV.best_params_['gamma']],

        "colsample_bytree": [GSCV.best_params_['colsample_bytree']], 
        "subsample":        [GSCV.best_params_['subsample']],


        "reg_alpha":        XGB_Options['reg_alpha'], 
        "reg_lambda":       XGB_Options['reg_lambda'], 
    }

    GSCV = GridSearchCV(xgb_model, 
                        params,
                        cv                 = XGB_Options['cv'],
                        scoring            = XGB_Options['scoring'], 
                        n_jobs             = XGB_Options['n_jobs'],
                        verbose            = XGB_Options['verbose'], 
                        return_train_score = True)

    GSCV.fit(X_train, Y_train)
    end = datetime.datetime.now()

    print('Time to fit', (end-start))
    print('best_params_:', GSCV.best_params_)
    print('best_score_:', GSCV.best_score_)


    print('Tuning: learning_rate')
    start = datetime.datetime.now()

    params = {  
        "learning_rate":    XGB_Options['learning_rate'], 
        "max_depth":        [GSCV.best_params_['max_depth']],
        "min_child_weight": [GSCV.best_params_['min_child_weight']],
        "n_estimators":     [GSCV.best_params_['n_estimators']],
        "gamma":            [GSCV.best_params_['gamma']],

        "colsample_bytree": [GSCV.best_params_['colsample_bytree']], 
        "subsample":        [GSCV.best_params_['subsample']],


        "reg_alpha":        [GSCV.best_params_['reg_alpha']],
        "reg_lambda":       [GSCV.best_params_['reg_lambda']]
    }

    GSCV = GridSearchCV(xgb_model, 
                        params,
                        cv                 = XGB_Options['cv'],
                        scoring            = XGB_Options['scoring'], 
                        n_jobs             = XGB_Options['n_jobs'],
                        verbose            = XGB_Options['verbose'], 
                        return_train_score = True)

    GSCV.fit(X_train, Y_train)
    end = datetime.datetime.now()

    print('Time to fit', (end-start))
    print('best_params_:', GSCV.best_params_)
    print('best_score_:', GSCV.best_score_)

    return GSCV


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
        mode ([type]): xgboost model
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
    Performs reads data from a determined source-file and generates features for an XGBoost model
    
    Optional Arguments:
        --source-file SOURCE_FILE
                                file containing the features and targets to calculate metrics
        --model_file MODEL_FILE
                                where the model will be saved model
        --enc_file ENC_FILE   where the encoder encoder will be saved
        --n_jobs N_JOBS       number of cores for the model to use

    Outputs:
        training_data_file      csv file that contains the features and the respective targets
        features_targets_file   csv file that contains the one-hot encoded features and the respective targets
        combinations_cats       csv file that contains the category combinations counts
        enc                     encoder that performs one-hot-encoding with handle_unknown='ignore'
        model                   tuned xgboost model

        Prints [acc, f1, prec, rec, roc_auc] for train and test datasets

    """

    parser = argparse.ArgumentParser(description='Parameters for XGBoost Prediction')
    parser.add_argument('--source-file', dest = 'source_file', action = 'store', type = str
                        , default='dataset/data.csv'
                        , help='file containing the features and targets to calculate metrics')
    parser.add_argument('--model_file', dest = 'model_file', action = 'store', type = str
                        , default='model_artifacts/xgb_tuned.pkl'
                        , help='where the model will be saved model')
    parser.add_argument('--enc_file', dest = 'enc_file', action = 'store', type = str
                        , default='model_artifacts/encoder.pkl'
                        , help='where the encoder encoder will be saved')      
    parser.add_argument('--n_jobs', dest = 'n_jobs', action = 'store', type = int
                        , default= cpu_count()//2
                        , help='number of cores for the model to use')

    args        = parser.parse_args()
    source_file = args.source_file
    model_file  = args.model_file
    enc_file    = args.enc_file
    n_jobs      = args.n_jobs

    dir_data_artifacts = 'data_artifacts'
    if not os.path.exists(dir_data_artifacts):
        os.mkdir(dir_data_artifacts)

    dir_model_artifacts = 'model_artifacts'
    if not os.path.exists(dir_model_artifacts):
        os.mkdir(dir_model_artifacts)

    # Read input data
    source_data  = pd.read_csv(source_file)

    cols_to_drop = [ 'PAY_' + str(x) for x in range(0, 7)] + ['ID']
    clean_data   = source_data.copy().drop(columns = cols_to_drop)

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

    training_data_file = os.path.join(dir_data_artifacts,'training_data.csv')
    print('Saving training_data_file to:', training_data_file)

    training_data = clean_data.copy()
    features          = training_data.select_dtypes(include=["object"]).columns
    combinations_cats = training_data[features].value_counts().to_frame('counts').reset_index()\
                            .sort_values(by='counts', ascending = False)

    combinations_cats_file = os.path.join(dir_data_artifacts,'combination_categories.csv')    
    print('Saving combinations_cats_file to:', combinations_cats_file)
    combinations_cats.to_csv(combinations_cats_file, index=False)

    cat_features       = training_data.select_dtypes(include=["object"]).columns.values
    print('cat_features:', cat_features)

    numerical_features = ['LIMIT_BAL', 'AGE', 'BAL_AMT1'\
                        ,'BAL_AMT2', 'BAL_AMT3', 'BAL_AMT4', 'BAL_AMT5', 'BAL_AMT6']
    print('numerical_features:', numerical_features)

    target             = ['default.payment.next.month']
    print('target:', target)

    # Preparing to one-hot-encode
    x = training_data[list(cat_features) + list(numerical_features)]
    y = training_data[target]

    # One-Hot-Encoding
    enc      = OneHotEncoder(handle_unknown='ignore')
    x_onehot = enc.fit_transform(x[cat_features]).toarray()

    enc_categories = list(itertools.chain(*np.array(enc.categories_, dtype=object)))
    x_onehot       = pd.DataFrame(x_onehot, columns = enc_categories).astype(str)

    # Features and targets
    features_targets         = pd.concat([x_onehot.reset_index(drop=True), x[numerical_features].reset_index(drop=True), y.reset_index(drop=True)]
                                        , axis = 1)
    features_targets.columns = features_targets.columns.str.lower()

    features_targets_file = os.path.join(dir_data_artifacts,'training_features_targets.csv')
    print('Saving features_targets_file to:', features_targets_file)
    features_targets.to_csv(features_targets_file, index=False)


    # save the encoder
    print('Saving enc_file to:', enc_file)
    with open(enc_file, 'wb') as f: 
        pickle.dump(enc, f)


    # Split train/tst
    x = features_targets.drop( columns = features_targets.columns[-1:])
    y = features_targets.drop( columns = features_targets.columns[:-1])


    (X_train, X_test, Y_train, Y_test) = train_test_split(x, y\
                                                        , test_size = 0.2, random_state= SEED\
                                                        , stratify  = y)



    # Separate majority and minority classes
    index_majority = Y_train[Y_train['default.payment.next.month'] == 0].index.values
    index_minority = Y_train[Y_train['default.payment.next.month'] == 1].index.values

    df_majority = X_train.loc[index_majority,:]
    df_minority = X_train.loc[index_minority,:]

    print('df_majority.shape', df_majority.shape, 'df_minority.shape', df_minority.shape)

    # Upsample minority class
    df_minority_upsampled = resample(df_minority 
                                    , replace      = True     # sample with replacement
                                    , n_samples    = df_majority.shape[0]    # to match majority class
                                    , random_state = SEED) # reproducible results

    print('df_majority.shape:', df_majority.shape, 'df_minority_upsampled.shape:', df_minority_upsampled.shape)

    # Combine majority class with upsampled minority class
    X_train_upsampled = pd.concat([df_majority, df_minority_upsampled])
    Y_train_upsampled = Y_train.loc[X_train_upsampled.index,:]

    X_train_upsampled = X_train_upsampled.reset_index(drop=True)
    Y_train_upsampled = Y_train_upsampled.reset_index(drop=True)
    
    # Display new class counts
    print('X_train_upsampled.shape:', X_train_upsampled.shape)
    print('Y_train_upsampled.shape:', Y_train_upsampled.shape)
    #training_data_upsampled['default.payment.next.month'].value_counts()

    #Model Training
    XGB_Options = { 
                    'n_jobs':           1, #cpu_count()//2,
                    'cv':               3,
                    'scoring':          'roc_auc',#'roc_auc',
                    'seed':             SEED, 
        
                    'max_depth':        np.arange(2,8,4),
                    'min_child_weight': np.arange(1,10,3),
                    'n_estimators':     [50, 100, 200], 

                    'gamma':            np.arange(0.05,0.45,0.15),
                    'colsample_bytree': np.arange(0.60, 0.95, 0.15),
                    'subsample':        np.arange(0.60, 0.95, 0.15),
                    'reg_alpha':        [1e-6, 1, 10], 
                    'reg_lambda':       [1e-6, 1, 10],
                    'learning_rate':    np.arange(0.025,0.150,0.050),
                    'verbose':         1                   
                }

    # Model Tuning
    eval_metric = ['logloss', 'auc'] 
    eval_set    = [(X_train_upsampled.values, Y_train_upsampled.values), (X_test.values, Y_test.values)]
    xgb_model   = xgb.XGBClassifier(random_state = XGB_Options['seed']\
                                    , n_jobs = cpu_count()//2\
                                    , eval_metric = eval_metric, use_label_encoder=False) 
    GSCV        = model_tuning(xgb_model, XGB_Options, X_train_upsampled.values, Y_train_upsampled.values)

    # Fitting the best set of parameters
    start = datetime.datetime.now()
    xgb_tuned           = xgb.XGBClassifier(random_state=XGB_Options['seed']\
                                            , n_jobs=cpu_count()//2\
                                            , use_label_encoder=False) #seed)
    xgb_tuned.set_params(**GSCV.best_params_)
    trained_xgb_tuned   = xgb_tuned.fit(X_train_upsampled.values, Y_train_upsampled.values\
                                        , eval_metric = eval_metric \
                                        , eval_set = eval_set, verbose=False)
    feature_importances = trained_xgb_tuned.feature_importances_ 
    end = datetime.datetime.now()
    print('Time to fit xgb', (end-start))

    # Save model
    print('Saving model_file to:', model_file)
    pickle.dump(trained_xgb_tuned, open(model_file, "wb"))
    with open(model_file, 'wb') as f: 
        pickle.dump(trained_xgb_tuned, f)

    #Final metrics
    print("Metrics XGB model tuned on train data")
    metrics_tuned_model_train = calc_metrics(trained_xgb_tuned, X_train_upsampled.values, Y_train_upsampled.values)

    print("Metrics XGB model tuned on test data")
    metrics_tuned_model_test =  calc_metrics(trained_xgb_tuned, X_test.values, Y_test.values)
    
