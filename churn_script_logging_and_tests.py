'''
Predict Churn Script Project- testing
Author Taha
Date : 20/05/2022
'''


import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info("Checking imported data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err



def test_eda(perform_eda, dataframe):
    '''
    test perform eda function
    '''
    try:
        perform_eda(data_frame=dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error("Testing perform eda: Error occured while performing eda ")
        raise err
    #checking save_fig
    try:
        assert os.path.isfile("./images/eda/marital_status.png")
        logging.info("Testing saved file: marital_status.png was created")
    except AssertionError as err:
        logging.error("Not such file on disk: marital_status.png")
        raise err
    try:
        assert os.path.isfile("./images/eda/churn.png")
        logging.info("Testing saved file: churn.png was created")
    except AssertionError as err:
        logging.error("Not such file on disk: churn.png")
        raise err
    try:
        assert os.path.isfile("./images/eda/transaction_count.png")
        logging.info("Testing saved file: transaction_count.png was created")
    except AssertionError as err:
        logging.error("Not such file on disk: transaction_count.png")
        raise err
    try:
        assert os.path.isfile("./images/eda/correlation.png")
        logging.info("Testing saved file: correlation.png was created")
    except AssertionError as err:
        logging.error("Not such file on disk: correlation.png")
        raise err

    try:
        assert os.path.isfile("./images/eda/age.png")
        logging.info("Testing saved file: age.png was created")
    except AssertionError as err:
        logging.error("Not such file on disk: age.png")
        raise err


def test_encoder_helper(encoder_helper, dataframe):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    dataframe['churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        encoded_dataframe = encoder_helper(
            data_frame=dataframe,
            category_lst= cat_columns,
            response='churn'
        )
        logging.info("Testing encoder helper: SUCCESS")
    except KeyError as err:
        logging.error("Testing encoder: Error occured while performing encoder helper")
        raise err

    expected_columns = [f'{col}_Churn' for col in cat_columns ]
    columns_found = set(encoded_dataframe.columns).intersection(set(expected_columns))
    try:
        assert len(expected_columns)==len(columns_found)
        logging.info("Testing encoded columns: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder: Wrong encoded dataframe")
        raise err




def test_perform_feature_engineering(perform_feature_engineering, dataframe):
    '''
    test perform_feature_engineering
    '''
    try:
        labels_train, labels_test, features_train, features_test = perform_feature_engineering(
            data_frame = dataframe,
            response= 'churn'
        )
        logging.info("Testing perform feature engineering: SUCCESS")
    except KeyError as err:
        logging.error("Testing Feature engineering: error occured\
             while performing feature engineering")
        raise err
    try:
        assert labels_train.shape[0] > 0
        assert labels_train.shape[1] > 0
        assert labels_test.shape[0] > 0
        assert labels_test.shape[1] > 0
        assert features_train.shape[0] > 0
        assert features_test.shape[0] > 0
        logging.info("Checking encoded dataframe: SUCCESS")
    except AssertionError as err:
        logging.error("The labels and features doesn't appear to have \
            columns or rows ")
        raise err


def test_train_models(train_models, dataframe):
    '''
    test train_models
    '''
    labels_train, labels_test, features_train, features_test = cl.perform_feature_engineering(
        dataframe, 'churn')

    try:
        train_models(labels_train, labels_test, features_train, features_test)
        logging.info("Performing training models: SUCCESS")
    except KeyError as err:
        logging.error("f")
        raise err
    try:
        assert os.path.isfile("./models/logistic_model.pkl")
        logging.info("Testing saved file: logistic model is created")
    except AssertionError as err:
        logging.error("Not such file on disk: logistic_model.pkl")
        raise err
    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info("Testing saved file: rfc model is created")
    except AssertionError as err:
        logging.error("Not such file on disk: rfc_model.pkl")
        raise err

if __name__ == "__main__":

    RESPONSE = 'churn'
    DATAFRAME = cl.import_data("./data/bank_data.csv")
    test_import(import_data=cl.import_data)
    test_eda(perform_eda=cl.perform_eda, dataframe=DATAFRAME)
    test_encoder_helper(encoder_helper=cl.encoder_helper, dataframe=DATAFRAME)
    test_perform_feature_engineering(
        perform_feature_engineering=cl.perform_feature_engineering,
        dataframe=DATAFRAME)
    test_train_models(cl.train_models, dataframe=DATAFRAME)
