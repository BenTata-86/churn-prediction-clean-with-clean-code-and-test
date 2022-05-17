'''
Churn Prediciton Project

author: Taha
date: May 6, 2022
'''
import os
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)

    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''

    plt.figure(figsize=(20, 10))
    #Churn Distribution Plot
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
        )
    churn_plot = sns.histplot(x='Churn', data=data_frame)
    fig = churn_plot.get_figure()
    fig.savefig('./images/eda/churn.png')
    fig.clf()

    # Maritial Status Plot
    maritial_plot = data_frame['Marital_Status'].value_counts('normalize').plot(kind='bar')
    fig = maritial_plot.get_figure()
    fig.savefig('./images/eda/marital_status.png')
    fig.clf()

    # Customer Age Distribution
    age_plot = sns.histplot(x='Customer_Age', data=data_frame)
    fig = age_plot.get_figure()
    fig.savefig('./images/eda/age.png')
    fig.clf()

    #Total Transaction Count Plot
    transaction_count_plot = sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    fig = transaction_count_plot.get_figure()
    fig.savefig('./images/eda/transaction_count.png')
    fig.clf()
    # Correlation Heatmap
    corr_heatmap = sns.heatmap(
        data_frame.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    fig = corr_heatmap.get_figure()
    fig.savefig('./images/eda/correlation.png')
    fig.clf()


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be\
            used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''

    encoder = TargetEncoder()
    encoded_data_frame = encoder.fit_transform(data_frame[category_lst], data_frame[response])
    for col in category_lst:
        data_frame[f'{col}_Churn'] = encoded_data_frame[col]

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be\
              used for naming variables or index y column]

    output:
              Features_train: Features training data
              Features_test: Features testing data
              labels_train: labels training data
              labels_test: labels testing data
    '''

    data_frame[response] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    data_frame_encoded = encoder_helper(data_frame, cat_columns, response)

    features = data_frame_encoded[keep_cols]
    labels = data_frame[response]

    return train_test_split(features, labels, test_size=0.3, random_state=42)


def classification_report_image(labels_train, labels_test, labels_lr, labels_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            labels_train: training response values
            labels_test:  test response values
            labels_train_preds_lr: training predictions from logistic regression
            labels_train_preds_rf: training predictions from random forest
            labels_test_preds_lr: test predictions from logistic regression
            labels_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    clf_report_lr_tr = classification_report(
        labels_train, labels_lr[0], output_dict=True)
    clf_report_lr_test = classification_report(
        labels_test, labels_lr[1], output_dict=True)

    clf_report_rf_tr = classification_report(
        labels_train, labels_rf[0], output_dict=True)
    clf_report_rf_test = classification_report(
        labels_test, labels_rf[1], output_dict=True)

    lr_train = sns.heatmap(pd.DataFrame(
        clf_report_lr_tr).iloc[:-1, :].T, annot=True)
    fig = lr_train.get_figure()
    fig.savefig('images/results/lr_train.png')
    fig.clf()

    lr_test = sns.heatmap(pd.DataFrame(
        clf_report_lr_test).iloc[:-1, :].T, annot=True)
    fig = lr_test.get_figure()
    fig.savefig('images/results/lr_test.png')
    fig.clf()

    rf_train = sns.heatmap(pd.DataFrame(
        clf_report_rf_tr).iloc[:-1, :].T, annot=True)
    fig = rf_train.get_figure()
    fig.savefig('images/results/rf_train.png')
    fig.clf()

    rf_test = sns.heatmap(pd.DataFrame(
        clf_report_rf_test).iloc[:-1, :].T, annot=True)
    fig = rf_test.get_figure()
    fig.savefig('images/results/rf_test.png')
    fig.clf()



def feature_importance_plot(model, features_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features_data.shape[1]), names, rotation=90)
    plt.savefig(f'{output_pth}/feature_importance.png')

def roc_curve_plot(lrc, model, labels_test, features_test):
    '''
    store roc_curve plot: storing roc_curve plots in results folder
    input:
              lrc: logistic regression model
              model: model object containing feature_importances_
              labels_test: labels testing data
              features_test: features testing data
    output:
              None
    '''
    #Roc Curves
    lrc_plot = plot_roc_curve(lrc, features_test, labels_test)
    plt.figure(figsize=(15, 8))
    axes_1 = plt.gca()
    plot_roc_curve(model, features_test, labels_test, ax=axes_1, alpha=0.8)
    lrc_plot.plot(ax=axes_1, alpha=0.8)
    plt.savefig('./images/results/roc_curves.png')
    plt.close()

def train_models(features_train, features_test, labels_train, labels_test):
    '''
    train, store model results: images + scores, and store models
    input:
              features_train: features training data
              features_test: features testing data
              labels_train: labels training data
              labels_test: labels testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(features_train, labels_train)

    lrc.fit(features_train, labels_train)

    labels_train_preds_rf = cv_rfc.best_estimator_.predict(features_train)
    labels_test_preds_rf = cv_rfc.best_estimator_.predict(features_test)

    labels_train_preds_lr = lrc.predict(features_train)
    labels_test_preds_lr = lrc.predict(features_test)

    labels_lr = labels_train_preds_lr, labels_test_preds_lr
    labels_rf = labels_train_preds_rf, labels_test_preds_rf

    model = cv_rfc.best_estimator_
    classification_report_image(labels_train,
                                labels_test,
                                labels_lr,
                                labels_rf)

    feature_importance_plot(model, features_test, output_pth='./images/results')

    roc_curve_plot(lrc, model, labels_test,features_test)
    #Save Models
    joblib.dump(model, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def main():
    '''
    train, store model results: images + scores, and store models
    input:
              None
    output:
              None
    '''
    pth = 'data/bank_data.csv'
    print('Importing data...')
    data_frame = import_data(pth)
    response = 'chunk'

    print('Performing EDA...')
    perform_eda(data_frame)

    print('Performing Feature Engineering...')
    features_train, features_test, labels_train, labels_test = perform_feature_engineering(
        data_frame, response)

    print('Training models...')
    train_models(features_train, features_test, labels_train, labels_test)


if __name__ == '__main__':
    main()
