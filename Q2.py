import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from preprocess import preprocess_data, load_data, preprocess_Q1, preprocess_Q2
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt


def random_forrest(X_train, X_test, y_train, y_test=None):
    rf = RandomForestClassifier(n_estimators=100)

    # Fit the model to the data
    rf.fit(X_train, y_train)

    # Predict the values for the test set
    y_pred = rf.predict(X_test)

    return y_pred, rf


def lasso_regression(X_train, X_test, y_train, y_test):
    import pandas as pd

    # Assuming X_train, y_train, X_test, y_test are already defined

    # Initialize the model
    model = Lasso(alpha=0.1)  # the alpha parameter controls the degree of regularization

    # Fit the model
    model.fit(X_train, y_train)

    # Predict the selling amount for the test set
    predictions = model.predict(X_test)

    # Calculate the RMSE
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse}")
    return predictions, model


def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                              'hotel_live_date'])
    # break data into two parts
    data1, data2 = train_test_split(data, test_size=0.5, random_state=42)
    X1, y1 = preprocess_Q1(data1)
    X2, y2 = preprocess_Q2(data2)

    # Split the data into a training set and a test set
    X1 = preprocess_data(X1)
    X2 = preprocess_data(X2, flag_train=False)

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    X1 = X1.drop(['h_booking_id'], axis=1)
    X2 = X2.drop(['h_booking_id'], axis=1)
    """return Q1"""
    rf_pred, rf = random_forrest(X1, X2, y1)
    #
    print(pd.Series(rf_pred).shape, X2.shape)
    X2 = pd.concat([X2, pd.Series(rf_pred)], axis=1)
    print(pd.Series(rf_pred).shape, X2.shape)

    id = X2_test['h_booking_id'].reset_index(drop=True)
    X2_train = X2_train.drop(['h_booking_id'], axis=1)
    X2_test = X2_test.drop(['h_booking_id'], axis=1)
    y_pred, lasso = lasso_regression(X2_train, X2_test, y2_train, y2_test)

    result = pd.concat([id, pd.Series(y_pred)], axis=1)
    result.columns = [id, 'predicted_selling_amount']
    result.to_csv('agoda_cost_of_cancellation.csv', index=False)
    joblib.dump(lasso, 'model2.pkl')


if __name__ == '__main__':
    np.random.seed(0)
    main()