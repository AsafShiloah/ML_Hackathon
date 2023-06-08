from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import *
def main(args):
    train_path = args[1]

    train = load_data(train_path, parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                                   'hotel_live_date'])
    #### Q1 model fit ####
    X1_train, y1_train = preprocess_Q1(train)
    X1_train = preprocess_data(X1_train)
    X1_train = X1_train.drop(['h_booking_id'], axis=1)

    model1 = RandomForestClassifier(n_estimators=100)
    model1.fit(X1_train, y1_train)

    #### Q2 model fit ####
    train1, train2 = train_test_split(train, test_size=0.5, random_state=42)
    X1_2, y1_2 = preprocess_Q1(train1)
    X2_train, y2_train = preprocess_Q2(train2)

    X_test = load_data('Agoda_Test_1.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                        'hotel_live_date'])

    id = X_test['h_booking_id'].reset_index(drop=True)
    X_test = preprocess_data(X_test, flag_train=False)
    X_test = X_test.drop(['h_booking_id'], axis=1)
    """return Q1"""
    y_pred = model1.predict(X_test)

    # f1 = f1_score(y_test, y_pred, average='macro')
    # print('F1 Score Our Test: ', f1)

    result = pd.concat([id, pd.Series(y_pred)], axis=1)
    result.columns = ['id', 'cancellation']
    result.to_csv('agoda_cancellation_prediction.csv', index=False)
    joblib.dump(model1, 'model1.pkl')

######################################################################33

    # break data into two parts


    X1 = preprocess_data(X1)

    X2_test = load_data('Agoda_Test_2.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                     'hotel_live_date'])

    X2_train = preprocess_data(X2_train, flag_train=False)
    X2_test = preprocess_data(X2_test, flag_train=False)

    X1 = X1.drop(['h_booking_id'], axis=1)

    id = X2_test['h_booking_id'].reset_index(drop=True)
    X2_train = X2_train.drop(['h_booking_id'], axis=1)
    X2_test = X2_test.drop(['h_booking_id'], axis=1)

    """return Q1"""
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X1, y1)
    rf_pred_train = rf.predict(X2_train)
    rf_pred_test = rf.predict(X2_test)

    X2_train['cancellation_datetime'] = rf_pred_train
    X2_test['cancellation_datetime'] = rf_pred_test

    # y2_test = y2_test.reset_index(drop=True)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X2_train, y2_train)
    y_pred = lasso.predict(X2_test)

    # change y_pred to -1 where r_pred is 0
    y_pred = np.where(rf_pred_test == 0, -1, y_pred)

    result = pd.concat([id, pd.Series(y_pred)], axis=1)
    result.columns = ['id', 'predicted_selling_amount']
    result.to_csv('agoda_cost_of_cancellation.csv', index=False)
    joblib.dump(lasso, 'model2.pkl')

if __name__ == '__main__':
    main()