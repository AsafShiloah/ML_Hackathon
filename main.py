from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from preprocess import *


def main(args):
    train_path = args[0]
    test1_path = args[1]
    test2_path = args[2]

    train = load_data(train_path, parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                               'hotel_live_date'])
    X1_train, y1_train = preprocess_Q1(train)

    X1_test = load_data(test1_path, parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                     'hotel_live_date'])
    id = X1_test['h_booking_id'].reset_index(drop=True)
    X1_train = preprocess_data(X1_train)
    X1_test = preprocess_data(X1_test, flag_train=False)
    X1_train = X1_train.drop(['h_booking_id'], axis=1)
    X1_test = X1_test.drop(['h_booking_id'], axis=1)
    model1 = RandomForestClassifier(n_estimators=100)
    model1.fit(X1_train, y1_train)
    y1_pred = model1.predict(X1_test)

    Q1_prediction = pd.concat([id, pd.Series(y1_pred)], axis=1)
    Q1_prediction.columns = ['id', 'cancellation']
    Q1_prediction.to_csv('agoda_cancellation_prediction_test.csv', index=False)
    # joblib.dump(rf, 'model1.pkl')
    print("finished Q1")
    ###################### Q2 ############################

    train1, train2 = train_test_split(train, test_size=0.5, random_state=42)
    X1_2, y1_2 = preprocess_Q1(train1)
    X2_train, y2_train = preprocess_Q2(train2)

    # Split the data into a training set and a test set
    X1_2 = preprocess_data(X1_2)

    X2_test = load_data(test2_path, parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                     'hotel_live_date'])

    X2_train = preprocess_data(X2_train, flag_train=False)
    X2_test = preprocess_data(X2_test, flag_train=False)

    X1_2 = X1_2.drop(['h_booking_id'], axis=1)

    id = X2_test['h_booking_id'].reset_index(drop=True)
    X2_train = X2_train.drop(['h_booking_id'], axis=1)
    X2_test = X2_test.drop(['h_booking_id'], axis=1)

    # fit and pred model 1_2
    model1_2 = RandomForestClassifier(n_estimators=100)
    model1_2.fit(X1_2, y1_2)
    model1_2_pred_train = model1_2.predict(X2_train)
    model1_2_pred_test = model1_2.predict(X2_test)

    X2_train['cancellation_datetime'] = model1_2_pred_train
    X2_test['cancellation_datetime'] = model1_2_pred_test

    model2 = Lasso(alpha=0.1)
    model2.fit(X2_train, y2_train)
    y_pred = model2.predict(X2_test)
    y_pred = np.where(model1_2_pred_test == 0, -1, y_pred)

    Q2_predictions = pd.concat([id, pd.Series(y_pred)], axis=1)
    Q2_predictions.columns = ['id', 'predicted_selling_amount']
    Q2_predictions.to_csv('agoda_cost_of_cancellation.csv', index=False)
    # joblib.dump(model2, 'model2.pkl')


if __name__ == '__main__':
    main(['agoda_cancellation_train.csv', 'Agoda_Test_1.csv', 'Agoda_Test_2.csv'])
