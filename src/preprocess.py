import pandas as pd
from pandas import DataFrame
from typing import Optional
import pycountry
import numpy as np
import joblib

dummies_lst = ['hotel_country_code', 'charge_option', 'accommadation_type_name', 'origin_country_code',
                                'language', 'original_payment_method', 'original_payment_type',
                                'original_payment_currency', 'guest_nationality_country_name', 'customer_nationality']


def get_country_code(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_2
    except AttributeError:
        print(f"Invalid country name: {country_name}")
        return None


def load_data(path: str, parse_dates=None) -> DataFrame:
    """
    Load data from a CSV file into a DataFrame.
    :param path: string
    :return: DataFrame
    """
    return pd.read_csv(path, parse_dates=parse_dates)


def preproceess_dummies(X: DataFrame, columns: list, flag_train: bool = True) -> DataFrame:
    """
    Create dummies for each column in columns list
    :param flag_train:
    :param X: DataFrame
    :param columns: list
    :return: DataFrame
    """
    for column in columns:
        X = pd.get_dummies(X, columns=[column], prefix=f'dummy_{column}_')
    if not flag_train:
        train_cols = joblib.load('train_cols.pkl')
        X = X.reindex(columns=train_cols, fill_value=0)
    return X


def process_dates(X: DataFrame) -> DataFrame:
    # Calculate trip duration in hours
    X['trip_duration'] = (X['checkout_date'] - X['checkin_date']).dt.total_seconds() / 3600

    # Calculate number of days before booking
    X['days_before_booking'] = (X['checkin_date'] - X['booking_datetime']).dt.total_seconds() / 3600

    # Check if check-in date is a weekend
    X['is_weekend'] = X['checkin_date'].dt.weekday.isin([5, 6])

    # Extract the month from the booking date
    X['month_ordered'] = X['booking_datetime'].dt.month.astype(int)

    # Extract the month from the check-in date
    X['month_checkin'] = X['checkin_date'].dt.month.astype(int)

    # Calculate the period between hotel live date and booking date in hours
    X['hotel_live_period'] = (X['booking_datetime'] - X['hotel_live_date']).dt.total_seconds() / 3600

    return X


def preprocess_drop(X: DataFrame, columns: list) -> DataFrame:
    """
    Drop columns from DataFrame
    :param X: DataFrame
    :param columns: list
    :return: DataFrame
    """
    return X.drop(columns=columns)


def preprocess_data(X: pd.DataFrame, flag_train: bool = True) -> pd.DataFrame:
    """
    Preprocess the data.
    :param df: DataFrame
    :return: DataFrame
    """
    # Drop columns
    X = process_dates(X)


    # Calculate the number of orders per hotel_id
    X['no_orders_of_hotel'] = X.groupby('hotel_id')['hotel_id'].transform('count')

    # Calculate the number of orders per h_customer_id
    X['no_orders_of_customer'] = X.groupby('h_customer_id')['h_customer_id'].transform('count')

    # X['customer_nationality'] = X['customer_nationality'].fillna(get_country_code(X['origin_country_code'])) if X[
    #     'origin_country_code'] else X['origin_country_code'].fillna(X['origin_country_code'].mode()[0])

    # todo: change all countries to countries code

    # convert bool to int:
    X['is_user_logged_in'] = X['is_user_logged_in'].astype(int)
    X['is_first_booking'] = X['is_first_booking'].astype(int)
    X['is_weekend'] = X['is_weekend'].astype(int)

    # fill na with 0 in requests:
    X['request_nonesmoke'] = X['request_nonesmoke'].fillna(0)
    X['request_latecheckin'] = X['request_latecheckin'].fillna(0)
    X['request_highfloor'] = X['request_highfloor'].fillna(0)
    X['request_largebed'] = X['request_largebed'].fillna(0)
    X['request_twinbeds'] = X['request_twinbeds'].fillna(0)
    X['request_airport'] = X['request_airport'].fillna(0)
    X['request_earlycheckin'] = X['request_earlycheckin'].fillna(0)

    # change all rating that are negative to 0
    X['hotel_star_rating'] = X['hotel_star_rating'].apply(lambda x: 0 if x < 0 else x)

    # change all rating that are larger than 5 to 5
    X['hotel_star_rating'] = X['hotel_star_rating'].apply(lambda x: 5 if x > 5 else x)

    """ -------------------------- cancellation policy handle --------------------------"""
    X['cancellation_policy_code'] = X['cancellation_policy_code'].combine(X['trip_duration'],
                                                                        lambda x, y: order_policies(x, y))
    ############################################################
    # reset index of X
    X = X.reset_index(drop=True)

    for i in range(0, 10):
        X[f'{i * 10 + 1}-{(i + 1) * 10}'] = np.zeros(X.shape[0])
    for idx, list_of_lists in enumerate(X['cancellation_policy_code']):
        # Loop through the lists in the current list of lists
        for lst in list_of_lists:
            # Extract the day and percentage
            day, percentage = int(lst[0]), int(lst[1])
            # Determine the appropriate columns for the current percentage
            cols = [f"{i * 10 + 1}-{(i + 1) * 10}" for i in range((percentage - 1) // 10 + 1) if i < 10]
            # Insert the day into the appropriate columns
            for col in cols:
                if X.at[idx, col] == 0:
                    X.at[idx, col] = str(day)

    # Replace the NaNs with an empty string
    X.fillna(0, inplace=True)

    # drop cancellation_policy_code:
    # X = X.drop(['cancellation_policy_code'], axis=1)

    X = X.drop(
        columns=['cancellation_policy_code', 'booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date',
                 'h_customer_id', 'hotel_id'])
    X = preproceess_dummies(X, dummies_lst, flag_train)
    if flag_train:
        train_cols = X.columns
        joblib.dump(train_cols, 'train_cols.pkl')
    return X


def order_policies(policies, duration):
    if policies == 'UNKNOWON':  # TODO: check if needed
        policies = '7D100P'
    policies_arr = []
    if "_" in policies:
        policies_arr = policies.split("_")
    else:
        policies_arr.append(policies)
    if 'D' not in policies_arr[-1]:
        policies_arr = policies_arr[:-1]

    policies_arr = [[policy.split('D')[0], policy.split('D')[1]] for policy in policies_arr]
    for tup in policies_arr:
        if 'N' in tup[1]:
            tup[1] = str(int((int(tup[1][:-1]) / (duration / 24)) * 100))
        else:
            tup[1] = tup[1][:-1]
    return policies_arr


def split_data_label(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    """
    Split the data into X and y.
    :param label:
    :param df: DataFrame
    :return: DataFrame, Series
    """
    X = df.drop([label], axis=1)
    y = df[label]
    return X, y


def preprocess_Q1(X: pd.DataFrame, y: Optional[pd.Series] = None):
    X, y = split_data_label(X, 'cancellation_datetime')
    y = y.fillna(0).apply(lambda x: 1 if x != 0 else 0)
    return X, y


def preprocess_Q2(X: pd.DataFrame, y: Optional[pd.Series] = None):
    X = X.drop(['cancellation_datetime'], axis=1)
    X, y = split_data_label(X, 'original_selling_amount')
    y = y.fillna(0)
    return X, y

if __name__ == "__main__":
    path = "train_data.csv"
    df = load_data(path, ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])
