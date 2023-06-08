import pandas as pd
from pandas import DataFrame
import sklearn as sk
from typing import Optional, NoReturn
import pycountry
import numpy as np


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


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    Preprocess the data.
    :param df: DataFrame
    :return: DataFrame
    """

    X = X.drop(['h_booking_id'], axis=1)

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
    X['hotel_live_period'] = (X['hotel_live_date'] - X['booking_datetime']).dt.total_seconds() / 3600

    """ -------------------------- preprocess dates END --------------------------"""

    # Calculate the number of orders per hotel_id
    X['no_orders_of_hotel'] = X.groupby('hotel_id')['hotel_id'].transform('count')

    # Calculate the number of orders per h_customer_id
    X['no_orders_of_customer'] = X.groupby('h_customer_id')['h_customer_id'].transform('count')

    """ -------------------------- count stuff END --------------------------"""

    X['customer_nationality'] = X['customer_nationality'].fillna(get_country_code(X['country_name'])) if X[
        'country_name'] else X['country_name'].fillna(X['country_name'].mode()[0])

    # todo: change all countries to countries code

    # fill na in origin country with
    X['origin_country_code'] = X['origin_country_code'].fillna(get_country_code(X['country_name'])) if X[
        'country_name'] else X['country_name'].fillna('KR')

    """ -------------------------- dummies values start --------------------------"""

    # create dummies for hotel_country_code
    X = pd.get_dummies(X, columns=['hotel_country_code'])

    # create dummies for charge_option
    X = pd.get_dummies(X, columns=['charge_option'])

    # create dummies for accommadation_type_name
    X = pd.get_dummies(X, columns=['accommadation_type_name'])

    # create dummies for origin country:
    X = pd.get_dummies(X, columns=['origin_country_code'], prefix='code_')  # TODO: check if need GLOBAL

    # create dummies for language:
    X = pd.get_dummies(X, columns=['language'], prefix='lang_')  # TODO: check if need GLOBAL

    # TODO:replace UNKNOWN in original payment method:
    # X['original_payment_method'] = X['original_payment_method'].replace('UNKNOWN', 'OTHER')

    # create dummies for payment method:
    X = pd.get_dummies(X, columns=['original_payment_method'], prefix='pay_method_')  # TODO: check if need GLOBAL

    # create dummies for original payment type:
    X = pd.get_dummies(X, columns=['original_payment_type'], prefix='pay_type_')  # TODO: check if need GLOBAL

    # create dummies for original payment currency:
    X = pd.get_dummies(X, columns=['original_payment_currency'],
                       prefix='pay_currency_')  # TODO: check if need GLOBAL, cor with country code

    # convert bool to int:
    X['is_user_logged_in'] = X['is_user_logged_in'].astype(int)
    X['is_first_booking'] = X['is_first_booking'].astype(int)

    # fill na with 0 in requests:
    X['request_nonesmoke'] = X['request_nonesmoke'].fillna(0)
    X['request_latecheckin'] = X['request_latecheckin'].fillna(0)
    X['request_highfloor'] = X['request_highfloor'].fillna(0)
    X['request_largebed'] = X['request_largebed'].fillna(0)
    X['request_twinbeds'] = X['request_twinbeds'].fillna(0)
    X['request_airport'] = X['request_airport'].fillna(0)
    X['request_earlycheckin'] = X['request_earlycheckin'].fillna(0)

    """ -------------------------- cancellation policy handle --------------------------"""
    X['cancellation_policy_code'] = X['cancellation_policy_code'].combine(X['trip_duration'],
                                                                            lambda x, y: order_policies(x, y))

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

    #drop cancellation_policy_code:
    X = X.drop['cancellation_policy_code']

def order_policies(policies, duration):
    # policies = row['cancellation_policy_code']
    # print(policies)
    # duration = row['trip_duration']
    if policies == 'UNKNOWON': # TODO: check if needed
        policies = '7D100P'
    policies_arr = []
    if "_" in policies:
        policies_arr = policies.split("_")
    else:
        policies_arr.append(policies)
    # print(policies,policies_arr)
    if 'D' not in policies_arr[-1]:
        # no show
        policies_arr = policies_arr[:-1]

    policies_arr = [[policy.split('D')[0], policy.split('D')[1]] for policy in policies_arr]
    for tup in policies_arr:
        if 'N' in tup[1]:
            tup[1] = str(int((int(tup[1][:-1])/(duration/24))*100))
        else:
            tup[1] = tup[1][:-1]
    return policies_arr









if __name__ == "__main__":
    path = "train_data.csv"
    df = load_data(path, ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])
    # for column_name in df.columns:
    #     print(df[column_name].describe())
    #     print("")
    #     print("")
    #

    # print(df['request_nonesmoke'].dropna().describe())
    # print('no_of_extra_bed:',df['no_of_extra_bed'].unique())
    # print('no_of_room',df['no_of_room'].unique())
    # # print('origin_country_code:', df['origin_country_code'].unique())
    # # print('origin_country_code:', df['origin_country_code'].dropna().unique())
    # print('origin_country_code:', df['origin_country_code'].describe())
    # TODO split no show via _ and then via 'D'
        # todo: handle UNKNOWN
    # print(df['origin_country_code'].mode()[0])
    # df['cancellation_policy_code'] = df['cancellation_policy_code'].str.split('_') if '_' in df['cancellation_policy_code'] else df['cancellation_policy_code']
    # print(df['cancellation_policy_code'].unique())
    # df['cancellation_policy_code']= [(policy.split('D')[0], policy.split('D')[1].rstrip('P')) for policy in df['cancellation_policy_code']]
    # print(df['cancellation_policy_code'])
    # print(df['cancellation_policy_code'].apply(lambda x: x.split('_') if '_' in x else x))
    # print(df['cancellation_policy_code'].apply(order_policies(df['cancellation_policy_code'])))
    # print(order_policies("7D100P_100P", 1))
    # df['trip_duration'] = ((df['checkout_date'] - df['checkin_date']).dt.total_seconds() / 3600)
    # # print(df['cancellation_policy_code'].apply(order_policies))
    #
    # df['cancellation_policy_code'] = df['cancellation_policy_code'].combine(df['trip_duration'], lambda x, y: order_policies(x, y))
    # # print(df['cancellation_policy_code'])
    # for i in range(0,10):
    #     df[f'{i * 10 + 1}-{(i + 1) * 10}'] = np.zeros(df.shape[0])

    # for idx, list_of_lists in enumerate(df['cancellation_policy_code']):
    #     # Loop through the lists in the current list of lists
    #     for lst in list_of_lists:
    #         # Extract the day and percentage
    #         day, percentage = int(lst[0]), int(lst[1])
    #         # Determine the appropriate columns for the current percentage
    #         cols = [f"{i * 10 + 1}-{(i + 1) * 10}" for i in range((percentage - 1) // 10 + 1) if i < 10]
    #         # Insert the day into the appropriate columns
    #
    #         for col in cols:
    #
    #             if df.at[idx, col] == 0:
    #                 df.at[idx, col] = str(day)
    #
    # # Replace the NaNs with an empty string
    # df.fillna('', inplace=True)
    # print(df[['cancellation_policy_code','1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90','91-100']])


