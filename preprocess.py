from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.express as px
import missingno as msno


def load_data(path, parse_dates=None):
    return pd.read_csv(path, parse_dates=parse_dates)


def split_train_test(data, test_ratio):
    return train_test_split(data, test_size=test_ratio, random_state=42)


def save_to_csv(data, path):
    data.to_csv(path)


def main():
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_cols', None)

    # Convert specific columns to DateTime type for date-related operations
    data = load_data("train_data.csv", ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date'])
    # todo: drop dates

    null = 100 * data.isnull().sum() / data.shape[0]
    # print(null)
    msno.matrix(data)
    msno.bar(data)
    msno.heatmap(data)


    # drop h_booking_id
    data = data.drop(['h_booking_id'], axis=1)

    """ -------------------------- preprocess dates START --------------------------"""

    # Calculate trip duration in hours
    data['trip_duration'] = (data['checkout_date'] - data['checkin_date']).dt.total_seconds() / 3600

    # Calculate number of days before booking
    data['days_before_booking'] = (data['checkin_date'] - data['booking_datetime']).dt.total_seconds() / 3600

    # Check if check-in date is a weekend
    data['is_weekend'] = data['checkin_date'].dt.weekday.isin([5, 6])

    # Extract the month from the booking date
    data['month_ordered'] = data['booking_datetime'].dt.month.astype(int)

    # Extract the month from the check-in date
    data['month_checkin'] = data['checkin_date'].dt.month.astype(int)

    # Calculate the period between hotel live date and booking date in hours
    data['hotel_live_period'] = (data['hotel_live_date'] - data['booking_datetime']).dt.total_seconds() / 3600

    """ -------------------------- preprocess dates END --------------------------"""

    """ -------------------------- count stuff START --------------------------"""

    # Calculate the number of orders per hotel_id
    data['no_orders_of_hotel'] = data.groupby('hotel_id')['hotel_id'].transform('count')

    # todo: think about hotel_id - we thought to drop it or dummies the problems is there are 25k

    # Calculate the number of orders per h_customer_id
    data['no_orders_of_customer'] = data.groupby('h_customer_id')['h_customer_id'].transform('count')

    """ -------------------------- count stuff END --------------------------"""

    """ -------------------------- dummies values start --------------------------"""

    # create dummies for hotel_country_code
    data = pd.get_dummies(data, columns=['hotel_country_code'])

    # create dummies for charge_option
    data = pd.get_dummies(data, columns=['charge_option'])

    # create dummies for accommadation_type_name
    data = pd.get_dummies(data, columns=['accommadation_type_name'])

    # create dummies for customer_nationality
    # data = pd.get_dummies(data, columns=['customer_nationality'])

    # count = len(data[(data['guest_is_not_the_customer'] == 0) & (
    #             data['customer_nationality'] != data['guest_nationality_country_name'])])
    # print(count)

    mismatched_rows = data[(data['guest_is_not_the_customer'] == 0) & (
                data['customer_nationality'] != data['guest_nationality_country_name'])]
    relevant_columns = ['customer_nationality', 'guest_nationality_country_name']
    # print(mismatched_rows[relevant_columns])

    #todo: where we left we have people who ordered their order (0) where customer_nationality != guest_nationality_country_name
    """ -------------------------- dummies values END --------------------------"""

    # todo: deal with out of range and mispelled values null values etc.


if __name__ == "__main__":
    main()
