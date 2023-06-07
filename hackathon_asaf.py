import pandas as pd
from pandas import DataFrame
import sklearn as sk
from typing import Optional, NoReturn
import pycountry

def get_country_code(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_2
    except AttributeError:
        print(f"Invalid country name: {country_name}")
        return None





def load_data(path: str) -> DataFrame:
    """
    Load data from a CSV file into a DataFrame.
    :param path: string
    :return: DataFrame
    """
    return pd.read_csv(path)


def preprocess_data(X: pd.DataFrame, y:Optional[pd.Series] = None) :
    """
    Preprocess the data.
    :param df: DataFrame
    :return: DataFrame
    """

    # fill na in origin country with
    X['origin_country_code'] = X['origin_country_code'].fillna(get_country_code(X['country_name'])) if X['country_name'] else X['country_name'].fillna('KR')
    # create dummies for origin country:
    X = pd.get_dummies(X, columns=['origin_country_code'], prefix='code_') # TODO: check if need GLOBAL

    # create dummies for language:
    X = pd.get_dummies(X, columns=['language'], prefix='lang_')  # TODO: check if need GLOBAL


    #TODO:replace UNKNOWN in original payment method:
        # X['original_payment_method'] = X['original_payment_method'].replace('UNKNOWN', 'OTHER')

    # create dummies for payment method:
    X = pd.get_dummies(X, columns=['original_payment_method'], prefix='pay_method_')  # TODO: check if need GLOBAL

    # create dummies for original payment type:
    X = pd.get_dummies(X, columns=['original_payment_type'], prefix='pay_type_')  # TODO: check if need GLOBAL

    # create dummies for original payment currency:
    X = pd.get_dummies(X, columns=['original_payment_currency'], prefix='pay_currency_')  # TODO: check if need GLOBAL, cor with country code

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


    # cancelation policy code:

     # X['dayes_before_checkin'] = X['cancellation_policy_code'].fillna(0)







if __name__ == "__main__":
    path = "train_data.csv"
    df = load_data(path)
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
     #TODO split no show via _ and then via 'D'









