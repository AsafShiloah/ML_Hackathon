from typing import Optional
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocess import load_data, preprocess_data, preprocess_Q1





# plot for each unique value in colum the perctentage of canceled orders

def plot_cancellation_percantage_with_feature_dist(df: pd.DataFrame, feature: str, label: pd.Series):
    # Assuming df is your DataFrame
    label = 'label'  # replace with your actual label column name

    df[label] = y
    # Filter the DataFrame to only include rows where 'label' is 1
    df_filtered = df[df[label] == 1]

    # Count the number of ones in 'label' for each unique value in 'feature'
    count_ones = df_filtered[feature].value_counts()

    # Calculate the total number of ones in 'label'
    total_ones = df[label].sum()

    # Calculate the percentage of ones for each feature
    percentage_ones = (count_ones / total_ones) * 100

    # Convert the Series to a DataFrame
    percentage_ones_df = percentage_ones.reset_index()
    percentage_ones_df.columns = [feature, 'percentage']

    # Add count_ones to the DataFrame
    percentage_ones_df['count_ones'] = count_ones.values


    # Calculate the percentage distribution of the feature
    feature_distribution = (df[feature].value_counts() / len(df)) * 100

    # Convert the Series to a DataFrame
    feature_distribution_df = feature_distribution.reset_index()
    feature_distribution_df.columns = [feature, 'feature_distribution']

    # Add count_feature to the DataFrame
    feature_distribution_df['count_feature'] = df[feature].value_counts().values

    # Create the subplot figure
    fig = make_subplots(rows=2, cols=1)

    # Add the first trace for the percentage of 1s in label
    fig.add_trace(
        go.Bar(x=percentage_ones_df[feature], y=percentage_ones_df['percentage'], text=percentage_ones_df['count_ones'],
               name='Percentage of 1s in label', textposition='outside'),
        row=1, col=1
    )

    # Add the second trace for the feature distribution
    fig.add_trace(
        go.Bar(x=feature_distribution_df[feature], y=feature_distribution_df['feature_distribution'],
               text=feature_distribution_df['count_feature'],
               name='Feature Distribution'),
        row=2, col=1
    )

    # Set the figure's layout
    # fig.update_layout(height=600, width=800,
    #                   title_text='Subplots of Percentage of 1s in label and Feature Distribution')

    feature = feature.replace('/', '_')
    fig.write_image(f'graphs/with_dist/percentage_of_cancellation_in_{feature}.png')

def pearson_correlation(df: pd.DataFrame, feature1: str, feature2: str):
    # Calculate the correlation between the two columns
    correlation = df[feature1].corr(df[feature2])

    # Display the correlation
    print(correlation)



def plot_cancellation_percentage(df: pd.DataFrame, feature: str, label: pd.Series):
    # Combine the two series into a DataFrame

    df['label'] = y
    df_filtered = df[df['label'] == 1]

    # Count the number of ones in 'label' for each unique value in 'feature'
    number_of_ones = df_filtered[feature].value_counts()

    # Calculate the total number of ones in 'label'
    feature_count = df[feature].value_counts()

    # Calculate the percentage of ones for each feature
    percentage_ones = (number_of_ones / feature_count) * 100

    feature_count_df = feature_count.reset_index()
    percentage_ones_df = percentage_ones.reset_index()
    # print(feature_count_df)
    # print(percentage_ones_df)

    # Convert the Series to a DataFrame
    # percentage_ones_df = percentage_ones.reset_index(drop=True, inplace=True)

    df_merge = pd.merge(percentage_ones_df, feature_count_df, on='index')
    # change columns name to be more clear
    df_merge.columns = [feature, 'percentage', 'feature_count']

    # change feature to type int
    df_merge[feature] = df_merge[feature].astype(int)

    # sort by percentage
    df_merge = df_merge.sort_values(by=[feature], ascending=False)
    # Create the bar plot
    fig = px.bar(df_merge, x=feature, y='percentage', color='feature_count',
                 labels={'percentage': 'Percentage of 1s in label', 'feature': 'feature'},
                 title='Percentage of 1s in label for each unique value in feature')

    # Set the position of the text
    # fig.update_traces(textposition='outside')
    #
    # fig.show()
    feature = feature.replace('/', '_')
    fig.write_image(f'graphs/without_dist/percentage_of_cancellation_in_{feature}.png')

def feature_eval_graphs(X, y):
    len = X.shape[1]
    for i, col in enumerate(X.columns):
        if 'dummy_' not in col and col[0].isdigit():
            print(col, (i / len)*100)
            plot_cancellation_percentage(X, col, y)

def categorial_correlation(feature1: str, feature2: str, df: pd.DataFrame):
    # Create a scatter plot
    pass
    # Display the plot
    # fig.show()


def eda(X: pd.DataFrame, y: Optional[pd.Series] = None):
    # feature_eval_graphs(X, y)
    categorial_correlation('customer_nationality', 'guest_nationality_country_name', X)






if __name__ == '__main__':
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                    'hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)
    # print(X.columns)
    eda(X, y)
