
from preprocess import *
import numpy as np
import plotly.graph_objects as go


def histogram_q4(X):
    # Create a histogram to visualize the distribution of cancellations
    fig = go.Figure(data=[go.Histogram(x=X['cancellation_margin'], nbinsx=300)])

    # Customize the plot layout
    fig.update_layout(title='Distribution of Cancellations',
                      xaxis_title='Cancellation Margin (Days)',
                      yaxis_title='Count')

    # Show the plot
    fig.show()

    fig = go.Figure(data=go.Box(x=X['cancellation_margin'], y=X['original_selling_amount']))
    fig.update_layout(title='Distribution of Booking Prices by Cancellation Margin',
                       xaxis_title='Cancellation Margin (Days)',
                       yaxis_title='Booking Price')
    fig.show()


if __name__ == "__main__":
    np.random.seed(0)
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date','hotel_live_date','cancellation_datetime'])
    data['cancellation_margin'] = (data['checkin_date'] - data['cancellation_datetime']).dt.days
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)
    histogram_q4(X)

