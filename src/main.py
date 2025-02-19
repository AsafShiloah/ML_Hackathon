import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from preprocess import *
import pandas as pd
import plotly.figure_factory as ff
from sklearn.feature_selection import mutual_info_classif
import plotly.express as px


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
    model1 = RandomForestClassifier(n_estimators=20)
    model1.fit(X1_train, y1_train)
    y1_pred = model1.predict(X1_test)

    Q1_prediction = pd.concat([id, pd.Series(y1_pred)], axis=1)
    Q1_prediction.columns = ['id', 'cancellation']
    Q1_prediction.to_csv('agoda_cancellation_prediction_test.csv', index=False)
    # joblib.dump(rf, 'model1.pkl')

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

    ##################### Q3 ############################
    #
    data = preprocess_data(train)
    X, y = preprocess_Q1(data)

    ### correlation matrix ###
    # Combine X and y into a single DataFrame
    data = pd.concat([X, y], axis=1)
    # Calculate correlation matrix
    corr = data.corr()

    # Get the top 5 most correlated features with the target variable
    top_features = corr.nlargest(5, y.name)[y.name].index

    # Create a correlation matrix for the top features
    corr_top_features = data[top_features].corr()

    # Create a heatmap
    fig = ff.create_annotated_heatmap(z=corr_top_features.values,
                                      x=list(corr_top_features.columns),
                                      y=list(corr_top_features.index),
                                      annotation_text=corr_top_features.round(2).values,
                                      colorscale='Viridis',
                                      showscale=True)
    fig.update_layout(title='Top 5 Most Correlated Features with Target',
                      xaxis_title='Features',
                      yaxis_title='Features')
    fig.show()

    ###### mutual_information ######

    X, y = preprocess_Q1(data)

    # Calculate Mutual Information
    mi = mutual_info_classif(X, y)

    # Create a DataFrame for visualization
    mi_series = pd.Series(mi, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)

    # Plotting using Plotly
    fig = px.bar(mi_series[:10].sort_values(ascending=True),
                 x=mi_series[:10],
                 y=mi_series[:10].index,
                 orientation='h',
                 labels={'x': 'Mutual Information', 'y': 'Feature'},
                 title='Top 10 Features ranked by Mutual Information')

    fig.show()

    ###### random forest ######

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)

    # Get the feature importances
    importances = clf.feature_importances_

    # Create a DataFrame for visualization
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    # Create a bar chart with Plotly
    fig = px.bar(feature_importances[:10].sort_values('importance', ascending=True),
                 x='importance',
                 y='feature',
                 orientation='h',
                 title='Feature Importance',
                 labels={'importance': 'Importance', 'feature': 'Feature'},
                 width=800, height=600)

    fig.show()


if __name__ == '__main__':
    main(sys.argv[1:])


