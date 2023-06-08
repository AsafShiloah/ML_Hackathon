from sklearn.ensemble import RandomForestClassifier

from preprocess import *
import pandas as pd
import numpy as np
from preprocess import *
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def select_best_features(X, y, k=1):
    # Encode categorical variables if present
    le = LabelEncoder()
    X_encoded = X.apply(le.fit_transform)

    # Perform feature selection using mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_encoded, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_features = X.columns[selected_indices]

    # Get feature scores
    feature_scores = selector.scores_

    # Plot feature scores
    fig = go.Figure(data=[go.Bar(y=feature_scores, x=X.columns, orientation='v')])
    fig.update_layout(
        title='Feature Scores',
        xaxis_title='Score',
        yaxis_title='Features',
        yaxis=dict(autorange="reversed")
    )
    pio.show(fig)

    # Find the most relevant feature and its score
    max_score_index = feature_scores.argmax()
    most_relevant_feature = X.columns[max_score_index]
    most_relevant_score = feature_scores[max_score_index]

    return most_relevant_feature, most_relevant_score

def mutual_information_2(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    most_relevant_feature, score = select_best_features(X_train, y_train, k=1)
    print("\n")
    print("Mutual Information: Mutual information is utilized as the correlation measure to assess the relationship\n"
      "between each feature and the target variable y.\n"
      "Mutual information quantifies the statistical dependency between variables based on the information gain concept.\n"
      "It measures how much information about one variable can be obtained from another variable.\n")

    print("The function aims to assist in feature selection for prediction tasks,\n"
      "highlighting the feature that provides the most informative relationship with the target variable,\n"
      "aiding in building predictive models or flagging orders at risk based on relevant features.\n")
    print("Most Relevant Feature:", most_relevant_feature)
    print("Score:", score)


def mutual_information(data):
    from sklearn.feature_selection import mutual_info_classif
    import plotly.express as px

    # Assuming df is your DataFrame and 'cancellation' is your target column
    X,y = preprocess_Q1(data)

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


def random_forest_classifier(data):
    X,y = preprocess_Q1(data)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)

    # Get the feature importances
    importances = clf.feature_importances_

    # Create a DataFrame for visualization
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    # Display the top 10 most important features
    print(feature_importances[:10])

    # Create a bar chart with Plotly
    fig = px.bar(feature_importances[:10].sort_values('importance', ascending=True),
                 x='importance',
                 y='feature',
                 orientation='h',
                 title='Feature Importance',
                 labels={'importance': 'Importance', 'feature': 'Feature'},
                 width=800, height=600)

    fig.show()


def RFE(X,y):
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    import plotly.express as px

    # Assuming df is your DataFrame and 'cancellation' is your target column
    # Set up the model
    estimator = LogisticRegression(max_iter=1000)

    # Recursive Feature Elimination with Cross Validation
    selector = RFECV(estimator, step=1, cv=5)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform RFE on scaled data
    selector.fit(X_scaled, y)

    # Get the selected features
    selected_features = X.columns[selector.support_]

    estimator.fit(X[selected_features], y)

    # Get coefficients
    coefficients = estimator.coef_[0]

    # For visualization
    feature_importances = pd.DataFrame({'feature': selected_features, 'importance': coefficients})

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    # Plotting using Plotly
    fig = px.bar(feature_importances[:10].sort_values('importance', ascending=True),
                 x='importance',
                 y='feature',
                 orientation='h',
                 title='Feature Importance',
                 labels={'importance': 'Importance', 'feature': 'Feature'},
                 width=800, height=600)

    fig.show()


def correlation_matrix(data):
    import plotly.figure_factory as ff

    # Calculate correlation matrix
    corr = data.corr()

    # Create a heatmap
    fig = ff.create_annotated_heatmap(z=corr.values,
                                      x=list(corr.columns),
                                      y=list(corr.index),
                                      annotation_text=corr.round(2).values,
                                      showscale=True)
    fig.write_image('feature_correlation_matrix.png')


def Lasso_correlation(X,y):
    from sklearn.linear_model import LassoCV

    # Perform Lasso Regression with Cross Validation to find optimal alpha
    lasso = LassoCV(cv=5).fit(X, y)

    # Coefficients of features
    coef = pd.Series(lasso.coef_, index=X.columns)

    # Get the features with non-zero coefficients
    selected_features = coef[coef != 0].index.tolist()

    # Assuming coef from the previous LassoCV
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': coef.abs()})

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    # Plotting using Plotly
    fig = px.bar(feature_importances[:10].sort_values('importance', ascending=True),
                 x='importance',
                 y='feature',
                 orientation='h',
                 title='Feature Importance',
                 labels={'importance': 'Importance', 'feature': 'Feature'},
                 width=800, height=600)

    fig.show()

def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date','hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)

    # random_forest_classifier(data)
    # RFE(X,y)
    # print("finish RFE")
    # Lasso_correlation(X,y)
    # print("finish Lasso_correlation")
    # correlation_matrix(data)
    # print("finish correlation_matrix")
    mutual_information(data)



if __name__ == '__main__':
    np.random.seed(0)
    main()