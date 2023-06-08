
from preprocess import *
import pandas as pd
import numpy as np
from preprocess import *
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def select_best_features(X, y, k=5):
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
    fig = go.Figure(data=[go.Bar(x=feature_scores, y=X.columns, orientation='h')])
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


def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date','hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)
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

if __name__ == '__main__':
    np.random.seed(0)
    main()