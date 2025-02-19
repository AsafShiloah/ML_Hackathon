
from preprocess import *
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score



def gradient_boosting(X_train, X_test, y_train, y_test, learning_rate=0.1, n_estimators=100, max_depth=5):
    # Initialize a Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

    # Fit the classifier to the training data
    gbc.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = gbc.predict(X_test)

    # Compute F1 score of the prediction
    f1 = f1_score(y_test, y_pred,average='macro')

    print('F1 Score:', f1)

    # Plot the confusion matrix
    plot_confusion_matrix(gbc, X_test, y_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, gbc.decision_function(X_test))
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig = go.Figure(data=[
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc),
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random')
    ])
    fig.update_layout(autosize=False, title='Receiver Operating Characteristic', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    fig.show()

    return f1

def gradient_boosting_cross_val(X, y, k=5, learning_rate=0.1, n_estimators=100, max_depth=3):
    # Initialize a Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

    # Compute cross-validation scores
    scores = cross_val_score(gbc, X, y, cv=k, scoring='f1_micro')

    # Create a bar plot to visualize the cross-validation scores
    fig = go.Figure([go.Bar(x=list(range(1, k+1)), y=scores, name='Cross-Validation Scores')])
    fig.update_layout(title='Cross-Validation Scores', xaxis=dict(title='Fold'), yaxis=dict(title='F1 Score'), autosize=False)
    fig.show()

    return scores


def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date','hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    gradient_boosting(X_train, X_test, y_train, y_test)
    # scores = gradient_boosting_cross_val(X, y, k=5, learning_rate=0.1, n_estimators=100, max_depth=3)
    # print("Cross-Validation Scores: ", scores)
    # print("Average F1 Score: ", scores.mean())


if __name__ == '__main__':
    np.random.seed(0)
    main()