
from preprocess import *
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import pandas as pd
import plotly.figure_factory as ff
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
import plotly.graph_objects as go

def naive_bayes(X_train, X_test, y_train, y_test):
    # Instantiate the Naive Bayes classifier
    gnb = GaussianNB()
    # Train the model
    gnb.fit(X_train, y_train)
    # Predict
    y_pred = gnb.predict(X_test)
    # Create confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"NaiveBayes F1 Score: {f1}")
    # Convert confusion matrix to DataFrame
    df_cm = pd.DataFrame(cnf_matrix, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])

    # Create a heatmap
    fig = ff.create_annotated_heatmap(z=df_cm.values, x=list(df_cm.columns), y=list(df_cm.index), colorscale='Viridis')
    fig.update_layout(height=500, width=500, title_text="<i><b>Confusion matrix</b></i>")
    fig.show()

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, gnb.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig = go.Figure(data=[
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc),
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash'))
    ])
    fig.update_layout(
        title_text='Receiver Operating Characteristic',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        autosize=False,
        width=500,
        height=500
    )
    fig.show()




def ensemble_naive_bayes(X_train, X_test, y_train, y_test, n_estimators_range=(10, 100, 10)):
    best_f1 = 0
    best_estimator_size = 0
    f1_scores = []

    for n_estimators in range(*n_estimators_range):
        # Create a Naive Bayes classifier
        gnb = GaussianNB()
        # Create a BaggingClassifier
        bagging_clf = BaggingClassifier(gnb, n_estimators=n_estimators, random_state=0)
        # Train the model
        bagging_clf.fit(X_train, y_train)
        # Make predictions
        y_pred = bagging_clf.predict(X_test)
        # Calculate the F1 score
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

        # If the score for the current number of estimators is better than the current best score, update the best score and the best estimator size
        if f1 > best_f1:
            best_f1 = f1
            best_estimator_size = n_estimators

    print(f"Best Ensemble Size: {best_estimator_size}")

    # Plot the F1 scores
    fig = go.Figure(data=go.Scatter(x=list(range(*n_estimators_range)), y=f1_scores))
    fig.update_layout(title='F1 Scores for Different Ensemble Sizes', xaxis_title='Ensemble Size', yaxis_title='F1 Score')
    fig.show()

    return best_estimator_size





def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date','hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    naive_bayes(X_train, X_test, y_train, y_test)
    # ensemble_naive_bayes(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    np.random.seed(0)
    main()