import pandas as pd
import sklearn as sk
import numpy as np
import plotly.graph_objects as go

from preprocess import *
from utils import *
from scipy.stats import norm
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import roc_curve, auc


def logistic_regression(X_train, X_test, y_train, y_test):
    # Initializing the logistic regression model
    lr = LogisticRegression()

    # Fitting the data to the model
    lr.fit(X_train, y_train)

    # Predicting the values for the test set
    y_pred = lr.predict(X_test)

    # Printing the accuracy of the model
    print('Logistic Accuracy: ', metrics.accuracy_score(y_test, y_pred))


def knn(X_train, X_test, y_train, y_test):
    # Initialize the KNN classifier with some value of K, for example, 5
    knn = KNeighborsClassifier(n_neighbors=5)

    # Fit the data to the model
    knn.fit(X_train, y_train)

    # Predict the values for the test set
    y_pred = knn.predict(X_test)

    # Printing the accuracy of the model
    print('KNN Accuracy: ', metrics.accuracy_score(y_test, y_pred))


def confusion_matrix(y_test, y_pred):
    # Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert Confusion Matrix to DataFrame
    cm_df = pd.DataFrame(cm, index=[i for i in "01"],
                         columns=[i for i in "01"])

    # Create a heatmap using Plotly
    fig = ff.create_annotated_heatmap(z=cm, x=list(cm_df.columns), y=list(cm_df.index), colorscale='Viridis')
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>', xaxis=dict(title='Predicted label'),
                      yaxis=dict(title='True label'))

    fig.show()

def decision_trees(X_train, X_test, y_train, y_test):
    # Initialize the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=0)

    # Fit the data to the model
    dt.fit(X_train, y_train)

    # Predict the values for the test set
    y_pred = dt.predict(X_test)

    # Printing the accuracy of the model
    print('Decision Tree Accuracy: ', metrics.accuracy_score(y_test, y_pred))


def graph_sigmoid(X,y):
    c = [custom[0], custom[-1]]
    model = LogisticRegression(penalty="none").fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]

    go.Figure([
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=[-0.1] * X.shape[0], mode='markers',
                     marker=dict(color=y, symbol="circle-open", colorscale=c, reversescale=True, size=1)),
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y_prob, mode='markers',
                     marker=dict(color=y_prob, colorscale=custom, reversescale=True, showscale=True, size=3))],
        layout=go.Layout(title=r"$(2)\text{ Predicted Class Probabilities}$",
                         scene_aspectmode="cube", showlegend=False,
                         scene=dict(xaxis_title="Feature 1",
                                    yaxis_title="Feature 2",
                                    zaxis_title="Probabilty of Assigning Class 1",
                                    camera=dict(eye=dict(x=1, y=-1.8, z=.1))))).show()


def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                    'hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    logistic_regression(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    decision_trees(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    main()
