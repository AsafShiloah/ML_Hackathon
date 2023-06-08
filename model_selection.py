from sklearn.linear_model import Lasso

from preprocess import *
from utils import *

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, make_scorer
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.manifold import Isomap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier

def logistic_regression(X_train, X_test, y_train, y_test):
    # Initializing the logistic regression model
    lr = LogisticRegression()

    # Fitting the data to the model
    lr.fit(X_train, y_train)

    # Predicting the values for the test set
    y_pred = lr.predict(X_test)

    # Printing the accuracy of the model
    f1 = f1_score(y_test, y_pred, average='macro')
    print('Decision Tree Accuracy: ', f1)


def knn(X_train, X_test, y_train, y_test):

    knn = KNeighborsClassifier(n_neighbors=5)

    # Fit the data to the model
    knn.fit(X_train, y_train)

    # Predict the values for the test set
    y_pred = knn.predict(X_test)

    # Compute and print the F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('F1 Score: ', f1)


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
    dt = DecisionTreeClassifier(max_depth=4)

    # Fit the data to the model
    dt.fit(X_train, y_train)

    # Predict the values for the test set
    y_pred = dt.predict(X_test)

    # Printing the accuracy of the model
    f1 = f1_score(y_test, y_pred, average='macro')
    print('Decision Tree Accuracy: ', f1)


def perform_cross_validation(X, y, depths, cv=5):
    """
    Perform cross-validation for hyperparameter tuning using a decision tree classifier.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.
        param_grid (dict): Dictionary specifying the hyperparameter grid to search.
        cv (int, optional): Number of cross-validation folds. Default is 5.

    Returns:
        float: The average accuracy score across cross-validation folds.
    """
    scores_by_depth = {}  # Dictionary to store scores for each depth

    # Define F1 scorer
    f1_scorer = make_scorer(f1_score)

    for depth in depths:
        dt_classifier = DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(dt_classifier, X, y, cv=cv, scoring=f1_scorer)
        scores_by_depth[depth] = scores.mean()

    return scores_by_depth


def random_forrest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100)

    # Fit the model to the data
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_

    # Create a list of feature names
    feature_names = X_train.columns

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for i, index in enumerate(indices):
        print(f"{i + 1}. {feature_names[index]} ({importances[index]:.4f})")
    # Predict the values for the test set
    y_pred = rf.predict(X_test)

    # Compute and print the F1 score
    f1 = f1_score(y_test, y_pred, average='macro')
    print('F1 Score: ', f1)
    return


def omri(X,y):
    params = [2, 4, 6, 8]

    scores_by_depth = perform_cross_validation(X, y, params)

    # Print the average scores for each tree depth
    for depth, score in scores_by_depth.items():
        print(f"Tree Depth: {depth}, Average Score: {score}")


def ada_boost(X, y):

    # Create the AdaBoost classifier
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)

    # Perform k-fold cross-validation with F1 score using macro averaging
    k = 5  # Number of folds
    scores = cross_val_score(adaboost, X, y, cv=k, scoring='f1_macro')

    # Print the average F1 score across all folds
    print("Average F1 Score (Macro):", scores.mean())


def random_forrest_k_cross(X, y):
    # Create the Random Forest classifier

    rf = RandomForestClassifier(n_estimators=100)

    # Perform k-fold cross-validation with F1 score using macro averaging
    k = 5  # Number of folds
    scores = cross_val_score(rf, X, y, cv=k, scoring='f1_macro')

    # Print the average F1 score across all folds
    print("Average F1 Score (Macro):", scores.mean())


def random_forrest_by_k(X_train, X_test, y_train, y_test):
    # Define a range of n_estimators values to try
    n_estimators_range = range(1, 150, 20)

    # Initialize a list to store the F1 scores
    f1_scores = []

    # Loop over the n_estimators values
    for n_estimators in n_estimators_range:
        print(n_estimators)
        # Initialize the Random Forest classifier with the current number of estimators
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Fit the data to the model
        rf.fit(X_train, y_train)

        # Predict the values for the test set
        y_pred = rf.predict(X_test)

        # Compute the F1 score and append it to the list of F1 scores
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

    # Create a DataFrame with the F1 scores
    df = pd.DataFrame({
        'n_estimators': n_estimators_range,
        'F1 Score': f1_scores
    })

    # Create a line plot of the F1 scores
    fig = px.line(df, x='n_estimators', y='F1 Score', title='F1 Score as a Function of n_estimators')
    fig.show()


def ridge_regression(X_train, X_test, y_train, y_test):
    # Create the Ridge Classifier
    ridge_classifier = RidgeClassifier()

    # Define the alpha values to search
    alphas = [0.1, 1.0, 10.0]

    # Perform grid search
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(ridge_classifier, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    # Get the alpha values and corresponding mean test scores
    alphas = grid_search.cv_results_['param_alpha'].data.astype(float)
    mean_test_scores = grid_search.cv_results_['mean_test_score']

    # Plot the results
    fig = go.Figure(data=go.Scatter(x=alphas, y=mean_test_scores, mode='markers'))
    fig.update_layout(
        xaxis_type='log',
        xaxis_title='Alpha',
        yaxis_title='Mean F1 Score (Macro)',
        title='Grid Search: Alpha vs. Mean F1 Score'
    )
    fig.show()

    # Get the best alpha value
    best_alpha = grid_search.best_params_['alpha']

    # Fit the data to the model with the best alpha
    ridge_classifier_best = RidgeClassifier(alpha=best_alpha)
    ridge_classifier_best.fit(X_train, y_train)

    # Predict the target variable for the test set
    y_pred = ridge_classifier_best.predict(X_test)

    # Calculate and print the F1 score
    f1 = f1_score(y_test, y_pred, average='macro')
    print('Best Alpha:', best_alpha)
    print('F1 Score (Macro):', f1)


def lasso_regression(X_train, X_test, y_train, y_test):
    # Create the Lasso regression model
    lasso = Lasso()

    # Define the alpha values to search
    alphas = [0.1, 1.0, 10.0]

    # Perform grid search
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    # Get the alpha values and corresponding mean test scores
    alphas = grid_search.cv_results_['param_alpha'].data.astype(float)
    mean_test_scores = grid_search.cv_results_['mean_test_score']

    # Plot the results
    fig = go.Figure(data=go.Scatter(x=alphas, y=mean_test_scores, mode='markers'))
    fig.update_layout(
        xaxis_type='log',
        xaxis_title='Alpha',
        yaxis_title='F1 Score (Macro)',
        title='Grid Search: Alpha vs. F1 Score (Macro)'
    )
    fig.show()

    # Get the best alpha value
    best_alpha = grid_search.best_params_['alpha']

    # Fit the data to the model with the best alpha
    lasso_best = Lasso(alpha=best_alpha)
    lasso_best.fit(X_train, y_train)

    # Predict the target variable for the test set
    y_pred = lasso_best.predict(X_test)

    # Convert predictions to binary values based on a threshold
    threshold = 0.5
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Calculate and print the F1 score
    f1_macro = f1_score(y_test, y_pred_binary, average='macro')
    print('Best Alpha:', best_alpha)
    print('F1 Score (Macro):', f1_macro)


def main():
    data = load_data('train_data.csv', parse_dates=['booking_datetime', 'checkin_date', 'checkout_date',
                                                    'hotel_live_date'])
    data = preprocess_data(data)
    X, y = preprocess_Q1(data)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, y_train = preprocess_train(X_train, y_train)
    # print(X_train.shape, y_train.shape)

    random_forrest(X_train, X_test, y_train, y_test)
    # ridge_regression(X_train, X_test, y_train, y_test)
    # logistic_regression(X_train, X_test, y_train, y_test)
    # decision_trees(X_train, X_test, y_train, y_test)
    # random_forrest_by_k(X_train, X_test, y_train, y_test)

    # random_forrest_k_cross(X, y)
    # ada_boost(X,y)
    # lasso_regression(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    np.random.seed(0)
    main()

