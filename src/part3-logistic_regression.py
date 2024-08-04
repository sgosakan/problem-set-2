'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


def logistic_regression():
# Your code here
    df_arrests = pd.read_csv('data/df_arrests.csv')

    # create two dfs from df_arrests
    df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'])

    # create features list
    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    # create a param grid
    param_grid = {'C': [0.01, 1, 100]}


    lr_model = lr()

    gridsearch_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv = 5)

    gridsearch_cv.fit(df_arrests_train[features, df_arrests_train['y']])

    optimal_C = gridsearch_cv.best_params_['C']
    print(f"The optimal value for C was {optimal_C}")

    if optimal_C == 0.01:
        regularization = "most"
    elif optimal_C == 100:
        regularization = "least"
    else:
        regularization = "in the middle"

    print(f"The regularization was {regularization}")

    df_arrests_test['pred_lr'] = gridsearch_cv.predict(df_arrests_test[features])

    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

if __name__ == "__main__":
    logistic_regression()