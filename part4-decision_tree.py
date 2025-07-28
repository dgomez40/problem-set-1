'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

# - Read in the dataframe(s) from PART 3
df_arrests_train = pd.read_csv("data/df_arrests_train.csv")
df_arrests_test = pd.read_csv("data/df_arrests_test.csv")

#- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 

param_grid_dt = {"max_depth" : [2, 4, 6]}

#- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
dt_model = DTC()

#- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
gs_cv_dt = GridSearchCV(cv=5, param_grid=param_grid_dt, estimator=dt_model)

#- Run the model 
features = ['num_fel_arrests_last_year', 'current_charge_felony']
x_train = df_arrests_train[features]
y_train = df_arrests_train['y']
x_test = df_arrests_test[features]

gs_cv_dt.fit(x_train, y_train)
df_arrests_train['train_dt'] = gs_cv_dt.fit(x_train, y_train)

#what was the optimal value for max_depth?

best_depth = gs_cv_dt.best_params_['max_depth']
print("What was the optimal value for max_depth?")
print(f"The optimal value was: {best_depth}meaning it had the most regularization")

#- Now predict for the test set. Name this column `pred_dt` 
df_arrests_test['pred_dt'] = gs_cv_dt.predict(x_test)
print(df_arrests_test)

df_arrests_train.to_csv('data/df_arrests_train.csv')
df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)









