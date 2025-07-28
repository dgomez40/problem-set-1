'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as DTC
from datetime import timedelta
from part5_calibration_plot import calibration_plot
import matplotlib.pyplot as plt
import seaborn as sns



# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`

    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    pred_universe_raw.to_csv('data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('data/arrest_events_raw.csv', index=False)

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = pd.read_csv('data/df_arrests.csv')

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')

    # PART 4: Call functions/instanciate objects from decision_tree
    #updated dataframes from part 3
    df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
    # PART 5: Call functions/instanciate objects from calibration_plot
    #logistic regression calibration
   
    calibration_plot(df_arrests_test['y'], df_arrests_test['pred_lr'], n_bins=5)

    #decision tree calibration
    calibration_plot(df_arrests_test['y'], df_arrests_test['pred_dt'], n_bins=5)

    #Which model is more calibrated? Print this question and your answer. 
    print("Answer: Based on the calibration plots, the decision tree model fit the sample data better.")

if __name__ == "__main__":
    main()