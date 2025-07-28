'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np
from datetime import timedelta



# Your code here


# Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
pred_universe = pd.read_csv('data/pred_universe_raw.csv')
arrest_events_raw = pd.read_csv('data/arrest_events_raw.csv')
# print(pred_universe_raw.to_string())
# print(arrest_events_raw.to_string())

#Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
df_arrests = pd.merge(pred_universe,arrest_events_raw, on='person_id', how='outer')
# print(df_arrests.to_string())
df_arrests.to_csv('data/df_arrests.csv', index=False)

# Create a column in `df_arrests` called `y` which equals 
# 1 if the person was arrested for a felony crime in the 365 days after 
# their arrest date in `df_arrests`. 
df_arrests['y'] = np.where((df_arrests['charge_degree'] == 'felony') & (pd.to_datetime(df_arrests['arrest_date_event']) - pd.to_datetime(df_arrests['arrest_date_univ']) < timedelta(days = 365)),1,0)
# print(df_arrests.to_string())


# Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
# Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 

print('What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?')
rearrests = (len([entry for entry in df_arrests["y"] if entry == 1]))
total_arrests = len(df_arrests['y'])
percentage = rearrests / total_arrests
percentage = round(percentage, 3)
print(f'the percentage of arrestees is {percentage * 100}%.')

# Create a predictive feature for `df_arrests` that is called `current_charge_felony` 
# which will equal one if the current arrest was for a felony charge, and 0 otherwise.

df_arrests['current_charge_felony'] = np.where(df_arrests['charge_degree'] == ('felony'),1,0)
# print(df_arrests.to_string())

#Use a print statment to print this question and its answer: What share of current charges are felonies?
print('what share of current charges are felonies?')
felonies = (len([entry for entry in df_arrests["current_charge_felony"] if entry == 1]))
percentage_of_felonies = felonies/total_arrests
percentage_of_felonies = round(percentage_of_felonies, 3)
print(f'the current share of charges that are felonies is about {percentage_of_felonies * 100}%.')

#Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year`
#  which is the total number arrests in the one year prior to the current charge. 

latest_charge = df_arrests['arrest_date_event'].max()
# print(latest_charge)
df_arrests['fel_arrests_last_year'] = np.where((df_arrests["charge_degree"] == 'felony') & (pd.to_datetime(df_arrests['arrest_date_event']) <= (pd.to_datetime(latest_charge)) - timedelta(days = 365)),1,0)
# print(df_arrests.to_string())
num_fel_arrests_last_year = (len([entry for entry in df_arrests["fel_arrests_last_year"] if entry == 1]))
print(f'the total number of arrests last year were {num_fel_arrests_last_year}.')

# Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?

num_fel_arrests_last_year = df_arrests['fel_arrests_last_year'] = ((df_arrests["charge_degree"] == 'felony') & (pd.to_datetime(df_arrests['arrest_date_event']) <= (pd.to_datetime(latest_charge)) - timedelta(days = 365)))
df_arrests['num_fel_arrests_last_year'] = num_fel_arrests_last_year
# print(df_arrests.to_string())

felony_counts = num_fel_arrests_last_year.groupby(df_arrests['person_id']).size()
# print(felony_counts.to_string())
average_felony_counts = round(felony_counts.mean(), 2)
print(f"the average number of felony arrests in the last year was {average_felony_counts} felonies.")

# - Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
# print(pred_universe['num_fel_arrests_last_year'].mean()) 
#I DIDNT UNDERSTAND, not working

# - Print pred_universe.head()
print(pred_universe.head())
df_arrests.to_csv('data/df_arrests.csv', index=False)




# - Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py



