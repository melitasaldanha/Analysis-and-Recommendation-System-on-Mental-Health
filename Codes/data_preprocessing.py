from glob import glob
import pandas as pd
import numpy as np


# Common Functions

def upload_files(path):
  filenames = glob(path+'/*.XPT')
  dataframes = [pd.read_sas(f) for f in filenames]

  return dataframes


def replace_col_name(df, curr_name, replace_by_name, col_list):
  if curr_name in col_list:
    df.rename(columns={curr_name:replace_by_name}, inplace=True)


def clean_merged_df(merged_df, categorical_cols, numerical_datatype):

  # Drop columns / rows having > 50% NaN
  merged_df.dropna(thresh=len(merged_df)/2, axis=1, inplace=True)
  merged_df.dropna(thresh=len(merged_df.columns)/2, axis=0, inplace=True)

  # Replace NaN by mode (Categorical columns) / mean (Numerical columns)
  for column in merged_df.columns:
    if column in categorical_cols:
      merged_df[column].fillna(merged_df[column].mode()[0], inplace=True)
    else:
      merged_df[column].fillna(merged_df[column].mean(), inplace=True)

  # Convert to int
  for column in merged_df.columns:
    if column in categorical_cols:
      merged_df[column] = merged_df[column].astype(int)
    else:
      merged_df[column] = merged_df[column].astype(numerical_datatype)


# Functions for each dataset

def preprocess_mental_health_depression(df):

  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'ECD010', 'ECQ020', 'ECD070A', 'ECD070B', 'ECQ080', 'ECQ090','WHQ030E', 'MCQ080E', 'ECQ150']
  categorical_cols = considered_cols

  for i in range(len(df)):

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_early_childhood(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'ECD010', 'ECQ020', 'ECD070A', 'ECD070B', 'ECQ080', 'ECQ090','WHQ030E', 'MCQ080E', 'ECQ150']
  numerical_cols = ['ECD010', 'ECD070A', 'ECD070B']
  categorical_cols = set(considered_cols)-set(numerical_cols)
  
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    for column in df[i].columns:
      if(column in categorical_cols):
        df[i].replace({7: -10, 9: -10}, inplace=True)
      else:
        df[i].replace({7777: -10, 9999: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_food_security(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'FSD032A', 'FSD032B', 'FSD032C', 'FSD032D', 'FSD032E', 'FSD041','FSD081', 'FSDHH', 'FSD411']
  categorical_cols = considered_cols

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_occupation(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'OCQ210', 'OCD231', 'OCD241', 'OCQ260', 'OCQ265', 'OCQ290G','OCD390G']
  numerical_cols = ['OCD231', 'OCD241']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    for column in df[i]:
      if column=='OCQ260':
        df[i].replace({77: -10, 99: -10}, inplace=True)
      elif column in categorical_cols:
        df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_sleep_disorders(df):

  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'SLD010H']
  categorical_cols = considered_cols

  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(float).astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({77: -10, 99: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_housing_characteristics(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'HOQ065', 'HOQ070', 'HOQ080']
  categorical_cols = considered_cols
  
  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_sexual_behavior(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'SXQ020', 'SXD030', 'SXQ260', 'SXQ292', 'SXQ294']
  numerical_cols = ['SXD030']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Replace column names (Same question column with different names)
    replace_col_name(df[i], 'SXQ021', 'SXQ020', df[i].columns)
    replace_col_name(df[i], 'SXD031', 'SXD030', df[i].columns)

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    for column in df[i].columns:
      if column in numerical_cols:
        df[i].replace({77: -10, 99: -10}, inplace=True)
      else:
        df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_diet_behavior_and_nutrition(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'DBQ700', 'DBQ197', 'DBD895', 'DBD900', 'DBD905', 'DBD910', 'DBQ915', 'DBQ920']
  numerical_cols = ['DBD895', 'DBD900', 'DBD905', 'DBD910']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Remove row where DBD895 = 5555 (Only 1 category, rest all numerical values in this column)
    if 'DBD859' in df[i].columns:
      df[i] = df[i][df[i]['DBD895'] != 5555]

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    for column in df[i].columns:
      if column in numerical_cols:
        df[i].replace({7777: -10, 9999: -10}, inplace=True)
      else:
        df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_hospital_utilization_and_access_care(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'HUQ010', 'HUQ020', 'HUQ050', 'HUQ090']
  categorical_cols = considered_cols

  # Pre-process
  for i in range(len(df)):

    # Replace column names (Same question column with different names)
    replace_col_name(df[i], 'HUQ051', 'HUQ050', df[i].columns)

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    for column in df[i].columns:
      if column=='HUQ050':
        df[i].replace({77: -10, 99: -10}, inplace=True)
      else:
        df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_consumer_behavior(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'CBD070', 'CBD090', 'CBD110', 'CBD120', 'CBD130']
  numerical_cols = ['CBD070', 'CBD090', 'CBD110', 'CBD120', 'CBD130']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Replace column names (Same question column with different names)
    replace_col_name(df[i], 'CBD071', 'CBD070', df[i].columns)
    replace_col_name(df[i], 'CBD091', 'CBD090', df[i].columns)
    replace_col_name(df[i], 'CBD111', 'CBD110', df[i].columns)
    replace_col_name(df[i], 'CBD121', 'CBD120', df[i].columns)
    replace_col_name(df[i], 'CBD131', 'CBD130', df[i].columns)

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({777777: -10, 999999: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  # Sum all columns, and drop individual columns
  merged_df['total_money_spent'] = merged_df[numerical_cols].sum(axis=1)
  merged_df.drop(merged_df.columns.difference(['SEQN', 'total_money_spent']), 1, inplace=True)

  return merged_df


def preprocess_oral_health(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'OHQ835', 'OHQ845', 'OHQ850', 'OHQ860', 'OHQ870']
  numerical_cols = ['OHQ870']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    for column in df[i].columns:
      if column in numerical_cols:
        df[i].replace({77: -10, 99: -10}, inplace=True)
      else:
        df[i].replace({7: -10, 9: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_body_measures(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BMXHIP']
  numerical_cols = ['BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BMXHIP']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    for column in df[i].columns:
      if column in numerical_cols:
        df[i][column] = df[i][column].astype(float)
      else:
        df[i][column] = df[i][column].astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({1: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, float)

  return merged_df


def preprocess_alcohol_consumption(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'ALQ120Q', 'ALQ140Q']
  numerical_cols = ['ALQ120Q', 'ALQ140Q']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int
    df[i] = df[i].astype(int)

    # Convert dont know, refused to NaN
    df[i].replace({777: -10, 999: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_income(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'IND235', 'INDFMMPI']
  numerical_cols = ['INDFMMPI']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Find NaN
    df[i].fillna(-10, inplace=True)

    # Convert to int / float
    for column in df[i].columns:
      if column!='INDFMMPI':
        df[i][column] = df[i][column].astype(int)
      

    # Convert dont know, refused to NaN
    df[i]['IND235'].replace({77: -10, 99: -10}, inplace=True)

    # Convert -10 to NaN
    df[i].replace(-10, np.NaN, inplace=True)

    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, float)

  return merged_df


def preprocess_household_people_count(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'DMDHHSIZ']
  numerical_cols = ['DMDHHSIZ']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Merge dataframes from different years
    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, int)

  return merged_df


def preprocess_cholesterol(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'LBXTC']
  numerical_cols = ['LBXTC']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Merge dataframes from different years
    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, float)

  return merged_df


def preprocess_blood_count(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'LBXHGB', 'LBXRBCSI']
  numerical_cols = ['LBXHGB', 'LBXRBCSI']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Merge dataframes from different years
    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, float)

  return merged_df


def preprocess_blood_pressure(df):
  
  merged_df = pd.DataFrame()

  considered_cols = ['SEQN', 'BPXSY1', 'BPXDI1']
  numerical_cols = ['BPXSY1', 'BPXDI1']
  categorical_cols = set(considered_cols) - set(numerical_cols)

  # Pre-process
  for i in range(len(df)):

    # Drop columns which are not be considered
    df[i].drop(df[i].columns.difference(considered_cols), 1, inplace=True)

    # Merge dataframes from different years
    merged_df = pd.concat([merged_df, df[i]], sort=False)

  clean_merged_df(merged_df, categorical_cols, float)

  return merged_df


if __name__ == '__main__':
  
  # Set data input and output paths
  data_upload_path = '/individual_datasets/'
  result_path = '/preprocessed_datasets/'

  print('Reading data..')

  # Get dataframes list
  mental_health_depression_dataframes = upload_files(data_upload_path + 'mental_health_depression')
  early_childhood_dataframes = upload_files(data_upload_path + 'early_childhood')
  food_security_dataframes = upload_files(data_upload_path + 'food_security')
  occupation_dataframes = upload_files(data_upload_path + 'occupation')
  sleep_disorders_dataframes = upload_files(data_upload_path + 'sleep_disorders')
  housing_characteristics_dataframes = upload_files(data_upload_path + 'housing_characteristics')
  sexual_behavior_dataframes = upload_files(data_upload_path + 'sexual_behavior')
  diet_behavior_and_nutrition_dataframes = upload_files(data_upload_path + 'diet_behavior_and_nutrition')
  hospital_utilization_and_access_to_care_dataframes = upload_files(data_upload_path + 'hospital_utilization_and_access_to_care')
  consumer_behavior_dataframes = upload_files(data_upload_path + 'consumer_behavior')
  oral_health_dataframes = upload_files(data_upload_path + 'oral_health')
  body_measures_dataframes = upload_files(data_upload_path + 'body_measures')
  alcohol_consumption_dataframes = upload_files(data_upload_path + 'alcohol_consumption')
  income_dataframes = upload_files(data_upload_path + 'income')
  household_people_count_dataframes = upload_files(data_upload_path + 'household_people_count')
  cholesterol_dataframes = upload_files(data_upload_path + 'cholesterol')
  blood_count_dataframes = upload_files(data_upload_path + 'blood_count')
  blood_pressure_dataframes = upload_files(data_upload_path + 'blood_pressure')

  print('Processing and cleaning data..')

  # Preprocess each variable separately based on requirement and export cleaned dataset
  preprocess_mental_health_depression(mental_health_depression_dataframes).to_csv(result_path+'mental_health_depression.csv', index=False)
  preprocess_early_childhood(early_childhood_dataframes).to_csv(result_path+'early_childhood.csv', index=False)
  preprocess_food_security(food_security_dataframes).to_csv(result_path+'food_security.csv', index=False)
  preprocess_occupation(occupation_dataframes).to_csv(result_path+'occupation.csv', index=False)
  preprocess_sleep_disorders(sleep_disorders_dataframes).to_csv(result_path+'sleep_disorders.csv', index=False)
  preprocess_housing_characteristics(housing_characteristics_dataframes).to_csv(result_path+'housing_characteristics.csv', index=False)
  preprocess_sexual_behavior(sexual_behavior_dataframes).to_csv(result_path+'sexual_behavior.csv', index=False)
  preprocess_diet_behavior_and_nutrition(diet_behavior_and_nutrition_dataframes).to_csv(result_path+'diet_behavior_and_nutrition.csv', index=False)
  preprocess_hospital_utilization_and_access_care(hospital_utilization_and_access_to_care_dataframes).to_csv(result_path+'hospital_utilization_and_access_care.csv', index=False)
  preprocess_consumer_behavior(consumer_behavior_dataframes).to_csv(result_path+'consumer_behavior.csv', index=False)
  preprocess_oral_health(oral_health_dataframes).to_csv(result_path+'oral_health.csv', index=False)
  preprocess_body_measures(body_measures_dataframes).to_csv(result_path+'body_measures.csv', index=False)
  preprocess_alcohol_consumption(alcohol_consumption_dataframes).to_csv(result_path+'alcohol_consumption.csv', index=False)
  preprocess_income(income_dataframes).to_csv(result_path+'income.csv', index=False)
  preprocess_household_people_count(household_people_count_dataframes).to_csv(result_path+'household_people_count.csv', index=False)
  preprocess_cholesterol(cholesterol_dataframes).to_csv(result_path+'cholesterol.csv', index=False)
  preprocess_blood_count(blood_count_dataframes).to_csv(result_path+'blood_count.csv', index=False)
  preprocess_blood_pressure(blood_pressure_dataframes).to_csv(result_path+'blood_pressure.csv', index=False)

  print('Cleaning complete!')
  print('Processed datasets saved to: ', result_path)