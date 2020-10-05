# Imports
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import statsmodels.api as sm  
from statsmodels.stats.mediation import Mediation
import patsy

def getData():

  # Read all the csv files into a pandas dataframe
  df_health = pd.read_csv('hospital_utilization_and_access_care.csv')
  df_income = pd.read_csv('income.csv')
  df_dep = pd.read_csv('mental_health_depression.csv')
  df_food = pd.read_csv('food_security.csv')
  df_alcohol = pd.read_csv('alcohol_consumption.csv')

	# Merge Dataframes into one single dataframe on the column SEQN
  df_temp = pd.merge(df_health, df_dep, on='SEQN')
  df_temp = pd.merge(df_temp, df_income, on='SEQN')
  df_temp = pd.merge(df_temp, df_food, on='SEQN')
  df = pd.merge(df_temp, df_alcohol, on='SEQN')
  
  return df



def mediation_analysis(data):

	# A regression model for effect of Independent variable along with mediator on Dependent variable
	outcome = np.asarray(data["DPQ020"])
	outcome_exog = patsy.dmatrix("FSDHH + INDFMMPI + HUQ090", data, return_type='dataframe')
	probit = sm.families.links.probit
	outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit()))

	# A regression model for effect of Independent variable on the mediator
	mediator = np.asarray(data['HUQ090'])
	mediator_exog = patsy.dmatrix('FSDHH + INDFMMPI', data, return_type='dataframe')
	mediator_model = sm.OLS(mediator, mediator_exog)

	tx_pos = [outcome_exog.columns.tolist().index("FSDHH"), mediator_exog.columns.tolist().index("FSDHH")]
	med_pos = outcome_exog.columns.tolist().index("HUQ090")

	# Mediation Analysis using Statistical Analysis Mediation package
	med = Mediation(outcome_model, mediator_model, tx_pos, med_pos).fit()
	output = np.round(med.summary(), decimals=3)
	return output

if __name__ == "__main__":

	# Get data from different datasets
	data = getData()

	# Pass the data for a Mediation Analysis
	summary = mediation_analysis(data)
	print(summary)