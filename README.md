# MENTAL HEALTH ANALYSIS - NHANES DATASET

## Team Members:
- Neha Shetty
- Shubhangi Kishore
- Melita Saldanha
- Ronak Khandelwal

## Data:
We have considered 18 questionnaires with data from 1999-2018 from the [NHANES dataset](https://www.cdc.gov/nchs/nhanes/index.htm). The questionnaires are as follows:
1. Mental Health - Depression
2. Early Childhood
3. Food Security
4. Occupation
5. Sleep Disorders
6. Housing Characteristics
7. Sexual Behavior
8. Diet Behavior and Nutrition
9. Hospital Utilization and Access to Care
10. Consumer Behavior
11. Oral Health
12. Body Measures
13. Alcohol Consumption
14. Income
15. Household People Count
16. Cholesterol Level
17. Blood Count
18. Blood Pressure
Data Pre-processing:

We merged the data for all the years and cleaned each questionnaire data individually, based on its specific requirements.   
**Filename:** _data_preprocessing.py_   
**Usage:** `python data_preprocessing.py`   

## Hypothesis Testing:
1. **Finding associations between depression questionnaire data and select features**   
**Description** - For pre-selected general datasets (Questionnaire, Laboratory, Examination) from NHANES we find the correlation to investigate the associations and derive Beta Values by performing regression, We also perform hypothesis testing using two tailed T Tests and calculating the resulting P value. We used a size of the test equal to α = 0.05 level and performed multi-test
correction.   
**Filename** - _correlation_all_features.py_    
**Usage** - `python3 correlation_all_features.py`

2. **Regression Analysis**   
**Description** - Performing Multivariate Linear regression on given datasets with the target column using tensorflow (making use of momentum vector).   
**Filename** - _regression_analysis.py_   
**Usage** - `regression_analysis.py “feature_filename” “target_filename” “target_col”`

## Recommendation System:   
**Frameworks and concepts used:** Spark, HDFS, Similarity and Collaborative filtering   
**Filename:** _Recommendation_system.py_   
**Usage:** `spark-submit Recommendation_system.py <target_column> <space_separated_list_of_csv_files>`

## Mediation Analysis:   
As all the features do not directly have an effect on mental health, we performed mediation analysis by introducing a mediator to check the effect of an independent variable on mental health features.   
**Filename:** _mediationAnalysis.py_   
**Usage:** `python mediationAnalysis.py`
