# Data Scientist Challenge

Assume you are working in a credit card company that is expecting a recession in the near future. The company would like to maintain a certain percentage of credit card defaults out of all the users. You will work on a dataset for the set of credit card users and predict the probability of credit card defaults.

## Dataset

This dataset contains information on default payments, demographic factors, credit data, history of payments, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

## Part 1: Exploratory Data Analysis (EDA) and Extract-Transform-Load (ETL)

- Explore the whole dataset and find any interesting insights
- Evaluate the data quality and figure out whether outliners exist
- Select the strongest predictors and perform feature engineering







## Part 2: Prediction Modeling

- Train a model to predict whether a user will default or not
- The model is not expected to be perfectly accurate and precise, but instead try to build a simple one involving fewer features based on the limited time frame
- Evaluate the performance of model

## Requirements

- [ x ] Jupyter Notebook for Exploratory Data Analysis (EDA) and Extract-Transform-Load (ETL):
	- [v0\_Baseline\_FeatureExploration/00\_ExploratoryAnalysis\_ETL.ipynb](v0_Baseline_FeatureExploration/00_ExploratoryAnalysis_ETL.ipynb): Contains the first pass of EDA and ETL, dropping features that would create subcategories with few items, have multiple nan values or undescribed values. As well a a new feature BAL\_AMT = BILL\_AMT - PAY\_AMT. Moreover, it was found that there are outliers in the numerical features and that also the categories to predict are umbalanced. Finally, the output of this is the data for training in its original format and one_hot_encoded in addition to the encoder.
	- [00\_ExploratoryAnalysis\_ETL](00_ExploratoryAnalysis_ETL.ipynb): Second iteration of EDA and ETL. Based on the findings of the first model training, some the categories with few elements were dropped, e.g. education = others and marriage = others. Exporting the data for training in its original format and one_hot_encoded in addition to the encoder.

- [ x ] Jupyter Notebook for Prediction Modeling:
	- [v0\_Baseline\_FeatureExploration/01\_Model.ipynb](v0_Baseline_FeatureExploration/01_Model.ipynb): First pass of the XGBoost model training. I split the data in train/test using a 80/20 split, stratified according to if the customer defaults or not and tuned using GridSearchCV. The training results show that, as expected, the categories  with few elements, e.g. education = others and marriage = others, impact the model negatively and as well have low predicting power according to SHAP values, hence in the next revision they are discarded.
	- [01\_Model.ipynb](v0_Baseline_FeatureExploration/01_Model.ipynb): Second iteration of the XGBoost model training. Following the same procedure that in the first pass the training results show that, in general there's an increase in AUC by a couple percentages for most categories or lowering the spread in its value. Therefore, confirming that the categories with few elements, e.g. education = others and marriage = others, impacted the model negatively. Moreover, the category education = unknown might be also considered to be discarded, but this is left for a further iteration of the model.

- [ x ] Implementation in Python of a simple data model that can be trained and predict whether a user will default or not, let us know if you prefer something other than Python

The jupyter notebooks cover this point. Since this is exploratory work and the model could be further refined, I haven't packaged the etl/training into python executables or flask/docker files.

## Optional

- [ ] Draft a plan with multiple phases of work on how to tackle the problem, let us know what are the tradeoffs and considerations
- [ ] Let us know what improvements can be made if we have more time and resources

## Deliverable

- [ x ] Jupyter Notebooks (Hosted on Github, Bitbucket, etc.)
- [ x ] Documentation on how to run the them

## Grading and Submission Requirements

- Graded based on thought process and completeness of modeling work flow. The model will only be expected to perform reasonably accurate and precise within limited time.

## Data Dictionary

| Data | Definition | Value(s) |
| --- | --- | --- |
| ID | ID of each client | |
| LIMIT_BAL | Amount of given credit in NT dollars (includes individual and family/supplementary credit | |
| SEX | Gender | (1=male, 2=female) |
| EDUCATION | Education | (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) |
| MARRIAGE | Marital status | (1=married, 2=single, 3=others) |
| AGE | Age in years | |
| PAY_0 | Repayment status in September, 2005 | (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) |
| PAY_2 | Repayment status in August, 2005 | (scale same as above) |
| PAY_3 | Repayment status in July, 2005 | (scale same as above) |
| PAY_4 | Repayment status in June, 2005 | (scale same as above) |
| PAY_5 | Repayment status in May, 2005 | (scale same as above) |
| PAY_6 | Repayment status in April, 2005 | (scale same as above) |
| BILL_AMT1 | Amount of bill statement in September, 2005 (NT dollar) | |
| BILL_AMT2 | Amount of bill statement in August, 2005 (NT dollar) | |
| BILL_AMT3 | Amount of bill statement in July, 2005 (NT dollar) | |
| BILL_AMT4 | Amount of bill statement in June, 2005 (NT dollar) | |
| BILL_AMT5 | Amount of bill statement in May, 2005 (NT dollar) | |
| BILL_AMT6 | Amount of bill statement in April, 2005 (NT dollar) | |
| PAY_AMT1 | Amount of previous payment in September, 2005 (NT dollar) | |
| PAY_AMT2 | Amount of previous payment in August, 2005 (NT dollar) | |
| PAY_AMT3 | Amount of previous payment in July, 2005 (NT dollar) | |
| PAY_AMT4 | Amount of previous payment in June, 2005 (NT dollar) | |
| PAY_AMT5 | Amount of previous payment in May, 2005 (NT dollar) | |
| PAY_AMT6 | Amount of previous payment in April, 2005 (NT dollar) | |
| default.payment.next.month | Default payment | (1=yes, 0=no) | |
