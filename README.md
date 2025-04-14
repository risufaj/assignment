# Census Income Classification Project

This repository contains an **end-to-end data science pipeline** for predicting whether an individual’s annual income is >= 50K or < 50K, using a subset of US Census data. 


The goal is to **classify** individuals into two categories:  
- **\`>= 50K\`**  
- **\`< 50K\`**


**Key Steps**:
1. **EDA**: We uncovered significant skew in capital gains/losses and dividends, many rare occupation categories, and a >10%% minority class of >=\$50K.
2. **Feature Engineering**: 
   - Added binary indicators + log transforms for zero-inflated numeric columns.
   - Merged rare categories into “Other.”
   - One-hot encoded categorical columns.
3. **Modeling**: 
   - Logistic Regression, Random Forest, XGBoost and Heuristic Method
   - Reasoned about how to pick the best model
4. **Final**: 
   - Best model has ~0.85 recall with ~0.37 precision, leading to a strong F1 improvement over simpler methods.


## Structure

Running this project assumes that `census_income_learn.csv`and `census_income_test.csv` are located in this folder. If not, please change `TRAIN_PATH` `TEST_PATH` in `config.py`

We used the following CSV files (provided by the US Census dataset sample):
- `census_income_learn.csv` for training
- `census_income_test.csv` for testing

**Project Files**:
- `requirements.txt`: Dependencies for this project
- `census_income_project.ipynb`: Main notebook 
- `README.md`: This documentation
- Additional scripts or notebooks for **EDA**, **transformations**, **modeling** steps
- Presentation
- MLFlow tracking of the experiments conducted in this task. Please unzip. 


