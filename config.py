"""
Global configuration for the Census Income project.
Stores lists of columns and other constants.
"""

# Categorical and numerical columns in the dataset
# mapping of index to actual column names

column_index_to_name = {
    0:  "age",
    1:  "class_of_worker",
    2:  "detailed_industry_recode",
    3:  "detailed_occupation_recode",
    4:  "education",
    5:  "wage_per_hour",
    6:  "unroll_in_edu_inst_last_wk",
    7:  "marital_stat",
    8:  "major_industry_code",
    9:  "major_occupation_code",
    10: "race",
    11: "hispanic_origin",
    12: "sex",
    13: "member_of_a_labor_union",
    14: "reason_for_unemployment",
    15: "full_or_part_time_employment_stat",
    16: "capital_gains",
    17: "capital_losses",
    18: "dividends_from_stocks",
    19: "tax_filer_stat",
    20: "region_of_previous_residence",
    21: "state_of_previous_residence",
    22: "detailed_household_and_family_stat",
    23: "detailed_household_summary_in_household",
    24: "migration_code_change_in_msa",
    25: "migration_code_change_in_reg",
    26: "migration_code_move_within_reg",
    27: "live_in_this_house_1_year_ago",
    28: "migration_prev_res_in_sunbelt",
    29: "num_persons_worked_for_employer",
    30: "family_members_under_18",
    31: "country_of_birth_father",
    32: "country_of_birth_mother",
    33: "country_of_birth_self",
    34: "citizenship",
    35: "own_business_or_self_employed",
    36: "fill_inc_questionnaire_for_veterans_admin",
    37: "veterans_benefits",
    38: "weeks_worked_in_year",
    39: "year",
    40: "income"  # binary target: "- 50000" or "50000+"
}

numerical_columns = [
    'age',
    'wage_per_hour', 
    'capital_gains',
    'capital_losses',
    'dividends_from_stocks',
    'num_persons_worked_for_employer',
    'weeks_worked_in_year'
]

categorical_columns = [
    'class_of_worker',
    'detailed_industry_recode',
    'detailed_occupation_recode',
    'education',
    'unroll_in_edu_inst_last_wk',
    'marital_stat',
    'major_industry_code',
    'major_occupation_code',
    'race',
    'hispanic_origin',
    'sex',
    'member_of_a_labor_union',
    'reason_for_unemployment',
    'full_or_part_time_employment_stat',
    'tax_filer_stat',
    'region_of_previous_residence',
    'state_of_previous_residence',
    'detailed_household_and_family_stat',
    'detailed_household_summary_in_household',
    'migration_code_change_in_msa',
    'migration_code_change_in_reg',
    'migration_code_move_within_reg',
    'live_in_this_house_1_year_ago',
    'migration_prev_res_in_sunbelt',
    'family_members_under_18',
    'country_of_birth_father',
    'country_of_birth_mother',
    'country_of_birth_self',
    'citizenship',
    'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veterans_admin',
    'veterans_benefits',
    'year'
]

# Ratio for grouping rare categories as 'Other'
RARE_CATEGORY_RATIO = 0.01

# Cross-validation folds
CV_SPLITS = 5

# Filenames for training / test data
TRAIN_PATH = "census_income_learn.csv"
TEST_PATH  = "census_income_test.csv"
