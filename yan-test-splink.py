#%%
import pandas as pd 
df_l = pd.read_parquet("./data/fake_df_l.parquet")
df_r = pd.read_parquet("./data/fake_df_r.parquet")
df_l.head(2)
#%%
from splink.duckdb.duckdb_linker import DuckDBLinker
import splink.duckdb.duckdb_comparison_library as cl
# import splink.duckdb.duckdb_comparison_template_library as ctl
#%%

settings = {
    "link_type": "link_only",
    "blocking_rules_to_generate_predictions": [
        "l.first_name = r.first_name",
        "l.surname = r.surname",
    ],
    "comparisons": [
        # ctl.name_comparison("first_name",),
        # ctl.name_comparison("surname"),
        # ctl.date_comparison("dob", cast_strings_to_date=True),
        # cl.name_comparison("first_name",),
        # cl.name_comparison("surname"),
        # cl.date_comparison("dob", cast_strings_to_date=True),
        cl.exact_match("city", term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("email"),
    ],       
}
#%%
linker = DuckDBLinker([df_l, df_r], settings, input_table_aliases=["df_left", "df_right"])
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)
#%%
linker.estimate_u_using_random_sampling(target_rows=1e6)
#%%
session_dob = linker.estimate_parameters_using_expectation_maximisation("l.dob = r.dob")
session_email = linker.estimate_parameters_using_expectation_maximisation("l.email = r.email")
session_first_name = linker.estimate_parameters_using_expectation_maximisation("l.first_name = r.first_name")
#%%
results = linker.predict(threshold_match_probability=0.9)
#%%
results.as_pandas_dataframe(limit=5)