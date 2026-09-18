from Useful_functions_sarimax import*

"""df_sav = pd.read_csv("HF_SAV_Valides_2022_a_2024.csv", index_col=0, parse_dates=True) # permet de convertir l’index en objets datetime plutôt que de le lire comme un str.
df_sav.index = pd.DatetimeIndex(df_sav.index.values, freq = df_sav.index.inferred_freq)
del df_sav["Flow A1"]
del df_sav["Flow A2"]
del df_sav["Flow A3i"]
del df_sav["Flow A3p"]
del df_sav["Flow A4"]"""

df_sav = pd.read_csv("HF_SAV_2022_a_2024_regularisee.csv", index_col = 0, parse_dates = True)
df_sav.index = pd.DatetimeIndex(df_sav.index.values, freq = df_sav.index.inferred_freq)

begin_test_index_2 = df_sav.index.get_loc("2023-12-07 20:00:00")

df_train_2 = df_sav.iloc[begin_test_index_2 - 96*10 : begin_test_index_2]
df_train_2.index = pd.DatetimeIndex(df_train_2.index.values, freq = df_train_2.index.inferred_freq)

df_results_cond = all_models_sarima(data = df_train_2["Conductivity SAV"], d_min = 1, p_max = 1, q_max = 2)
df_results_cond.to_csv("Condu_sarima_train2.csv")