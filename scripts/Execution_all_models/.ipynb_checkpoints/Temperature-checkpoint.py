import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sarima import *
from preprocessing import SAV_preprocessing

# Data importation
df_sav = SAV_preprocessing()

# Construction of the Training Set and the Test Set
begin_test_index = df_sav.index.get_loc("2023-12-07 20:00:00")

df_train = df_sav.iloc[begin_test_index - 96*10 : begin_test_index]
df_train.index = pd.DatetimeIndex(df_train.index.values, freq = df_train.index.inferred_freq)

# All possible models
df_results_temp = all_models_sarima(data = df_train["Temperature SAV"], d_min = 1, p_max = 3, q_max = 2)
df_results_temp.to_csv(Path(__file__).resolve().parent.parent / "All_models" / "Temp_sarima.csv")