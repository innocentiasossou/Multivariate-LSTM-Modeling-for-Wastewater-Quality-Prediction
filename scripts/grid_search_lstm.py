from lstm import *
from sarima import *
from preprocessing import *
import tempfile
import keras_tuner as kt
import json

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR


# Data from Seine aval (SAV)
df_sav = SAV_preprocessing()

# Rainfall data
df_pluvio = rainfall_preprocessing()

# Data from Seine centre (SEC)
df_sec = SEC_preprocessing()

# SEC flow
debit_sec = SEC_flow_preprocessing()


### Construction of the training and validation datasets
# Training set
df_train = X_construction(start = '2022-11-25 07:00:00',
                          end = '2023-11-24 06:45:00',
                          df_sav = df_sav,
                          debit_sec = debit_sec,
                          df_sec = df_sec,
                          df_pluvio = df_pluvio) # 1 year

# Validation sets
df_val1 = X_construction(start = '2023-11-24 07:00:00',
                        end = '2023-12-23 06:45:00',
                        df_sav = df_sav,
                        debit_sec = debit_sec,
                        df_sec = df_sec,
                        df_pluvio = df_pluvio) # 1 month in winter

df_val2 = X_construction(start = '2024-03-15 00:00:00',
                    end = '2024-04-14 23:45:00',
                    df_sav = df_sav,
                    debit_sec = debit_sec,
                    df_sec = df_sec,
                    df_pluvio = df_pluvio) # 1 month in spring

df_val3 = X_construction(start = '2024-07-01 00:00:00',
                         end = '2024-07-31 23:45:00', 
                        df_sav = df_sav,
                        debit_sec = debit_sec,
                        df_sec = df_sec,
                        df_pluvio = df_pluvio) # 1 month in summer

df_val4 = X_construction(start = '2024-09-01 00:00:00',
                        end = '2024-09-30 23:45:00',
                        df_sav = df_sav,
                        debit_sec = debit_sec,
                        df_sec = df_sec,
                        df_pluvio = df_pluvio) # 1 month in fall



### Data transformation
# a and b represent the endpoints of the interval to which the data will belong
scaler = MinMaxScaler(feature_range = (a, b)) # a, b = -1, 1

df_train = pd.DataFrame(scaler.fit_transform(df_train),
                               columns = df_train.columns,
                               index = df_train.index)

df_val1 = pd.DataFrame(scaler.transform(df_val1),
                      columns = df_val1.columns,
                      index = df_val1.index)

df_val2 = pd.DataFrame(scaler.transform(df_val2),
                      columns = df_val2.columns,
                      index = df_val2.index)

df_val3 = pd.DataFrame(scaler.transform(df_val3),
                      columns = df_val3.columns,
                      index = df_val3.index)

df_val4 = pd.DataFrame(scaler.transform(df_val4),
                      columns = df_val4.columns,
                      index = df_val4.index)

df_val_list = [df_val1, df_val2, df_val3, df_val4]


# Dataset for LSTM
train, valid = multi_train_valid(df_train, df_val_list)


### Grid search
# Tuner creation

multi_dir = tempfile.mkdtemp()

tuner_multi = kt.BayesianOptimization(
    build_grid_search_model_multi,
    objective = 'val_loss',
    max_trials = 100,
    executions_per_trial = 1,
    directory = multi_dir, 
    project_name = 'pred' 
)

# Hyperparameter search
tuner_multi.search(
    train,
    epochs = 30,
    validation_data = valid,
    batch_size = 128,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)
    ]
)

# Saving the best hyperparameters
best_hp = tuner_multi.get_best_hyperparameters(1)[0]

with open("All_models/best_lstm_hyperparameters.json", "w") as f:
    json.dump(best_hp.values, f, indent = 4)