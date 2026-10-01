import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

h = 96 # value of the forecast horizon 
a, b = -1, 1


# Construction of exogenous variables
def exog_construction(start, end, df_sav, debit_sec, df_sec, df_pluvio, decalage = 24):
    """
    Merge "Total flow" from df_train with all columns from debit_sec and df_sec over a specified period.

    decalage: integer, used to shift the flow data by 6 hours.
    """
    
    index1_df_sav = df_sav.index.get_loc(start)
    index2_df_sav = df_sav.index.get_loc(end)
    
    index1_debit_sec = debit_sec.index.get_loc(start)
    index2_debit_sec = debit_sec.index.get_loc(end)
    
    index1_df_pluvio = df_pluvio.index.get_loc(start)
    index2_df_pluvio = df_pluvio.index.get_loc(end)
    
    flow = df_sav[['Total flow']].iloc[index1_df_sav : index2_df_sav + decalage + 1].shift(-decalage).dropna()
    flow.columns = [r'Flow SAV$_{t+24}$']
    
    debit = debit_sec.iloc[index1_debit_sec : index2_debit_sec + decalage + 1].shift(-decalage).dropna()
    debit.columns = [r'Flow SEC$_{t+24}$']
    
    # Rainfall measurements are taken at time t
    pluvio = df_pluvio.loc[start:end]
    
    sec = df_sec.loc[start:end]
    
    fusion = pd.concat([flow, debit, sec, pluvio], axis=1)
     
    return fusion


def X_construction(start, end, df_sav, debit_sec, df_sec, df_pluvio):
    df = pd.DataFrame()
    
    df = pd.concat([df_sav.loc[start:end], exog_construction(start, end, df_sav, debit_sec, df_sec, df_pluvio)], axis = 1)
    
    return df


# Data preparation for multivariate forecasting with exogenous variables across different validation sets

def split_inputs_and_targets_multi(mulvar_series, horizon = h): 
    return mulvar_series[:, :-horizon], mulvar_series[:, -horizon:, 0:4]


def multi_train_valid(df_train, df_val_list, train_batch_size = 128, val_batch_size = 64, seq_length = 960):
    train = tf.keras.utils.timeseries_dataset_from_array(
        df_train.to_numpy(),
        targets = None,
        sequence_length = seq_length + h,
        batch_size = train_batch_size , 
        shuffle = True 
    ).map(split_inputs_and_targets_multi)


    # Validation
    validation_datasets = []

    for df_val in df_val_list:
        val_ds = tf.keras.utils.timeseries_dataset_from_array(
            df_val.to_numpy(),
            targets = None,
            sequence_length = seq_length + h,
            batch_size = val_batch_size,
            shuffle = False
        ).map(split_inputs_and_targets_multi)

        validation_datasets.append(val_ds)

    # Concatenate all validation datasets
    valid = validation_datasets[0]
    for ds in validation_datasets[1:]:
        valid = valid.concatenate(ds)
       
    return train, valid

# Function to create multiple size bases with `seq_length = 960` for use in a multi-day test
def build_multi_test_sets(first_end_date, scaler, df_sav, debit_sec, df_sec, df_pluvio, n_days = 10, seq_length = 960):
    """
    Constructs n_days datasets for multivariate LSTM forecasting from a given start date.
    Each dataset is used to forecast one day and has a length of seq_length.
    
    first_end_date: end date of the first dataset of length seq_length.
    """

    first_end_date = pd.Timestamp(first_end_date)
    step = pd.Timedelta(minutes = 15)  # the data is collected at 15-minute intervals

    tests_multi = []

    for i in range(n_days):
        
        end_date = first_end_date + pd.Timedelta(days = i) # For forecasting day i, the data are truncated one time step before that day.
        hist_start = end_date - step * seq_length + step

        df_for_test = X_construction(start = hist_start,
                                end = end_date,
                                df_sav = df_sav,
                                debit_sec = debit_sec,
                                df_sec = df_sec,
                                df_pluvio = df_pluvio)
        
        df_for_test_scaled = pd.DataFrame(scaler.transform(df_for_test),
                      columns = df_for_test.columns,
                      index = df_for_test.index)

        test_multi = tf.keras.utils.timeseries_dataset_from_array(
            data = df_for_test_scaled.to_numpy(),
            targets = None,
            sequence_length = seq_length,
            batch_size = 1)

        tests_multi.append(test_multi)

    return tests_multi


# Model training
def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs = 100, patience = 5):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor = "val_loss",       # The validation loss is monitored.
        patience = patience,               # Training is stopped if there is no improvement for 5 epochs.
        restore_best_weights = True # The best-performing model is retained.
    )
    
    model.compile(loss = tf.keras.losses.Huber(), 
                  optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), metrics = ["mae"])
    
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                        callbacks=[early_stopping_cb])
    
    valid_loss, valid_mae = model.evaluate(valid_set)
    
    
    train_loss, train_mae = model.evaluate(train_set)
    
    return history, f"Train_mae = {train_mae}, Valid_mae = {valid_mae}"


# This function plots the evolution of the errors across epochs.
def plot_training_history(history):
    plt.figure(figsize = (12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loss'], label = 'Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label = 'Train MAE')
    plt.plot(history.history['val_mae'], label = 'Val MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

# Selection of the best multivariate model
def build_grid_search_model_multi(hp):
    model = Sequential()

    model.add(tf.keras.layers.Input(shape = (960, 12)))
    
    # Number of LSTM layers (1 à 3)
    num_layers = hp.Int('num_layers', 1, 3)
    
    # Adding the first LSTM layer
    model.add(LSTM(
        units = hp.Choice('units_1', [32, 64, 128, 256]), 
        return_sequences = num_layers > 1))
    model.add(Dropout(hp.Float('dropout_1', 0.0, 0.2, step=0.1)))
    
    # Additional LSTM layers
    for i in range(2, num_layers + 1):
        model.add(LSTM(
            units = hp.Choice(f'units_{i}', [32, 64, 128]),
            return_sequences = i < num_layers))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.0, 0.2, step = 0.1)))
    
    # Output layer
    model.add(Dense(h*4))
    model.add(tf.keras.layers.Reshape((h,4)))
    
    # Compilation
    model.compile(
        optimizer = Adam(hp.Float('learning_rate', 1e-5, 1e-2, step = 10, sampling ='log')),
        loss = tf.keras.losses.Huber(),
        metrics = ['mae'])
    
    return model


# Function for computing errors after forecasting
def errors(df, pred):
    mae = np.mean(np.abs(df.values - pred.values))
    mape = 100 * np.mean(np.abs((df.values - pred.values) / df.values))
    
    print("MAE test : "+str(mae))
    print("MAPE test : "+str(mape)+" %")

    return mae, mape
    

# Representation of predictions for comparison
def plot_compare_models(lstm_model, sarima_models, tests, cols, scaler, df_test, df_sav, lag = 96):
    """
    tests : data used to predict with LSTM model
    df_test : is the real/true test data
    """
    ## Simple baseline predictions: observed value h lags earlier
    df_pred_baseline = df_sav[cols].shift(lag)
    df_pred_baseline = df_pred_baseline.loc[df_test.index]
    
    ## LSTM predictions
    predictions = [lstm_model.predict(tests[i]) for i in range(len(tests))] # test shape : (nb_batch, seq_length, nb_features)
    pred_d = [
        ((predictions[i][0] - a) / (b - a)) * (scaler.data_max_[0:4] - scaler.data_min_[0:4]) + scaler.data_min_[0:4]
             for i in range(len(tests))
             ]                         

    # put all the predictions together
    pred_flat = np.concatenate(pred_d, axis = 0)

    df_pred_d = pd.DataFrame(pred_flat, index = df_test.index, columns = cols)
    
    ## Predictions from each univariate SARIMA model
    pred_sarima = {}
    for col in cols:
        pred = sarima_models[col].get_forecast(steps = len(df_test[col]))
        pred_sarima[col] = pred.predicted_mean
    
    ## Representations and error computation
    all_errors = []
    
    for col in cols:
        plt.plot(df_test[col], label = f"True {col}")
        plt.plot(df_pred_baseline[col], label = f"Naive prediction (t-{lag})", color = "orange")
        plt.plot(pred_sarima[col], label = "SARIMA prediction", color = "green")
        plt.plot(df_pred_d[col], label = "LSTM prediction", color = "red")
        # Date format
        ax = plt.gca() 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        
        plt.title(f"Comparison for {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()


        print("\n===== Errors for", col, "=====")
        print("\nNaive model:")
        mae_naive, mape_naive = errors(df_test[col], df_pred_baseline[col])

        print("\nLSTM model:")
        mae_lstm, mape_lstm = errors(df_test[col], df_pred_d[col])
        
        print("\nSARIMA model:")
        mae_sarima, mape_sarima = errors(df_test[col], pred_sarima[col])

        all_errors.append({
        "Variable": col,
        "Persistence_MAE": mae_naive,
        "Persistence_MAPE": mape_naive,
        "SARIMA_MAE": mae_sarima,
        "SARIMA_MAPE": mape_sarima,
        "LSTM_MAE": mae_lstm,
        "LSTM_MAPE": mape_lstm })

    return pd.DataFrame(all_errors)


# Function for computing errors after forecasting---> FOR MAIN.QMD FILE
def errors_pdf_display(df, pred):
    mae = np.mean(np.abs(df.values - pred.values))
    mape = 100 * np.mean(np.abs((df.values - pred.values) / df.values))

    return mae, mape 


# Representation of predictions for comparison ---> FOR MAIN.QMD FILE
def plot_compare_models_pdf_display(lstm_model, sarima_models, tests, cols, scaler, df_test, df_sav, lag = 96):
    """
    tests : data used to predict with LSTM model
    df_test : is the real/true test data
    """
    ## Simple baseline predictions: observed value h lags earlier
    df_pred_baseline = df_sav[cols].shift(lag)
    df_pred_baseline = df_pred_baseline.loc[df_test.index]
    
    ## LSTM predictions
    predictions = [lstm_model.predict(tests[i], verbose = 0) for i in range(len(tests))] # test shape : (nb_batch, seq_length, nb_features)
    pred_d = [
        ((predictions[i][0] - a) / (b - a)) * (scaler.data_max_[0:4] - scaler.data_min_[0:4]) + scaler.data_min_[0:4]
             for i in range(len(tests))
             ]                         

    # put all the predictions together
    pred_flat = np.concatenate(pred_d, axis = 0)

    df_pred_d = pd.DataFrame(pred_flat, index = df_test.index, columns = cols)
    
    ## Predictions from each univariate SARIMA model
    pred_sarima = {}
    for col in cols:
        pred = sarima_models[col].get_forecast(steps = len(df_test[col]))
        pred_sarima[col] = pred.predicted_mean

    ## Representations and error computation
    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        ax.plot(df_test[col], label = f"True {col}")
        ax.plot(df_pred_baseline[col], label = f"Naive prediction (t-{lag})", color = "orange")
        ax.plot(pred_sarima[col], label = "SARIMA prediction", color = "green")
        ax.plot(df_pred_d[col], label = "LSTM prediction", color = "red")
        # Date format 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        
        ax.set_title(f"Comparison for {col}")
        ax.legend()
    plt.tight_layout()
    plt.show()
        
    # Errors
    all_errors = []

    for col in cols:
        # Naive model
        mae_naive, mape_naive = errors_pdf_display(df_test[col], df_pred_baseline[col])

        # LSTM model
        mae_lstm, mape_lstm = errors_pdf_display(df_test[col], df_pred_d[col])
        
       # SARIMA model
        mae_sarima, mape_sarima = errors_pdf_display(df_test[col], pred_sarima[col])

        all_errors.append({
        "Variable": col,
        "Persistence_MAE": mae_naive,
        "Persistence_MAPE": mape_naive,
        "SARIMA_MAE": mae_sarima,
        "SARIMA_MAPE": mape_sarima,
        "LSTM_MAE": mae_lstm,
        "LSTM_MAPE": mape_lstm })

    return pd.DataFrame(all_errors)


# Code to directly insert the calculated errors into the table in the “main.qmd” file
def create_quarto_table(results, horizon = 96):

    variables = [
        "Conductivity SAV",
        "pH SAV",
        "Temperature SAV",
        "TSS SAV"
    ]

    models = ["Persistence", "SARIMA", "LSTM"]

    rows = []

    # Find the MAPE ranking for each variable
    mape_ranking = {}

    for var in variables:
        values = {
            model: results.loc[
                results["Variable"] == var,
                f"{model}_MAPE"
            ].iloc[0]
            for model in models
        }

        sorted_models = sorted(values, key = values.get)

        mape_ranking[var] = {sorted_models[0]: "*", sorted_models[1]: "**"}

    for model in models:
        # MAE line
        mae_cells = []

        for var in variables:
            value = results.loc[
                results["Variable"] == var,
                f"{model}_MAE"
            ].iloc[0]

            mae_cells.append(f"{value:.3f}")

        rows.append(
            f"| **{model}** | MAE | "
            + " | ".join(mae_cells)
            + " |"
        )

        # MAPE line
        mape_cells = []

        for var in variables:
            value = results.loc[
                results["Variable"] == var,
                f"{model}_MAPE"
            ].iloc[0]

            if model in mape_ranking[var]:
                symbol = mape_ranking[var][model]
                cell = f"${value:.3f}^{{{symbol}}}$"
            else:
                cell = f"{value:.3f}"

            mape_cells.append(cell)

        rows.append(
            f"| | MAPE (\\%) | "
            + " | ".join(mape_cells)
            + " |"
        )

    table = f"""
: Forecasting performance (MAE and MAPE) of all models for a forecasting horizon of $h={horizon}$. The symbol * indicates the best performance (lowest error) and ** indicates the second best performance for each metric within each row. {{#tbl-results}}

| Model       | Metric     | pH SAV |Temperature SAV|Conductivity SAV| TSS SAV |
|:------------|:-----------|:------:|:-------------:|:--------------:|:--------:|
{chr(10).join(rows)}
"""

    return table
