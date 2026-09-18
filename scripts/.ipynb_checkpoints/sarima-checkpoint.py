import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.signal import periodogram
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from pathlib import Path
import json
import seaborn as sns
from scipy.stats import norm


# Function for Augmented Dickey-Fuller Test
def adf_test(data):
    col = data.columns

    for j,name in enumerate(col):
        adf_test = adfuller(data[name], autolag = 'AIC')
        results = pd.Series(adf_test[0:2], index = ['ADF Test Statistic','P-Value'])
        print(name)
        print('Augmented Dickey-Fuller Test Results:')
        print(results)
        if (adf_test[1] < 0.05):
            isStationary = True
        else:
            isStationary = False
        print("Is the time series stationary?  " + str(isStationary))
        print("---------------------------------------------------------")


# Function for KPSS stationary test
def kpss_test(serie):
    import warnings
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    for col in serie.columns:
        statistic, p_value, n_lags, critical_values = kpss(serie[col])
        # Format Output
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critial Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')
        print(f'Result: The {col} series is {"not " if p_value < 0.05 else ""}stationary')
        print("---------------------------------------------------------")


# To find all possibles models for a SARIMA configuration
def all_models_sarima(data, p_max, q_max, d_min = 0, d_max = 1, p_min = 1, P_max = 2, Q_max = 2, D = 1, s = 96):
    colonnes = ["Model","Convergence", "AIC", "BIC", "HQIC", "Log-Likelihood", "MSE", 
                    "MAE", "p-value AR(p)", "p-value MA(q)", "p-value S-AR(P)", 
                    "p-value S-MA(Q)", r"p-value $\sigma^2$ "]
    df_results = pd.DataFrame(columns = colonnes)
    
    compteur = 0
    
    for P in range(P_max+1):
        for Q in range(Q_max+1):
            for d in range(d_min, d_max+1):
                for p in range(p_min, p_max+1):
                    for q in range(q_max+1):
                        indice = "("+str(p)+","+str(d)+","+str(q)+")-("+str(P)+","+str(D)+","+str(Q)+","+str(s)+")"
                        
                        model = sm.tsa.statespace.SARIMAX(data,
                                          order = (p,d,q),
                                          seasonal_order=(P, D, Q, s),
                                          enforce_stationarity = True,
                                          enforce_invertibility = True)
                        results = model.fit()
                        
                        if q > 0:
                            p_ma = results.pvalues["ma.L"+str(q)]
                        else:
                            p_ma = 0.0
                        if P > 0:
                            p_arS = results.pvalues["ar.S.L96"]
                        else:
                            p_arS = 0.0
                        if Q > 0:
                            p_maS = results.pvalues["ma.S.L96"]
                        else:
                            p_maS = 0.0
    
                        resultats = [indice, results.mle_retvals["converged"], results.aic, results.bic, 
                                     results.hqic, results.llf, results.mse, results.mae, results.pvalues["ar.L"+str(p)], 
                                     p_ma, p_arS, p_maS,results.pvalues["sigma2"]]
                        
                        del model
                        del results
                        
                        df_results.loc[compteur] = resultats
                        compteur = compteur + 1
                        print(compteur)
    
    df_results = df_results.set_index("Modèle")
    
    return df_results


# Function used to filtered all training models
def filtered_models(Condu_train, p_value = 0.05):
    new_data = Condu_train[(Condu_train["Convergence"] == True) & (Condu_train["p-value AR(p)"] < p_value) & (Condu_train["p-value MA(q)"] < p_value) & (Condu_train["p-value S-AR(P)"] < p_value) & (Condu_train["p-value S-MA(Q)"] < p_value)]
    return new_data


# Function used to find the best SARIMA / SARIMAX model
def choice(Temp_train1_results):
    print("AIC choice:", end = ' ')
    print(Temp_train1_results.where(Temp_train1_results["AIC"] == Temp_train1_results["AIC"].min()).dropna().index)
    print("BIC choice:", end = ' ')
    print(Temp_train1_results.where(Temp_train1_results["BIC"] == Temp_train1_results["BIC"].min()).dropna().index)
    print("HQIC choice:", end = ' ')
    print(Temp_train1_results.where(Temp_train1_results["HQIC"] == Temp_train1_results["HQIC"].min()).dropna().index)
    print("Log-Vraisemblance choice:", end = ' ')
    print(Temp_train1_results.where(Temp_train1_results["Log-Vraisemblance"] == Temp_train1_results["Log-Vraisemblance"].max()).dropna().index)


# To plot ACF and PACF for model residuals
def acf_pacf_resid(results_model, col = None):
    lag = 300
    lag1 = 300
    fig = plt.figure(figsize = (18, 5))
    layout = (1, 3)
    
    ts1_ax = plt.subplot2grid(layout, (0,0))
    ts2_ax = plt.subplot2grid(layout, (0,1))
    ts3_ax = plt.subplot2grid(layout, (0,2))
    results_model.resid.plot(ax = ts1_ax)
    smt.graphics.plot_acf(results_model.resid, lags = lag, ax = ts2_ax)
    ts2_ax.set_title(f'Simple autocorrelogram of residuals of {col}')
    smt.graphics.plot_pacf(results_model.resid, lags = lag1, ax = ts3_ax)
    ts3_ax.set_title(f'Partial autocorrelogram of residuals of {col}')
    plt.tight_layout()
    plt.show()


# For residuals normality test
def hist_norm(results_t1, name):
    plt.figure()
    
    # Histogram of residuals with KDE
    sns.histplot(results_t1.resid, kde = True, bins = 50, stat = 'density', color = 'skyblue', edgecolor = 'white', alpha = 0.7, label = 'Residuals distribution')
    
    # Normal density curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, loc = np.mean(results_t1.resid), scale = np.std(results_t1.resid))
    plt.plot(x, p, 'r-', linewidth = 2.5, label = 'Normal distribution')
    plt.title(f'{name} Residuals vs Normal Distribution')
    plt.xlabel('Residual values')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
# To compute predictions errors
def errors(df_test, pred):
    mse = np.mean((df_test.values - pred)**2)
    mae = np.mean(np.abs(df_test.values - pred))
    mape = 100 * np.mean(np.abs((df_test.values - pred) / df_test.values))
    print("MSE test : "+str(mse))
    print("MAE test : "+str(mae))
    print("MAPE test : "+str(mape)+" %")


# To compare by drawing predictions vs test data
def plot_predictions(df_test, predictions, col):
    plt.plot(df_test[col], label = 'Test Data')
    plt.plot(predictions.predicted_mean, label = 'Predictions', color = 'red')
    plt.legend()
    plt.title(f'Real data vs. model predictions for {col}')
    plt.tight_layout()
    plt.show()

    # Errors
    errors(df_test[col], predictions.predicted_mean)


# Saves the estimated parameters of a SARIMA model
def save_sarima_params(results, filepath):

    model = results.model

    info = {
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "enforce_stationarity": model.enforce_stationarity,
        "enforce_invertibility": model.enforce_invertibility,
        "params": results.params.to_dict()
    }

    with open(filepath, "w") as f:
        json.dump(info, f, indent = 4)


# Reconstructs a SARIMA model using previously estimated parameters
def load_sarima_params(series, filepath):
   
    with open(filepath, "r") as f:
        info = json.load(f)

    model = sm.tsa.statespace.SARIMAX(
        series,
        order = tuple(info["order"]),
        seasonal_order = tuple(info["seasonal_order"]),
        enforce_stationarity = info["enforce_stationarity"],
        enforce_invertibility = info["enforce_invertibility"]
    )

    params = pd.Series(info["params"])

    results = model.filter(params)

    return results



## Function used to display plots in "main.qmd"

# For normality test
def hist_norm_pdf_display(results_t1, name, ax):
    # Histogram of residuals with KDE
    sns.histplot(results_t1.resid, kde = True, bins = 50, stat = 'density',
                 color = 'skyblue', edgecolor = 'white', alpha = 0.7,
                 label = 'Residuals distribution', ax = ax)

    # Normal density curve
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, loc = np.mean(results_t1.resid), scale = np.std(results_t1.resid))
    ax.plot(x, p, 'r-', linewidth = 2.5, label = 'Normal distribution')

    ax.set_title(f'{name} Residuals vs Normal Distribution')
    ax.set_xlabel('Residual values')
    ax.set_ylabel('Density')
    ax.legend()


# For residuals acf and pacf displaying in "main.qmd"
def acf_pacf_resid_pdf_display(results_model, col, axes):
    lag = 300
    lag1 = 300

    ts1_ax = axes[0]
    ts2_ax = axes[1]
    ts3_ax = axes[2]

    results_model.resid.plot(ax = ts1_ax)
    smt.graphics.plot_acf(results_model.resid, lags = lag, ax = ts2_ax)
    smt.graphics.plot_pacf(results_model.resid, lags = lag1, ax = ts3_ax)

    ts1_ax.set_title(f'{col} - Residuals')
    ts2_ax.set_title(f'ACF of residuals of {col}')
    ts3_ax.set_title(f'PACF of residuals of {col}')

    ts2_ax.set_ylim(-0.7, 1.1)
    ts3_ax.set_ylim(-0.7, 1.1)