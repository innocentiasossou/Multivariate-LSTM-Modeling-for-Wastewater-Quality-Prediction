import pandas as pd
import numpy as np
from pathlib import Path

# SAV = Seine aval and SEC = Seine centre

def rainfall_preprocessing():
    """ Validates rainfall data and returns the data in DataFrame format.
    
        The raw data is originally collected at one-hour intervals.
    """
    # Importing Raw Data
    rainfall_path = Path(__file__).resolve().parent.parent / "Raw_Data" / "Rainfall_Data" / "ExportBassin2022-2025.xlsx"
    df_pluvio = pd.read_excel(rainfall_path, header = 2, index_col = 0)
    
    # Converting indexes to datetime format
    df_pluvio.index = pd.to_datetime(df_pluvio.index, format = "%d/%m/%Y %H:%M")
    
    # Conversion of Cumulative Values to Hourly Increments
    df_pluvio["Pluvio SAV"] = df_pluvio["SAV"].diff().fillna(0)
    df_pluvio["Pluvio SEC"] = df_pluvio["SEC"].diff().fillna(0)
    del df_pluvio["SAV"]
    del df_pluvio["SEC"]
    
    # Resample in 15 minutes
    df_pluvio_resample = df_pluvio.resample('15Min').ffill() / 4
    
    df_pluvio_resample.index.name = None

    return df_pluvio_resample


def SEC_flow_preprocessing():
    """ Validates "Seine centre" flow data and returns the data in DataFrame format.

        The raw data is originally collected at 5-minute intervals.
    """
    # Importing Raw Data
    SEC_flow_path = Path(__file__).resolve().parent.parent / "Raw_Data" / "Data_Seine_centre" / "Débit_SEC_2022_2025.csv"
    debit_sec = pd.read_csv(SEC_flow_path, sep = ";", index_col = 0, parse_dates = True, dayfirst = True) 

    del debit_sec["USI.COL.S1.CAP_Q.DEB.MES_VAL_TM Moyenne:Qualif"]
    debit_sec = debit_sec.rename(columns = {"USI.COL.S1.CAP_Q.DEB.MES_VAL_TM Moyenne:Valeur" : "Flow SEC"})

    # Resample in 15 minutes
    debit_sec = debit_sec.resample("15min").mean()

    # Interpolation to fill in missing data
    debit_sec = debit_sec.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)

    debit_sec.index.name = None

    return debit_sec


def SEC_preprocessing():
    """
    Validates the other data from "Seine centre" and returns these data in DataFrame format.

    The raw data is initially collected at 5-minute intervals.
    """
    # Importing Raw Data
    SEC_path = Path(__file__).resolve().parent.parent / "Raw_Data" / "Data_Seine_centre"
    df_data_1 = pd.read_excel(SEC_path / "Sec_2022.xlsx", header = 1, index_col = 0)
    df_data_2 = pd.read_excel(SEC_path / "Sec_2023.xlsx", header = 1, index_col = 0)
    df_data_3 = pd.read_excel(SEC_path / "Sec_2024.xlsx", header = 1, index_col = 0)

    df = pd.concat([df_data_1, df_data_2, df_data_3])
    df.index = pd.to_datetime(df.index, format = "%d/%m/%Y %H:%M")

    # Removing unused columns and renaming columns
    df.drop(columns = ['Analyse Interne : Brute NNH4', 
                       'Température à réception BRUTE', 
                       'ETAGE 2 : Analyseur NH4 - Sortie 2ème étage vers P3',
                       'ETAGE 2 : Analyseur NH4 - Sortie 2ème étage vers P4',
                       'Eau brute - Matières en suspension',
                       'Prétraitement : H2S chambre sécurité 5 min'
                      ], inplace = True)
    
    df = df.rename(columns = {"Brute : Sonde pH 5 min" : "pH SEC"})
    df = df.rename(columns = {"BRUTE : Sonde température 5 min" : "Temperature SEC"})
    df = df.rename(columns = {"Décantation : Turbidité canal d'alimentation (NTU) 5 minutes" : "Turbidity SEC"})

    # Resample in 15 minutes
    df = df.resample("15Min").mean()

    # Removing physically impossible negative turbidity values and interpolation
    df.loc[df["Turbidity SEC"] < 0, "Turbidity SEC"] = np.nan
    df["Turbidity SEC"] = df["Turbidity SEC"].interpolate(method = 'linear', limit_direction = 'forward', axis = 0)

    # Replacing zeros with NaNs and interpolation
    for name in df.columns:
        df.loc[df[name] == 0, name] = np.nan
    df = df.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)
    
    # Removing values outside the 1st–99th percentile range and interpolation
    for name in df.columns: 
        q_inf = df[name].diff().quantile(0.01)
        q_sup = df[name].diff().quantile(0.99)
        df.loc[df[name].diff() < q_inf, name] = np.nan 
        df.loc[df[name].diff() > q_sup, name] = np.nan
    df = df.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)
    
    # Removing observations before 25 November 2022, as pH and temperature data are only available from this date
    index_position = df.index.get_loc('2022-11-25 07:00:00')
    df = df.iloc[index_position:]

    return df


def SAV_preprocessing():
    """
    Validates data from "Seine aval" and returns these data in DataFrame format.

    The raw data is initially collected at 15-minute intervals.
    """
    # Importing Raw Data
    SAV_path = Path(__file__).resolve().parent.parent / "Raw_Data" / "Data_Seine_aval"
    df_data_1 = pd.read_excel(SAV_path  / "31_12_21 au 31_12_22.xlsx", header = 1, index_col = 0)
    df_data_2 = pd.read_excel(SAV_path  / "31_12_22 au 01_01_25.xlsx", header = 1, index_col = 0)

    # Renaming due to a change in the tag names in df_data_2 and concatenation
    df_data_2 = df_data_2.rename(columns={"Total eau brute station (valable jusqu'en 2023)\n" : "Total eau brute station"})
    
    df_data = pd.concat([df_data_1,df_data_2])
    df_data.index = pd.to_datetime(df_data.index, format = "%d/%m/%Y %H:%M")

    # Removing unused columns and renaming columns
    df_data.drop(columns = ["Débit d'eau brute A1",
                            "Débit d'eau brute A2",
                            "Débit d'eau brute A3 imp.",
                            "Débit d'eau brute A3 pair",
                            "Débit d'eau brute A4"
                           ], inplace = True)

    df_data = df_data.rename(columns={"Mesure de la conductivité au prétraitement" : "Conductivity SAV"})
    df_data = df_data.rename(columns={"Mesure du pH au prétraitement" : "pH SAV"})
    df_data = df_data.rename(columns={"Température EB canal A4" : "Temperature SAV"})
    df_data = df_data.rename(columns={"Pret : MES eaux prétraitées vers VA5" : "TSS SAV"})
    df_data = df_data.rename(columns={"Total eau brute station" : "Total flow"})

    """
    Several duplicate observations are present in the dataset. Based on a 
    verification of the original Excel file, the first occurrences of duplicated 
    observations are removed and the last occurrences are retained, as most of 
    the data associated with the first occurrences are missing in the original file.

    """
    # removing duplicates and organizing indexes
    df_data = df_data[~df_data.index.duplicated(keep='last')]

    full_index = pd.date_range(start = df_data.index.min(), end = df_data.index.max(), freq = '15min')
    df_data = df_data.reindex(full_index)

     # Replacing zeros with NaNs and interpolation
    for name in df_data.columns:
        df_data.loc[df_data[name] == 0, name] = np.nan
    
    df_data = df_data.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)

    # Removing values outside the 1st–99th percentile range and interpolation
    for name in df_data.columns:
        q_inf = df_data[name].diff().quantile(0.01)
        q_sup = df_data[name].diff().quantile(0.99)
        df_data.loc[df_data[name].diff() < q_inf, name] = np.nan 
        df_data.loc[df_data[name].diff() > q_sup, name] = np.nan
        
    df_data = df_data.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)

    # Removing outliers from a specified data range (see the article for more details) and interpolation
    df_data.loc[df_data["Conductivity SAV"] < 200, "Conductivity SAV"] = np.nan
    df_data.loc[df_data["pH SAV"] < 4, "pH SAV"] = np.nan
    df_data.loc[df_data["TSS SAV"] > 650, "TSS SAV"] = np.nan

    df_data["Conductivity SAV"] = df_data["Conductivity SAV"].interpolate()
    df_data["pH SAV"] = df_data["pH SAV"].interpolate()
    df_data["TSS SAV"] = df_data["TSS SAV"].interpolate()

    return df_data