import os
import numpy as np
from datetime import datetime
from jinja2 import Template
import codecs
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

def create_rawDataFrames(srcPath):
    """
    Creates dataFrames from the csv files in the srcPath and then saves them in the data/raw_dataFrames directory using pickle.
        
    Parameters
    ----------
    srcPath : str
        The path to the csv files.

    Returns
    -------
    dataDic : dict
        A dictionary with the dataFrames.    
    """
    if not os.path.exists("data/raw_dataFrames"):
        os.makedirs("data/raw_dataFrames")

    dataDic = {"train": pd.read_csv(f"{srcPath}train_timeseries/train_timeseries.csv"),
               "test": pd.read_csv(f"{srcPath}test_timeseries/test_timeseries.csv"),
               "validation": pd.read_csv(f"{srcPath}validation_timeseries/validation_timeseries.csv"),
               "soil" : pd.read_csv(f"{srcPath}soil_data.csv"),
               }
    
    dfs = {
    k: dataDic[k].set_index(['fips', 'date'])
    for k in dataDic.keys() if k != "soil"
    }
    dfs["soil"] = dataDic["soil"]
    
    for k in dfs.keys():
        with open(f"data/raw_dataFrames/{k}.pickle", "wb") as f:
            pickle.dump(dfs[k], f)
    return dfs


def load_rawDataFrames():
    """
    This function loads the dataFrames dict from the data/raw_dataFrames directory.
        
    Returns
    -------
    dfs : dict
        A dictionary with the dataFrames.    
    """
    dfs = {}
    for file in os.listdir("data/raw_dataFrames"):
        with open(f"data/raw_dataFrames/{file}", "rb") as f:
            dfs[file.split(".")[0]] = pickle.load(f)
    return dfs

def interpolate_nans(padata, pkind="linear"):
    """
    Taken from: https://stackoverflow.com/a/53050216/2167159

    Parameters
    ----------
    padata : np.array
        The array to interpolate.
    pkind : str
        The interpolation method.
        
    Returns
    -------
    f(aindexes) : np.array
        The interpolated array.
    """
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes,
                 padata[agood_indexes],
                 bounds_error=False,
                 copy=False,
                 fill_value="extrapolate",
                 kind=pkind,
               )
    return f(aindexes)


def date_encode(date):
    """
    Encode the cycling feature : the day of the year as a sin and cos function.
    Taken from https://www.pure.ed.ac.uk/ws/portalfiles/portal/217133242/DroughtED_MINIXHOFER_DOA18062021_AFV.pdf

    Parameters
    ----------
    date : str
        Date to encode.
        
    Returns
    -------
    np.sin(2 * np.pi * date.timetuple().tm_yday / 366) : float
        Sin of the date.
    np.cos(2 * np.pi * date.timetuple().tm_yday / 366) : float
        Cos of the date.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    return (
        np.sin(2 * np.pi * date.timetuple().tm_yday / 366),
        np.cos(2 * np.pi * date.timetuple().tm_yday / 366),
    )

def setup_encoders_targets():
    """
    A function to setup the class2id and id2class dictionaries.
    """
    class2id= { 'None': 0, 'D0': 1, 'D1': 2, 'D2': 3, 'D3': 4, 'D4': 5}
    id2class = {v: k for k, v in class2id.items()}

    return class2id, id2class


def loadXY(
    dfs,
    df,
    random_state=42,
    window_size=180, # how many days in the past (paper_default: 180).
    target_size=6, # how many weeks into the future (paper_default: 6).
    fuse_past=True, # add the past drought observations? (paper_best: True)
    return_fips=False, # return the county identifier.
    encode_season=True, # encode the season using the function above (paper_best: True).
    use_prev_year=False, # add observations from 1 year prior? (paper_best: True).
    ):
    """
    Load the data and create the X and y arrays.
    Adapted from https://www.pure.ed.ac.uk/ws/portalfiles/portal/217133242/DroughtED_MINIXHOFER_DOA18062021_AFV.pdf

    Parameters
    ----------
    dfs : dict
        Dictionary containing the dataFrames.
    df : pd.DataFrame
        DataFrame to load the data from.
    random_state : int
        Random state to use.
    window_size : int
        Number of days in the past used for prediction.
    target_size : int
        Number of weeks into the future (the size of the output vector).
    fuse_past : bool
        Add the past drought observations.
    return_fips : bool
        Return the county identifier.
    encode_season : bool
        Encode the season.
    use_prev_year : bool
        Add observations from 1 year prior.
        
    Returns
    -------
    X : np.array
        The input array.
    y : np.array
        The output array.
    fips : np.array
        The county identifier.
    dico_trad : dict
        The dictionary containing the translation of the categorical features, only created when processing the training data.
    list_cat : list
        The list of the number of unique categories in each categorical feature.
    """

    df = dfs[df]
    soil_df = dfs["soil"]
    time_data_cols = sorted(
        [c for c in df.columns if c not in ["fips", "date", "score"]]
    )
    static_data_cols = sorted(
        [c for c in soil_df.columns if c not in ["soil", "lat", "lon"]]
    )
    count = 0
    score_df = df.dropna(subset=["score"])
    X_static = np.empty((len(df) // window_size, len(static_data_cols)))
    X_fips_date = []
    add_dim = 0
    if use_prev_year:
        add_dim += len(time_data_cols)
    if fuse_past:
        add_dim += 1
        if use_prev_year:
            add_dim += 1
    if encode_season:
        add_dim += 2
    X_time = np.empty(
        (len(df) // window_size, window_size, len(time_data_cols) + add_dim)
    )
    y_past = np.empty((len(df) // window_size, window_size))
    y_target = np.empty((len(df) // window_size, target_size))
    if random_state is not None:
        np.random.seed(random_state)
    for fips in tqdm(score_df.index.get_level_values(0).unique()):
        if random_state is not None:
            start_i = np.random.randint(1, window_size)
        else:
            start_i = 1
        fips_df = df[(df.index.get_level_values(0) == fips)]
        X = fips_df[time_data_cols].values
        y = fips_df["score"].values
        X_s = soil_df[soil_df["fips"] == fips][static_data_cols].values[0]
        for i in range(start_i, len(y) - (window_size + target_size * 7), window_size):
            X_fips_date.append((fips, fips_df.index[i : i + window_size][-1]))
            X_time[count, :, : len(time_data_cols)] = X[i : i + window_size]
            if use_prev_year:
                if i < 365 or len(X[i - 365 : i + window_size - 365]) < window_size:
                    continue
                X_time[count, :, -len(time_data_cols) :] = X[
                    i - 365 : i + window_size - 365
                ]
            if not fuse_past:
                y_past[count] = interpolate_nans(y[i : i + window_size])
            else:
                X_time[count, :, len(time_data_cols)] = interpolate_nans(
                    y[i : i + window_size]
                )
            if encode_season:
                enc_dates = [
                    date_encode(d) for f, d in fips_df.index[i : i + window_size].values
                ]
                d_sin, d_cos = [s for s, c in enc_dates], [c for s, c in enc_dates]
                X_time[count, :, len(time_data_cols) + (add_dim - 2)] = d_sin
                X_time[count, :, len(time_data_cols) + (add_dim - 2) + 1] = d_cos
            temp_y = y[i + window_size : i + window_size + target_size * 7]
            y_target[count] = np.array(temp_y[~np.isnan(temp_y)][:target_size])
            X_static[count] = X_s
            count += 1
    print(f"loaded {count} samples")
    results = [X_static[:count], X_time[:count], y_target[:count]]
    if not fuse_past:
        results.append(y_past[:count])
    if return_fips:
        results.append(X_fips_date)
    return results


def normalize(scaler_dict, scaler_dict_static, scaler_dict_past, X_static, X_time, y_past=None, fit=False):
    """
    Normalize the data using a RobustScaler.

    Parameters
    ----------
    scaler_dict : dict
        Dictionary containing the scalers for the time series.
    scaler_dict_static : dict
        Dictionary containing the scalers for the static data.
    scaler_dict_past : dict
        Dictionary containing the scalers for the past drought observations.
    X_static : np.array
        The static data.
    X_time : np.array
        The time series data.
    y_past : np.array
        The past drought observations.
    fit : bool
        Fit the scalers.
    
    Returns
    -------
    scaler_dict : dict
        Dictionary containing the scalers for the time series.
    scaler_dict_static : dict
        Dictionary containing the scalers for the static data.
    scaler_dict_past : dict
        Dictionary containing the scalers for the past drought observations.
    X_static : np.array
        The normalized static data.
    X_time : np.array
        The normalized time series data.
    y_past : np.array
        The normalized past drought observations.
    """
    for index in tqdm(range(X_time.shape[-1])):
        if fit:
            scaler_dict[index] = RobustScaler().fit(X_time[:, :, index].reshape(-1, 1))
        X_time[:, :, index] = (
            scaler_dict[index]
            .transform(X_time[:, :, index].reshape(-1, 1))
            .reshape(-1, X_time.shape[-2])
        )
    for index in tqdm(range(X_static.shape[-1])):
        if fit:
            scaler_dict_static[index] = RobustScaler().fit(
                X_static[:, index].reshape(-1, 1)
            )
        X_static[:, index] = (
            scaler_dict_static[index]
            .transform(X_static[:, index].reshape(-1, 1))
            .reshape(1, -1)
        )
    index = 0
    if y_past is not None:
        if fit:
            scaler_dict_past[index] = RobustScaler().fit(y_past.reshape(-1, 1))
        y_past[:, :] = (
            scaler_dict_past[index]
            .transform(y_past.reshape(-1, 1))
            .reshape(-1, y_past.shape[-1])
        )
        return scaler_dict, scaler_dict_static, scaler_dict_past, X_static, X_time, y_past
    return scaler_dict, scaler_dict_static, scaler_dict_past, X_static, X_time


def data_pretreatment_baseline():
    """
    
    """
    with open("data/data.pkl", "rb") as f:
        data = pickle.load(f)
    X_tabular_train = data["X_tabular_train"]
    X_time_train = data["X_time_train"]
    X_tabular_valid = data["X_tabular_validation"]
    X_time_valid = data["X_time_validation"]
    X_tabular_test = data["X_tabular_test"]
    X_time_test = data["X_time_test"]

    scaler_dict = {}
    scaler_dict_static = {}
    scaler_dict_past = {}

    scaler_dict, scaler_dict_static, scaler_dict_past, X_tabular_train, X_time_train = normalize(scaler_dict,
                                                                                                 scaler_dict_static,
                                                                                                 scaler_dict_past,
                                                                                                 X_tabular_train,
                                                                                                 X_time_train,
                                                                                                 fit=True)
    scaler_dict, scaler_dict_static, scaler_dict_past, X_tabular_valid, X_time_valid = normalize(scaler_dict,
                                                                                                      scaler_dict_static,
                                                                                                      scaler_dict_past,
                                                                                                      X_tabular_valid,
                                                                                                      X_time_valid)
    scaler_dict, scaler_dict_static, scaler_dict_past, X_tabular_test, X_time_test = normalize(scaler_dict,
                                                                                               scaler_dict_static,
                                                                                               scaler_dict_past,
                                                                                               X_tabular_test,
                                                                                               X_time_test)
    data["X_tabular_train"] = X_tabular_train
    data["X_time_train"] = X_time_train
    data["X_tabular_validation"] = X_tabular_valid
    data["X_time_validation"] = X_time_valid
    data["X_tabular_test"] = X_tabular_test
    data["X_time_test"] = X_time_test

    return data

def data_pretreatment_hm():
    """
    
    """
    with open("data/data.pkl", "rb") as f:
        data = pickle.load(f)
    X_tabular_train = data["X_tabular_train"]
    X_time_train = data["X_time_train"]
    X_tabular_valid = data["X_tabular_validation"]
    X_time_valid = data["X_time_validation"]
    X_tabular_test = data["X_tabular_test"]
    X_time_test = data["X_time_test"]

    STATIC_COLS = ['fips', 'lat', 'lon', 'elevation', 'slope1', 'slope2', 'slope3',
                   'slope4', 'slope5', 'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE',
                   'aspectS', 'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND',
                   'URB_LAND', 'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND',
                   'CULT_LAND', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']

    ordered_cols = sorted([c for c in STATIC_COLS if c not in ["soil", "lat", "lon"]])
    # cat_cols = [ordered_cols.index(i) for i in ["fips", "SQ1", "SQ2", "SQ3", "SQ4", "SQ5", "SQ6", "SQ7"]]
    cat_cols = [ordered_cols.index(i) for i in ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5", "SQ6", "SQ7"]]
    X_tabular_cat_train = X_tabular_train[:,cat_cols].astype(int)
    X_tabular_train = X_tabular_train[:,[i for i in range(X_tabular_train.shape[1]) if i not in cat_cols]]

    X_tabular_cat_valid = X_tabular_valid[:,cat_cols].astype(int)
    X_tabular_valid = X_tabular_valid[:,[i for i in range(X_tabular_valid.shape[1]) if i not in cat_cols]]

    X_tabular_cat_test = X_tabular_test[:,cat_cols].astype(int)
    X_tabular_test = X_tabular_test[:,[i for i in range(X_tabular_test.shape[1]) if i not in cat_cols]]

    dico_trad = {}
    for cat in range(X_tabular_cat_train.shape[1]):
        dico_trad[cat] = {j: i for i,j in enumerate(sorted(np.unique_values(X_tabular_cat_train[:,cat])))}
        dico_trad[cat]["unknown"] = len(np.unique_values(X_tabular_cat_train[:,cat]))
    
    for cat in range(len(cat_cols)):
        X_tabular_cat_train[:,cat] = [dico_trad[cat][i] for i in X_tabular_cat_train[:,cat]]
        X_tabular_cat_valid[:,cat] = [dico_trad[cat][i] if i in dico_trad[cat] else dico_trad[cat]["unknown"] for i in X_tabular_cat_valid[:,cat]]
        X_tabular_cat_test[:,cat] = [dico_trad[cat][i] if i in dico_trad[cat] else dico_trad[cat]["unknown"] for i in X_tabular_cat_test[:,cat]]

    scaler_dict = {}
    scaler_dict_static = {}
    scaler_dict_past = {}

    scaler_dict, scaler_dict_static, scaler_dict_past, X_tabular_train, X_time_train = normalize(scaler_dict,
                                                                                                 scaler_dict_static,
                                                                                                 scaler_dict_past,
                                                                                                 X_tabular_train,
                                                                                                 X_time_train,
                                                                                                 fit=True)
    scaler_dict, scaler_dict_static, scaler_dict_past, X_tabular_valid, X_time_valid = normalize(scaler_dict,
                                                                                                      scaler_dict_static,
                                                                                                      scaler_dict_past,
                                                                                                      X_tabular_valid,
                                                                                                      X_time_valid)
    scaler_dict, scaler_dict_static, scaler_dict_past, X_tabular_test, X_time_test = normalize(scaler_dict,
                                                                                               scaler_dict_static,
                                                                                               scaler_dict_past,
                                                                                               X_tabular_test,
                                                                                               X_time_test)
    data["X_tabular_train"] = X_tabular_train
    data["X_tabular_cat_train"] = X_tabular_cat_train
    data["X_time_train"] = X_time_train
    data["X_tabular_validation"] = X_tabular_valid
    data["X_tabular_cat_validation"] = X_tabular_cat_valid
    data["X_time_validation"] = X_time_valid
    data["X_tabular_test"] = X_tabular_test
    data["X_tabular_cat_test"] = X_tabular_cat_test
    data["X_time_test"] = X_time_test

    return data


def create_simple_report(template_path, output_path, data):
    """
    from: https://stackoverflow.com/a/63622717/21823869
    """
    with open(template_path, "r") as f:
        template = Template(f.read(), trim_blocks=True)
    rendered = template.render(exp=data)
    output_file = codecs.open(output_path, "w", "utf-8")
    output_file.write(rendered)
    output_file.close()
    print(f"Report saved at {output_path}")
