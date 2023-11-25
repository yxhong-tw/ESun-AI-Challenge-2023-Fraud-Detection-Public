"""
本檔案包含了數個「常用」的函式。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

from json import load
from logging import DEBUG, FileHandler, Formatter, Logger, StreamHandler, getLogger
from os.path import exists, join
from pandas import DataFrame, read_csv, read_hdf
from typing import Any, Dict

logger = getLogger(__name__)


def get_df(data_name: str) -> DataFrame:
    """ Load data from csv or hdf5 file.

    Args:
        data_name (str): The name of data which you want to load.

    Returns:
        DataFrame: The dataframe of loaded data.
    """

    data_path = join("data", (data_name + ".csv"))

    if not exists(path=data_path):
        data_path = join("data", (data_name + ".h5"))

        return read_hdf(path_or_buf=data_path, key="data")

    return read_csv(filepath_or_buffer=data_path)


def get_logger(log_name: str) -> Logger:
    """ Initialize logger.

    Args:
        log_name (str): The file name of log.

    Returns:
        logging.Logger: The logger.
    """

    formatter = Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    filename = join("logs", log_name)
    fh = FileHandler(filename=filename, mode="a", encoding="UTF-8")
    fh.setFormatter(fmt=formatter)
    fh.setLevel(level=DEBUG)

    sh = StreamHandler()
    sh.setFormatter(fmt=formatter)
    sh.setLevel(level=DEBUG)

    logger = getLogger()
    logger.addHandler(hdlr=fh)
    logger.addHandler(hdlr=sh)
    logger.setLevel(level=DEBUG)

    return logger


def get_model_params(model_params_path: str) -> Dict[str, Any]:
    """ Load model parameters from json file.

    Args:
        model_params_path (str): The path of model parameters file.

    Returns:
        Dict[str, Any]: The model parameters.
    """

    model_params = None

    with open(file=model_params_path, mode="r", encoding="UTF-8") as f:
        model_params = load(fp=f)
        f.close()

    return model_params


def preprocess_data(df: DataFrame, fill_nan: bool) -> Dict[str, Any]:
    """ Preprocess the input data.

    Args:
        df (DataFrame): The dataframe of input data.
        fill_nan (bool): Whether fill nan or not.

    Returns:
        Dict[str, Any]: The processed input data.
    """

    try:
        x = df.drop(labels="label", axis=1)
        y = df.label
    except:
        x = df
        y = None

    if fill_nan:
        nan = 100000000

        x["etymd"] = x["etymd"].fillna(value=nan).astype(dtype=int)
        x["mcc"] = x["mcc"].fillna(value=nan).astype(dtype=int)
        x["stocn"] = x["stocn"].fillna(value=nan).astype(dtype=int)
        x["scity"] = x["scity"].fillna(value=nan).astype(dtype=int)
        x["stscd"] = x["stscd"].fillna(value=nan).astype(dtype=int)
        x["hcefg"] = x["hcefg"].fillna(value=nan).astype(dtype=int)
        x["csmcu"] = x["csmcu"].fillna(value=nan).astype(dtype=int)

        x["locdt_7_remainder"] = x["locdt"].apply(func=lambda x: x % 7)
        x["loctm"] = x["loctm"].apply(func=lambda x: (
            (int(x / 10000) * 3600) + ((int(x / 100) % 100) * 60) + (x % 100)))

    try:
        x = x.drop(labels="txkey", axis=1)
    except:
        logger.info(msg="x does not have \"txkey\" feature.")

    preprocessed_data = {
        "x": x,
        "y": y,
    }

    # For not OpenFE data.
    # preprocessed_data["categ_features"] = [
    #     "chid", "cano", "contp", "etymd", "mchno", "acqic", "mcc", "ecfg",
    #     "insfg", "bnsfg", "stocn", "scity", "stscd", "ovrlt", "flbmk", "hcefg",
    #     "csmcu", "flg_3dsmk", "locdt_7_remainder"
    # ]

    # If you re-train the model and re-infer the data, you should update the "categ_features" below.
    # For OpenFE data (30 features).
    preprocessed_data["categ_features"] = [
        "chid", "cano", "contp", "etymd", "mchno", "acqic", "mcc", "ecfg",
        "insfg", "bnsfg", "stocn", "scity", "stscd", "ovrlt", "flbmk", "hcefg",
        "csmcu", "flg_3dsmk", "locdt_7_remainder", "autoFE_f_19"
    ]

    # For OpenFE data (200 features).
    # preprocessed_data["categ_features"] = [
    #     "chid", "cano", "contp", "etymd", "mchno", "acqic", "mcc", "ecfg",
    #     "insfg", "bnsfg", "stocn", "scity", "stscd", "ovrlt", "flbmk", "hcefg",
    #     "csmcu", "flg_3dsmk", "locdt_7_remainder", "autoFE_f_19",
    #     "autoFE_f_31", "autoFE_f_43", "autoFE_f_45", "autoFE_f_56",
    #     "autoFE_f_73", "autoFE_f_79", "autoFE_f_82", "autoFE_f_89",
    #     "autoFE_f_90", "autoFE_f_91", "autoFE_f_101", "autoFE_f_105",
    #     "autoFE_f_133", "autoFE_f_137", "autoFE_f_139", "autoFE_f_144",
    #     "autoFE_f_159", "autoFE_f_160", "autoFE_f_163", "autoFE_f_177",
    #     "autoFE_f_187", "autoFE_f_189"
    # ]

    return preprocessed_data
