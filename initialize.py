"""
本檔案包含了所有和「初始化」有關的函式。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

import numpy
import random

from catboost import CatBoostClassifier, Pool
from configparser import ConfigParser
from logging import getLogger
from os.path import join
from pandas import DataFrame, core
from pickle import load
from typing import Any, Dict, Tuple

from utils import get_df, get_model_params, preprocess_data

logger = getLogger(__name__)


def initialize(configs: ConfigParser, mode: str,
               use_openfe_data: bool) -> Dict[str, Any]:
    """ Initialize parameters.

    Args:
        configs (ConfigParser): The configs.
        mode (str): The mode.
        use_openfe_data (bool): Whether to use openfe data.

    Raises:
        ValueError: (Train) The numbers of rows are not matched.
        ValueError: (Validate) The numbers of rows are not matched.

    Returns:
        Dict[str, Any]: The parameters.
    """

    logger.info(msg=f"Initialization has been started.")

    parameters = {}
    for section in configs.sections():
        for option, value in configs.items(section=section):
            parameters[option] = value

    seed = configs.getint(section="GENERAL", option="seed")
    initialize_seeds(seed=seed)

    if mode == "inference":
        data_name = "unlabeled"

        origin_unlabeled_df, unlabeled_data = initialize_dataset(
            data_name=data_name, return_pool=False if use_openfe_data else True)

        if use_openfe_data:
            data_name = (configs.get(section="OPENFE", option="data_version") +
                         "_" + data_name)

            _, unlabeled_data = initialize_dataset(data_name=data_name,
                                                   return_pool=False)

            # If you re-train the model and re-infer the data, you should update the "drop_columns" below.
            # For OpenFE data (30 features).
            drop_columns = [
                "autoFE_f_0", "autoFE_f_2", "autoFE_f_18", "autoFE_f_29"
            ]

            logger.info(
                msg=
                "Below features are dropped because they contain NaN values.")

            for column_name in drop_columns:
                logger.info(msg=f"{column_name}")
                unlabeled_data["x"] = unlabeled_data["x"].drop(
                    labels=column_name, axis=1)

            logger.info(
                msg="Change the column type to \"str\" if it is \"category\".")

            for column_name in unlabeled_data["x"].columns.values:
                if unlabeled_data["x"][column_name].dtype.name == 'category':
                    unlabeled_data["x"][column_name] = unlabeled_data["x"][
                        column_name].astype(dtype=str)

            unlabeled_data = Pool(
                data=unlabeled_data["x"],
                cat_features=unlabeled_data["categ_features"])

        parameters["origin_unlabeled_df"] = origin_unlabeled_df
        parameters["unlabeled_data"] = unlabeled_data
    elif mode == "openfe-train":
        # For get_and_save_candidate_features().
        numerical_features = ["conam", "iterm", "flam1", "csmam"]
        categorical_features = [
            "chid", "cano", "contp", "etymd", "mchno", "acqic", "mcc", "ecfg",
            "insfg", "bnsfg", "stocn", "scity", "stscd", "ovrlt", "flbmk",
            "hcefg", "csmcu", "flg_3dsmk", "locdt_7_remainder"
        ]
        ordinal_features = ["locdt", "loctm"]

        # For get_and_save_features().
        _, train_data = initialize_dataset(data_name="train",
                                           return_pool=False)
        task = "classification"
        metric = "binary_logloss"
        n_data_blocks = 8
        min_candidate_features = 300000
        stage2_params = {
            "importance_type": "gain",
            "n_estimators": 1000,
            "n_jobs": 32,
            "num_leaves": 64,
            "seed": seed
        }
        n_jobs = 32

        directory = join("outputs", "openfes")
        version = configs.get(section="GENERAL", option="version")

        parameters["categorical_features"] = categorical_features
        parameters["directory"] = directory
        parameters["metric"] = metric
        parameters["min_candidate_features"] = min_candidate_features
        parameters["n_data_blocks"] = n_data_blocks
        parameters["n_jobs"] = n_jobs
        parameters["numerical_features"] = numerical_features
        parameters["ordinal_features"] = ordinal_features
        parameters["seed"] = seed
        parameters["stage2_params"] = stage2_params
        parameters["task"] = task
        parameters["train_data"] = train_data
        parameters["version"] = version
    elif mode == "openfe-inference":
        _, train_data = initialize_dataset(data_name="train",
                                           return_pool=False)

        _, unlabeled_data = initialize_dataset(data_name="unlabeled",
                                               return_pool=False)

        _, validate_data = initialize_dataset(data_name="new_public",
                                              return_pool=False)

        directory = join("outputs", "openfes")
        data_version = configs.get(section="OPENFE", option="data_version")

        file = join(directory, f"{data_version}_features.pkl")
        with open(file=file, mode="rb") as f:
            features = load(f)
            f.close()

        n_jobs = 32
        version = configs.get(section="GENERAL", option="version")

        parameters["directory"] = directory
        parameters["features"] = features
        parameters["n_jobs"] = n_jobs
        parameters["train_data"] = train_data
        parameters["unlabeled_data"] = unlabeled_data
        parameters["validate_data"] = validate_data
        parameters["version"] = version
    # mode == "evalute" or mode == "cv_train_with_optuna", "train" or mode == "train_with_optuna"
    else:
        data_name = "train"

        _, train_data = initialize_dataset(
            data_name=data_name,
            return_pool=False if use_openfe_data else True)

        if use_openfe_data:
            data_name = (configs.get(section="OPENFE", option="data_version") +
                         "_" + data_name + "_x")

            _, train_openfe_data = initialize_dataset(data_name=data_name,
                                                      return_pool=False)

            if len(train_openfe_data["x"]) != len(train_data["y"]):
                message = "(Train) The numbers of rows are not matched."
                raise ValueError(message)

            train_data["x"] = train_openfe_data["x"]

            drop_columns = set()

            logger.info(msg="(Train) Checking need to drop columns.")
            for column_name in train_data["x"].columns.values:
                if train_data["x"][column_name].isna().sum() > 0:
                    logger.info(
                        msg=
                        f"{column_name}: {train_data['x'][column_name].isna().sum()}, {train_data['x'][column_name].dtype}"
                    )

                    drop_columns.add(column_name)
                elif train_data["x"][column_name].dtype.name == "category":
                    logger.info(msg=f"The {column_name}'s dtype is category.")

                    train_data["x"][column_name] = train_data["x"][
                        column_name].astype(dtype=str)

        # For validate data.
        data_name = "new_public"

        _, validate_data = initialize_dataset(
            data_name=data_name,
            return_pool=False if use_openfe_data else True)

        if use_openfe_data:
            data_name = (configs.get(section="OPENFE", option="data_version") +
                         "_" + data_name + "_x")

            _, validate_openfe_data = initialize_dataset(data_name=data_name,
                                                         return_pool=False)

            if len(validate_openfe_data["x"]) != len(validate_data["y"]):
                message = "(Validate) The numbers of rows are not matched."
                raise ValueError(message)

            validate_data["x"] = validate_openfe_data["x"]

            logger.info(msg="Checking need to drop columns.")
            for column_name in validate_data["x"].columns.values:
                if validate_data['x'][column_name].isna().sum() > 0:
                    logger.info(
                        msg=
                        f"{column_name}: {validate_data['x'][column_name].isna().sum()}, {validate_data['x'][column_name].dtype}"
                    )

                    drop_columns.add(column_name)
                elif validate_data["x"][column_name].dtype.name == 'category':
                    validate_data["x"][column_name] = validate_data["x"][
                        column_name].astype(dtype=str)

        if use_openfe_data:
            logger.info(msg="Drop columns.")

            for column_name in drop_columns:
                logger.info(msg=f"{column_name} has been dropped.")

                train_data["x"] = train_data["x"].drop(labels=column_name,
                                                       axis=1)
                validate_data["x"] = validate_data["x"].drop(
                    labels=column_name, axis=1)

            train_data = Pool(data=train_data["x"],
                              label=train_data["y"],
                              cat_features=train_data["categ_features"])

            validate_data = Pool(data=validate_data["x"],
                                 label=validate_data["y"],
                                 cat_features=validate_data["categ_features"])

        version = configs.get(section="GENERAL", option="version")

        parameters["train_data"] = train_data
        parameters["validate_data"] = validate_data
        parameters["version"] = version

    if mode == "evalute" or mode == "inference" or mode == "train":
        model = initialize_model(configs=configs)
        parameters["model"] = model

    logger.info(msg=f"The details of parameters: \n{parameters}")
    logger.info(msg=f"Initialization has been finished.")

    return parameters


def initialize_dataset(data_name: str,
                       return_pool: bool = True) -> Tuple[DataFrame, Any]:
    """ Initialize dataset.

    Args:
        data_name (str): The name of data.
        return_pool (bool, optional): Whether to return Pool. Defaults to True.

    Returns:
        Tuple[DataFrame, Any]: The origin dataframe and preprocessed data.
    """

    logger.info(msg=f"Loading {data_name} data.")

    origin_df = get_df(data_name=data_name)

    fill_nan = True if data_name == "train" or data_name == "unlabeled" or data_name == "new_public" else False

    preprocessed_data = preprocess_data(df=origin_df, fill_nan=fill_nan)

    if return_pool:
        # "new_public" is validate data.
        if "train" in data_name or "new_public" in data_name:
            preprocessed_data = Pool(
                data=preprocessed_data["x"],
                label=preprocessed_data["y"],
                cat_features=preprocessed_data["categ_features"])
        else:
            preprocessed_data = Pool(
                data=preprocessed_data["x"],
                cat_features=preprocessed_data["categ_features"])

    logger.info(msg=f"{data_name} data has been loaded.")

    return origin_df, preprocessed_data


def initialize_model(
        configs: ConfigParser = None,
        model_params: Dict[str, Any] = None) -> CatBoostClassifier:
    """ Initialize model.

    Args:
        configs (ConfigParser, optional): The configs. Defaults to None.
        model_params (Dict[str, Any], optional): The model parameters. Defaults to None.

    Returns:
        CatBoostClassifier: The model.
    """

    models = {"CatBoostClassifier": CatBoostClassifier}
    model = None

    # Load model parameters from configs.
    if model_params == None:
        model_name = configs.get(section="MODEL", option="model_name")
        model_params_path = join("configs", "params", (model_name + ".json"))
        model_params = get_model_params(model_params_path=model_params_path)
    # Load model parameters from initialized parameters.
    else:
        model_name = "CatBoostClassifier"

    model = models[model_name](**model_params)

    # Load checkpoint.
    try:
        if configs.get(section="MODEL", option="checkpoint_version") != "None":
            checkpoint_path = join(
                "outputs", "checkpoints",
                (configs.get(section="MODEL", option="checkpoint_version") +
                 ".cbm"))
            model.load_model(fname=checkpoint_path)

            logger.info(
                msg=f"Checkpoint has been loaded from {checkpoint_path}.")
        else:
            logger.info(msg=f"No checkpoint has been loaded.")
    except:
        logger.info(
            msg=
            "Raise an exception when loading checkpoint. \nIt may be caused by no configs is given."
        )

    logger.info(msg=f"The details of {model_name}: \n{model.get_params()}")
    logger.info(msg=f"{model_name} has been initialized.")

    return model


# References:
# - [AWS - Tune a CatBoost model](https://docs.aws.amazon.com/sagemaker/latest/dg/catboost-tuning.html#:~:text=Tunable%20CatBoost%20hyperparameters)
# - [CatBoost - Training Parameters](https://catboost.ai/en/docs/references/training-parameters/)
# - [CSDN - catboost参数详解及实战（强推）](https://blog.csdn.net/a7303349/article/details/125570737)
# - [知乎 - catboost：kaggle参数设置参考](https://zhuanlan.zhihu.com/p/136697031)
# - [腾讯云 - 一文详尽系列之CatBoost](https://cloud.tencent.com/developer/article/1543079)
# - [CatBoost GitHub - hyperparameters_tuning_using_optuna_and_hyperopt.ipynb](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb)
def initialize_model_parameters(trial=None) -> Dict[str, Any]:
    """ Initialize model parameters.

    Args:
        trial (_type_, optional): The trial object (Optuna required). Defaults to None.

    Returns:
        Dict[str, Any]: The model parameters.
    """

    # For feature_weights.
    # locdt = trial.suggest_float("locdt", 0.0, 10.0) if trial != None else 1.0
    # loctm = trial.suggest_float("loctm", 0.0, 10.0) if trial != None else 1.0
    # chid = trial.suggest_float("chid", 0.0, 10.0) if trial != None else 1.0
    # cano = trial.suggest_float("cano", 0.0, 10.0) if trial != None else 1.0
    # contp = trial.suggest_float("contp", 0.0, 10.0) if trial != None else 1.0
    # etymd = trial.suggest_float("etymd", 0.0, 10.0) if trial != None else 1.0
    # mchno = trial.suggest_float("mchno", 0.0, 10.0) if trial != None else 1.0
    # acqic = trial.suggest_float("acqic", 0.0, 10.0) if trial != None else 1.0
    # mcc = trial.suggest_float("mcc", 0.0, 10.0) if trial != None else 1.0,
    # conam = trial.suggest_float("conam", 0.0, 10.0) if trial != None else 1.0
    # ecfg = trial.suggest_float("ecfg", 0.0, 10.0) if trial != None else 1.0,
    # insfg = trial.suggest_float("insfg", 0.0, 10.0) if trial != None else 1.0
    # iterm = trial.suggest_float("iterm", 0.0, 10.0) if trial != None else 1.0
    # bnsfg = trial.suggest_float("bnsfg", 0.0, 10.0) if trial != None else 1.0
    # flam1 = trial.suggest_float("flaml", 0.0, 10.0) if trial != None else 1.0
    # stocn = trial.suggest_float("stocn", 0.0, 10.0) if trial != None else 1.0
    # scity = trial.suggest_float("scity", 0.0, 10.0) if trial != None else 1.0
    # stscd = trial.suggest_float("stscd", 0.0, 10.0) if trial != None else 1.0
    # ovrlt = trial.suggest_float("ovrlt", 0.0, 10.0) if trial != None else 1.0
    # flbmk = trial.suggest_float("flbmk", 0.0, 10.0) if trial != None else 1.0
    # hcefg = trial.suggest_float("hcefg", 0.0, 10.0) if trial != None else 1.0
    # csmcu = trial.suggest_float("csmcu", 0.0, 10.0) if trial != None else 1.0
    # csmam = trial.suggest_float("csmam", 0.0, 10.0) if trial != None else 1.0
    # flg_3dsmk = trial.suggest_float("flg_3dsmk", 0.0, 10.0) if trial != None else 1.0

    model_params = {
        # Only for Bayesian bootstrap, range = [0.0, 1.0], default: 1.0
        "bagging_temperature": 0.65,
        # trial.suggest_float("bagging_temperature", 0.0, 1.0, step=0.05)
        # if trial else 1.0,

        # "Ordered" is better than "Plain".
        "boosting_type": "Ordered",

        # ["Bayesian", "Bernoulli", "MVS", "Poisson", "No"]
        # default: "Bayesian" for GPU, "MVS" for CPU
        # "MVS" is CPU-only.
        # "Poison" is GPU-only.
        "bootstrap_type": "Bayesian",

        # default: 6
        # "depth": 6,
        "depth": trial.suggest_int("depth", 6, 9) if trial else 6,

        # Use all gpus if no setting this parameter.
        "devices": "0",
        # "devices":
        # "0-1",

        # The metric used for calculation during training and validating.
        "eval_metric": "F1",

        # ["Median", "Uniform", "UniformAndQuantiles",
        #  "MaxLogSum", "MinEntropy", "GreedyLogSum], default: "GreedyLogSum"
        "feature_border_type": "GreedyLogSum",

        # "feature_weights": [
        #     locdt,
        #     loctm,
        #     chid,
        #     cano,
        #     contp,
        #     etymd,
        #     mchno,
        #     acqic,
        #     mcc,
        #     conam,
        #     ecfg,
        #     insfg,
        #     iterm,
        #     bnsfg,
        #     flam1,
        #     stocn,
        #     scity,
        #     stscd,
        #     ovrlt,
        #     flbmk,
        #     hcefg,
        #     csmcu,
        #     csmam,
        #     flg_3dsmk
        # ],

        # [SymmetricTree, Depthwise, Lossguide], default: "SymmetricTree"
        "grow_policy": "SymmetricTree",

        # default: 500
        # "iterations": 450,
        "iterations":
        trial.suggest_int("iterations", 300, 800) if trial else 500,

        # default: 3.0
        "l2_leaf_reg": 9.78874631581315,
        # trial.suggest_float("l2_leaf_reg", 1.0, 10.0) if trial else 3.0,

        # default: 0.03
        "learning_rate": 0.2218586703700778,
        # trial.suggest_float("learning_rate", 0.01, 0.8) if trial else 0.03,

        # default: "Logloss"
        "loss_function": "Logloss",

        # ["Forbidden", "Max", "Min"], default: "Min"
        "nan_mode": "Forbidden",

        # default: 0
        "random_seed": 48763,

        # default: 1.0
        "random_strength": 0.1275436492418589,
        # trial.suggest_float("random_strength", 0.0, 10.0) if trial else 1.0,

        # ["CPU", "GPU"], default: "CPU"
        "task_type": "GPU"
    }

    return model_params


def initialize_seeds(seed: int):
    """ Initialize seeds.

    Args:
        seed (int): The seed.
    """

    numpy.random.seed(seed=seed)
    core.common.random_state(seed)
    random.seed(a=seed)

    logger.info(msg=f"Seed has been set into {seed}.")
