"""
本檔案包含了所有和「OpenFE」有關的函式。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

from logging import getLogger
from openfe import OpenFE, get_candidate_features, transform
from os.path import join
from pandas import DataFrame
from pickle import dump
from typing import Any, Dict

logger = getLogger(__name__)


def get_and_save_candidate_features(params: Dict[str, Any]):
    """ Get and save candidate features.

    Args:
        params (Dict[str, Any]): The parameters for getting candidate features.
    """

    logger.info(msg="Getting candidate features has been started.")

    candidate_features = get_candidate_features(
        numerical_features=params["numerical_features"],
        categorical_features=params["categorical_features"],
        ordinal_features=params["ordinal_features"])

    logger.info(msg="Getting candidate features has been finished.")

    file = join(params["directory"],
                        f"{params['version']}_candidate_features.pkl")

    logger.info(msg="Saving candidate features has been started.")

    with open(file=file, mode="wb") as f:
        dump(obj=candidate_features, file=f)
        f.close()

    logger.info(msg="Saving candidate features has been finished.")

    params["candidate_features"] = candidate_features


def get_and_save_features(params: Dict[str, Any]):
    """ Get and save features.

    Args:
        params (Dict[str, Any]): The parameters for getting features.
    """

    logger.info(msg="Getting features has been started.")

    ofe = OpenFE()
    features = ofe.fit(data=params["train_data"]["x"],
                       label=params["train_data"]["y"],
                       task=params["task"],
                       candidate_features_list=params["candidate_features"],
                       categorical_features=params["categorical_features"],
                       metric=params["metric"],
                       n_data_blocks=params["n_data_blocks"],
                       min_candidate_features=params["min_candidate_features"],
                       stage2_params=params["stage2_params"],
                       n_jobs=params["n_jobs"],
                       seed=params["seed"])

    logger.info(msg="Getting features has been finished.")

    file = join(params["directory"],
                        f"{params['version']}_features.pkl")

    logger.info(msg="Saving features has been started.")

    with open(file=file, mode="wb") as f:
        dump(obj=features, file=f)
        f.close()

    logger.info(msg="Saving features has been finished.")

    file = join(params["directory"], f"{params['version']}_openfe.pkl")

    logger.info(msg="Saving openfe model has been started.")

    with open(file=file, mode="wb") as f:
        dump(obj=ofe, file=f)
        f.close()

    logger.info(msg="Saving openfe model has been finished.")


def get_and_save_x_data(params: Dict[str, Any]):
    """ Get and save x data.

    Args:
        params (Dict[str, Any]): The parameters for getting x data.
    """

    if len(params["features"]) > 30:
        params["features"] = params["features"][:30]

    logger.info(msg="Transforming data has been started.")

    train_x, unlabeled_x = transform(X_train=params["train_data"]["x"],
                                     X_test=params["unlabeled_data"]["x"],
                                     new_features_list=params["features"],
                                     n_jobs=params["n_jobs"])

    validate_x, _ = transform(X_train=params["validate_data"]["x"],
                              X_test=DataFrame(),
                              new_features_list=params["features"],
                              n_jobs=params["n_jobs"])

    logger.info(msg="Transforming data has been finished.")
    logger.info(msg="Saving transformed data has been started.")

    file = join(params["directory"], f"{params['version']}_train_x.h5")
    train_x.to_hdf(path_or_buf=file,
                   key="data",
                   mode="w",
                   format="table",
                   index=False)

    file = join(params["directory"],
                        f"{params['version']}_unlabeled.h5")
    unlabeled_x.to_hdf(path_or_buf=file,
                       key="data",
                       mode="w",
                       format="table",
                       index=False)

    file = join(params["directory"],
                        f"{params['version']}_new_public_x.h5")
    validate_x.to_hdf(path_or_buf=file,
                      key="data",
                      mode="w",
                      format="table",
                      index=False)

    logger.info(msg="Saving transformed data has been finished.")


def openfe_inference(params: Dict[str, Any]):
    """ Do OpenFE inference.

    Args:
        params (Dict[str, Any]): The parameters for OpenFE inference.
    """

    get_and_save_x_data(params=params)


def openfe_train(params: Dict[str, Any]):
    """ Do OpenFE training.

    Args:
        params (Dict[str, Any]): The parameters for OpenFE training.
    """

    get_and_save_candidate_features(params=params)
    get_and_save_features(params=params)
