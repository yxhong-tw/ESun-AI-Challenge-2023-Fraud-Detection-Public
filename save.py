"""
本檔案包含了所有和「儲存結果」有關的函式。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

from configparser import ConfigParser
from logging import getLogger
from numpy import count_nonzero, ndarray
from os.path import join
from pandas import DataFrame, Series, concat
from typing import Any, Dict

logger = getLogger(name=__name__)


def save_results(configs: ConfigParser, mode: str, params: Dict[str, Any]):
    """ Save results.

    Args:
        configs (ConfigParser): The config parser.
        mode (str): The mode of this execution.
        params (Dict[str, Any]): The parameters for saving results.
    """

    logger.info(msg=f"Saving results has been started.")

    version = configs.get(section="GENERAL", option="version")

    if mode == "train" or mode == "train_with_optuna" or mode == "cv_train_with_optuna":
        model = params["model"]
        checkpoint_path = join("outputs", "checkpoints",
                                       (version + ".cbm"))
        model.save_model(fname=checkpoint_path)
        logger.info(msg=f"Saving model has been finished.")
    elif mode == "inference":
        origin_unlabeled_df = params["origin_unlabeled_df"]
        prediction = params["prediction"]

        prediction_path = join("outputs", "predictions",
                                       (version + ".csv"))

        write_predictions(origin_unlabeled_df=origin_unlabeled_df,
                          output_path=prediction_path,
                          prediction=prediction)

    logger.info(msg=f"Saving results has been finished.")


def write_predictions(origin_unlabeled_df: DataFrame, output_path: str,
                      prediction: ndarray):
    """ Write predictions.

    Args:
        origin_unlabeled_df (DataFrame): The original unlabeled data.
        output_path (str): The output path.
        prediction (ndarray): The prediction.
    """

    df = origin_unlabeled_df.txkey.copy()
    df = concat(objs=[df, Series(data=prediction, name="pred")], axis=1)

    df.to_csv(path_or_buf=output_path, index=False)

    logger.info(
        msg=
        f"Number of data's prediction is 1: {count_nonzero(a=prediction == 1)}"
    )
    logger.info(msg=f"Number of data: {len(prediction)}")

    logger.info(msg=f"Writing predictions has been finished.")
