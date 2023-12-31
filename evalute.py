"""
本檔案包含了所有和「評估」有關的函式。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

from logging import getLogger
from typing import Any, Dict

logger = getLogger(name=__name__)


def evalute(parameters: Dict[str, Any]):
    """ Evalute the result generated by the model.

    Args:
        parameters (Dict[str, Any]): The parameters for evaluting.
    """

    logger.info(msg=f"Evaluting has been started.")

    model = parameters["model"]
    data = parameters["train_data"]

    logger.info(
        f"The evalution of input data: \n{model.eval_metrics(data=data, metrics=['F1', 'Precision', 'Recall'])}"
    )

    logger.info(msg=f"Evaluting has been finished.")
