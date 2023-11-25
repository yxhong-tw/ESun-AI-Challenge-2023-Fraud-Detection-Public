"""
本檔案包含了所有和「訓練」有關的函式。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

from catboost import cv
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from logging import getLogger
from optuna import Trial, create_study
from os.path import join
from pickle import dump
from sys import stderr, stdout
from typing import Any, Dict

from initialize import initialize_model, initialize_model_parameters

logger = getLogger(name=__name__)


def _cv_train_with_optuna(trial: Trial, params: Dict[str, Any]) -> float:
    """ Do once cross validation training with Optuna.

    Args:
        trial (Trial): The trial object (Optuna required).
        params (Dict[str, Any]): The parameters for training (Not equal to the model_params).

    Returns:
        float: The best test score.
    """

    logger.info(msg=f"Cross validation training with Optuna has been started.")

    model_params = initialize_model_parameters(trial=trial)

    eval_metric = model_params["eval_metric"]
    loss_func_name = model_params["loss_function"]
    train_data = params["train_data"]

    scores = cv(pool=train_data, params=model_params, fold_count=5)

    best_iter = scores[f"test-{eval_metric}-mean"].idxmax()
    best_test_loss = scores[f"test-{loss_func_name}-mean"][best_iter]
    best_test_score = scores[f"test-{eval_metric}-mean"][best_iter]
    best_train_loss = scores[f"train-{loss_func_name}-mean"][best_iter]
    best_train_score = scores[f"train-{eval_metric}-mean"][best_iter]

    logger.info(msg=f"Best Iteration: {best_iter}")
    logger.info(msg=f"Best Iteration's testing loss: {best_test_loss}")
    logger.info(msg=f"Best Iteration's testing score: {best_test_score}")
    logger.info(msg=f"Best Iteration's training loss: {best_train_loss}")
    logger.info(msg=f"Best Iteration's training score: {best_train_score}")

    logger.info(
        msg=f"Cross validation training with Optuna has been finished.")

    return best_test_score


def _train_with_optuna(trial: Trial, params: Dict[str, Any]) -> float:
    """ Do once training with Optuna.

    Args:
        trial (Trial): The trial object (Optuna required).
        params (Dict[str, Any]): The parameters for training (Not equal to the model_params).

    Returns:
        float: The best training loss or validating F1.
    """

    logger.info(msg=f"Training with Optuna has been started.")

    model_params = initialize_model_parameters(trial=trial)
    model = initialize_model(configs=None, model_params=model_params)

    train_data = params["train_data"]
    # validate_data = params["validate_data"]

    model.fit(
        train_data,
        #   use_best_model=True,
        #   eval_set=validate_data,
        log_cout=stdout,
        log_cerr=stderr)

    params["model"] = model

    # f1 = model.get_best_score()["validation"]["F1"]
    loss = model.get_best_score()["learn"]["Logloss"]

    # logger.info(msg=f"The best validation F1: {f1}")
    logger.info(msg=f"The best training loss: {loss}")

    # logger.info(msg=f"The model parameters: \n{model.get_all_params()}")
    logger.info(msg=f"Training with Optuna has been finished.")

    # return f1
    return loss


def cv_train_with_optuna(params: Dict[str, Any]):
    """ Do full cross validation training with Optuna.

    Args:
        params (Dict[str, Any]): The parameters for training.
    """

    study = create_study(direction="maximize")
    study.optimize(
        func=lambda trial: _cv_train_with_optuna(trial=trial, params=params),
        n_trials=30)

    trial = study.best_trial

    file = join("outputs", "studies",
                f"{params['version']}_cv_optuna_study.pkl")
    with open(file=file, mode="wb") as f:
        dump(obj=study, file=f)
        f.close()

    model_params = initialize_model_parameters(trial=None)

    logger.info(msg=f"The best trial: {trial.value}")
    logger.info(msg="The best trial params: ")
    for key, value in trial.params.items():
        model_params[key] = value
        logger.info(f"{key}: {value}")

    params["model"] = initialize_model(configs=None, model_params=model_params)

    train(params=params)


def train(params: Dict[str, Any]):
    """ Do once training.

    Args:
        params (Dict[str, Any]): The parameters for training.
    """

    fit_log = StringIO()

    logger.info(msg=f"Training has been started.")

    model = params["model"]
    train_data = params["train_data"]

    with redirect_stderr(new_target=fit_log), redirect_stdout(
            new_target=fit_log):
        model.fit(train_data, log_cout=stdout, log_cerr=stderr)

    params["model"] = model

    logger.info(msg=f"The training log: \n{fit_log.getvalue()}")
    logger.info(msg=f"Training has been finished.")


def train_with_optuna(params: Dict[str, Any]):
    """ Do full training with Optuna.

    Args:
        params (Dict[str, Any]): The parameters for training.
    """

    study = create_study(direction="maximize")
    study.optimize(
        func=lambda trial: _train_with_optuna(trial=trial, params=params),
        # n_trials=30)
        n_trials=5)

    trial = study.best_trial

    file = join("outputs", "studies", f"{params['version']}_optuna_study.pkl")
    with open(file=file, mode="wb") as f:
        dump(obj=study, file=f)
        f.close()

    logger.info(msg=f"The best trial: {trial.value}")
    logger.info(msg="The best trial params: ")
    for key, value in trial.params.items():
        logger.info(f"{key}: {value}")
