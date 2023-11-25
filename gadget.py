"""
本檔案包含了數個「獨立測試」的函式，和主程式執行無關。
詳細功能與輸入、輸出請見各函式的 docstring。
"""

from os.path import join
from pandas import DataFrame, read_csv, concat
from typing import Tuple
from ydata_profiling import ProfileReport

from utils import get_df


def check_data_nan():
    """ Check the given data has NaN values or not.
    """

    data = read_csv('outputs/predictions/HAHAHAA-10.csv')
    print(f"The length of the given data: {len(data)}")

    print("The given data has below columns: ")
    for column_name in data.columns.values:
        print(column_name)

        nan_values_number = data[column_name].isna().sum()
        if nan_values_number > 0:
            print(f"Notice: {column_name} has {nan_values_number} NaN values.")


def concat_data(dfs: Tuple[DataFrame]) -> DataFrame:
    """ Concatenate the given dataframes.

    Args:
        dfs (Tuple[DataFrame]): The given dataframes.

    Returns:
        DataFrame: The concatenated dataframe.
    """

    return DataFrame(concat(objs=dfs, axis=0))


def concat_and_save_data():
    """ Concatenate the given dataframes and save the result.
    """

    dfs = load_data_x()
    df = concat_data(dfs)
    save_data(df=df)


def do_eda():
    """ Analyze the data and generate a report.
    """

    df = read_csv(filepath_or_buffer='data/train.csv')
    fraud_df = df.loc[df.label == 1]

    report = ProfileReport(df=df, title='Data Analysis (training.csv)')
    report.to_file(
        output_file=
        'outputs/EDAs/ydata-profiling/data_analysis (training.csv).html')

    report = ProfileReport(df=fraud_df,
                           title='Data Analysis (only fraud data)')
    report.to_file(
        output_file='outputs/EDAs/ydata-profiling/fraud_data_analysis.html')


def get_small_data():
    """ Get small data for testing.
    """

    data = get_df(data_name="new_public")
    data = data.head(100000)

    # Need to change the save file name in save_data() function.
    save_data(df=data)


def load_data_x() -> Tuple[DataFrame]:
    """ Load two dataframes.

    Returns:
        Tuple[DataFrame]: The two dataframes.
    """

    data_1 = get_df(data_name="old_train")
    data_2 = get_df(data_name="new_public")

    return data_1, data_2


def save_data(df: DataFrame):
    """ Save the concatenated dataframe.

    Args:
        df (DataFrame): The concatenated dataframe.
    """

    file = join("data", "train.csv")
    df.to_csv(path_or_buf=file, index=False)


if __name__ == "__main__":
    """ Main function.
    """

    # check_data_nan()
    # concat_and_save_data()
    # do_eda()
    # get_small_data()
