#!/usr/bin/env python3
#
#  MIT License
#
#  Copyright (c) 2021 Ingwersen_erik
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR ABOUT THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
"""
This module contains functions that handle dataframes preparation.
"""
from __future__ import annotations

import difflib
import logging
import re

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from multipledispatch import dispatch
from pandas._typing import Axes
from pandas.util._decorators import doc
from varname import argname

from datatools.core.dtype_cast import iterable_not_string
from datatools.pandas_register import register_dataframe_method


if TYPE_CHECKING:
    pass

__all__ = [
    "clean_names",
    "col_filter",
    "columns_avg",
    "columns_needed",
    "divide_keep_remainder",
    "drop_duplicates_and_log",
    "drop_dup_and_log",
    "drop_na_and_log",
    "find_week",
    "frame_melt",
    "fuzzy_match",
    "get_cnpj",
    "get_df_name",
    "get_missing_ship_to",
    "get_numeric_cols",
    "group_percentage",
    "moving_average",
    "multiply",
    "new_col",
    "new_cols",
    "num_cols",
    "optimize",
    "optimize_floats",
    "optimize_ints",
    "optimize_objects",
    "read_clean_file",
    "round_numeric_columns",
    "save_to_db",
    "select",
    "update_bo_remaining",
    "week_number",
    "week_numbers",
]


def clean_names(
    x_df: pd.DataFrame | Iterable,
    keep_names=True,
    attr_name="column_map",
) -> pd.DataFrame | dict[Axes, Axes]:
    """
    Clean dataframe or list of column names. The method cleans column names by:

        1. Making everything lower case
        2. Replace " " by '_'
        3. Removing special character '$'
        4. removing parenthesis '(' or ')'

    Parameters
    ----------
    x_df : pd.DataFrame | Iterable
        Dataframe, or list of names or name of columns to be cleaned.
    keep_names : bool, defaults to True
        If True, save column names mapping as dataframe attribute.
        Valid only if :param:`x_df` consists of a dataframe.
    attr_name : str, defaults to 'column_map'
        Attribute name to use if `keep_names` is True.

    Returns
    -------
    x_df : pd.DataFrame | dict[types.Axes, types.Axes]
        Dataframe with cleaned columns, or dictionary with old and new
        clean_names.

    Notes
    -----
    If using function to clean column names, and :param:`x_df` is not a
    dataframe, column renaming is not applied automatically and operation
    must be performed manually later.

    .. versionadded:: 1.2.0
        Optionally, original names can be saved as attributes to dataframe.
        This allows for easier conversion when saving back dataframes as
        outputs.

    .. versionadded:: 1.2.1
        Broadened input parameter :param:`x_df` supported datatype,
        to allow for conversion of lists of column names or a column name.
    """
    # Careful upon making changes the "If/Else" conditions that follows,
    # as their order of appearance matter. Strings are also iterables, but
    # when searching for iterables, the wanted ones are lists, sets and
    # other objects alike.
    if isinstance(x_df, pd.DataFrame):
        column_names = x_df.columns
    elif isinstance(x_df, str):
        keep_names = False
        column_names = [x_df]
    elif isinstance(x_df, Iterable):
        keep_names = False
        column_names = x_df
    else:
        raise TypeError(
            "Invalid x_df type. Function supports dataframes, strings or "
            f"iterable objects, not {type(x_df)}"
        )
    column_maps = {
        col: (
            str(col)
            .lower()
            .replace(" ", "_")
            .replace("  ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("$", "")
            .replace("/", "_")
            .replace("?", "")
            .replace("+", "_")
            .replace("__", "_")
            .replace("-", "_")
            .replace("\t", "_")
            .replace("\n", "_")
        )
        if not str(col).isnumeric()
        else col
        for col in column_names
    }

    if isinstance(x_df, pd.DataFrame):
        if keep_names:
            x_df.attrs[attr_name] = column_maps
        return x_df.rename(columns=column_maps)
    return column_maps


@doc(clean_names)
@register_dataframe_method
def clean_colnames(
    df: pd.DataFrame,
    keep_names=True,
    attr_name="column_map",
) -> pd.DataFrame:
    """Dataframe method, that applies :func:`clean_names` dataframe columns."""
    return clean_names(df, keep_names=keep_names, attr_name=attr_name)


def num_cols(
    df: pd.DataFrame,
    min_value: int | None = None,
    max_value: int | None = None,
    periods: int | None = None,
) -> list:
    """
    Get a list of numeric column names from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to get numeric column names from.
    min_value : int | None, default None
        Minimum value to include in the list.
        If None, no minimum value is used.
    max_value : int | None, default None
        Maximum value to include in the list.
        If None, no maximum value is used.
    periods : int | None, default None
        Represents the total amount of periods to be included. Here's what
        each range of values (positive, None and negative) represent:

            * ``None`` - no limit is used.

            * ``negative`` - values are included from the end of the list.

            * ``positive`` - values are included from the beginning of the list.

    Returns
    -------
    list
        List of numeric column names.
    """
    numeric_cols = [col for col in df.columns if str(col).isnumeric()]

    if min_value:
        numeric_cols = [col for col in numeric_cols if int(col) >= int(min_value)]

    if max_value:
        numeric_cols = [col for col in numeric_cols if int(col) <= int(max_value)]

    numeric_cols = sorted(numeric_cols)

    if periods:
        if periods < 0:
            numeric_cols = numeric_cols[periods:]
        else:
            numeric_cols = numeric_cols[:periods]

    return numeric_cols


def columns_needed(week: str | int, weeks: list, periods=4):
    """
    Get a list of columns needed for a given week.

    Parameters
    ----------
    week : str | int
        Week to get columns for.
    weeks : list
        List of weeks to get columns for.
    periods : int, default 4
        Amount of periods to include in the list.

    Returns
    -------
    list
        List of columns needed.
    """
    return [wk for wk in sorted(weeks) if int(wk) >= int(week)][:periods]


def moving_average(
    df: pd.DataFrame,
    group_cols: list[str],
    week_col: str,
    var_col: str,
    colname: str,
    periods: int = 4,
) -> pd.DataFrame:
    """
    Compute moving average, performing roll operation on groups of row values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        List of columns to group by.
    week_col : str
        Column name to use as week.
    var_col : str
        Column name to use as variable.
    colname : str
        Column name to use as output.
    periods : int
        Period to consider for the moving average. Defaults to 4.

    Returns
    -------
    pd.DataFrame
        Dataframe with added column
    """
    group_cols = list(
        set([group_cols] if isinstance(group_cols, str) else group_cols) - {week_col}
    )
    assert len(group_cols) > 0
    df[colname] = (
        df.sort_values(group_cols)
        .iloc[::-1]
        .groupby(group_cols, as_index=False)[var_col]
        .transform(lambda x: x.rolling(window=periods).mean())[var_col]
    )
    return df


@dispatch(pd.DataFrame, str, float)
def col_filter(df: pd.DataFrame, column_name: str, value: Any) -> pd.DataFrame:
    """
    Filter rows by a column name and values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column_name : str
        Column name to filter by.
    value : Any
        Value to filter by.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if value == "nan":
        return df.query(f"{column_name} == {column_name}")
    elif value == "isna":
        return df.query(f"{column_name} != {column_name}")
    return df.query(f"{column_name} == @value")


def select(
    df: pd.DataFrame,
    columns: str | list[str],
    inverse: bool | None = False,
) -> pd.DataFrame:
    """
    Filter df by selecting subset of columns.

    When inverse is True, function will select all columns from
    df excluding columns specified at “columns“.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : str | List[str]
        Columns to select.
    inverse : bool, default False
        If True, select all columns from df excluding columns specified at
        “columns“.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if inverse:
        columns = list(set(df.columns) - set(columns))
    return df[columns]


def new_col(
    df: pd.DataFrame,
    colname: str,
    value: Any,
    if_exists: str | None = "replace",
) -> pd.DataFrame:
    """
    Add a new column to dataframe with default specified default value.

    Value can be an array of values for each row (array must have same
    length as df) or some default value.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    colname : str
        Column name to add.
    value : Any
        Value to add to column.
    if_exists : str, default 'replace'
        If 'replace', replace the existing column with the new value.

    Returns
    -------
    pd.DataFrame
        Dataframe with the added column.
    """
    if if_exists not in ["replace", "skip"]:
        logging.warning(
            f"{if_exists} is not a valid option for argument "
            'if_exists. Defaulting to "replace"'
        )
        if_exists = "replace"

    if colname not in df.columns or if_exists == "replace":
        df[colname] = value
    return df


def new_cols(
    df: pd.DataFrame, colnames: list[str], value: Any, **kwargs: Any
) -> pd.DataFrame:
    """
    Add new columns to DataFrame with a constant value.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    colnames : List[str]
        List of column names to add.
    value : Any
        Value to assign to all rows of new columns.
    **kwargs : Any
        Keyword arguments to pass to pd.DataFrame.assign().

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns.
    """
    for col in colnames:
        df = df.pipe(new_col, col, value, **kwargs)
    return df


# noinspection PyShadowingNames
def find_week(
    row: pd.Series,
    colname: str = "year_week_base_scm",
    new_col: str = "iod_year_week",
) -> pd.Series:
    """
    Find week number within "year_week_base_scm" column from WOS_HST dataframe.

    Parameters
    ----------
    row : pd.Series
        Row to perform extraction.
    colname : str
        Column name to try to extract week number from. Defaults to
        "year_week_base_scm".
    new_col: str
        Column name to add extracted week number to. Defaults to "week".

    Returns
    -------
    pd.Series
        Row with extracted week number.

    Examples
    --------
    >>> row = pd.Series({'year_week_base_scm': '2021W20'})
    >>> find_week(row)
    year_week_base_scm     2021W20
    iod_year_week         20202120
    dtype: object
    >>> row = pd.Series({'year_week_base_scm': '202110'})
    >>> find_week(row)
    year_week_base_scm    202110
    iod_year_week         202110
    dtype: object
    """
    try:
        split_year_week = str(row[colname]).lower().split("w")
        if len(split_year_week) == 1:
            row[new_col] = f"{split_year_week[0]}"
        else:
            year = split_year_week[0]
            week = split_year_week[1]
            row[new_col] = f"20{year}{int(week):02d}"

    except Exception as e:
        logging.error(f"Could not convert week: {row}. \n Error Message: {e}")
        row[new_col] = "NOT_FOUND"

    return row


def multiply(
    df: pd.DataFrame,
    scalar_col: str,
    var_cols: str | list[str, ...],
    decimals: int | None = 0,
) -> pd.DataFrame:
    """
    Multiply a column(s) by another column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    scalar_col : str
        Name of column to be multiplied by
    var_cols : str | List[str, ...]
        Name of column(s) to be multiplied
    decimals : Optional[int], defaults to 0
        Amount of decimals to round to. Defaults to 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with multiplied columns
    """
    var_cols = [var_cols] if isinstance(var_cols, str) else var_cols
    _var_cols = [col for col in var_cols if col in df.columns]
    df.loc[:, _var_cols] = (
        df.loc[:, _var_cols].multiply(df.loc[:, scalar_col], axis=0).round(decimals)
    )
    return df


def group_percentage(
    df: pd.DataFrame,
    group_cols: Axes | list[Axes],
    var_cols: Axes | list[Axes],
    col_names: Axes | list[Axes],
) -> pd.DataFrame:
    """
    Find the percentage of each row compared to the sum of its group.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be processed.
    group_cols : types.Axes | List[types.Axes]
        The columns to be used for grouping.
    var_cols : types.Axes | List[types.Axes]
        The columns to be used for finding the proportional value.
    col_names : types.Axes | List[types.Axes]
        The column names to be used for the output dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe with the proportional values
    """
    df[col_names] = df.groupby(group_cols)[var_cols].transform(
        lambda x: x / float(x.sum())
    )
    return df


def columns_avg(
    df: pd.DataFrame,
    var_cols: Axes | list[Axes],
    col_name: Axes,
) -> pd.DataFrame:
    """
    Find average value of combination of columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be processed.
    var_cols : types.Axes | List[types.Axes]
        The columns to be used for finding the average value.
    col_name : types.Axes
        The column name to be used for the output dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe with the average values.
    """
    df[col_name] = df[var_cols].mean(axis=1)
    return df


def frame_melt(
    df: pd.DataFrame,
    value_vars: Axes | list[Axes],
    var_name: Axes,
    value_name: Axes,
    id_vars: list[Axes] | None = None,
) -> pd.DataFrame:
    """
    Melt dataframe, passing either ``value_vars`` columns or ``id_vars``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be processed.
    value_vars : types.Axes | List[types.Axes]
        The columns to be used for melting.
    var_name : types.Axes
        The column name to be used for the variable column.
    value_name : types.Axes
        The column name to be used for the value column.
    id_vars : List[types.Axes] | None
        The columns to be used for identifying the rows.

    Returns
    -------
    pd.DataFrame
        The dataframe with the melted values.

    Examples
    --------
    >>> _df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
    ...                    'B': {0: 1, 1: 3, 2: 5},
    ...                    'C': {0: 2, 1: 4, 2: 6}})
    >>> res = frame_melt(_df, value_vars=['B'], var_name='myVarname',
    ... value_name='myValname')
    >>> res[sorted(res.columns)] # doctest: +NORMALIZE_WHITESPACE
       A  C  myValname myVarname
    0  a  2          1         B
    1  b  4          3         B
    2  c  6          5         B
    """
    if value_vars:
        id_vars = set(df.columns) - set(value_vars)
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)


def get_df_name(df_to_find_name: pd.DataFrame) -> str:
    """
    Return the name of a dataframe.

    To retrieve the dataframe's name, function loops through all
    global declared variables, compares type of the global variable
    with :param:`df_to_find_name`, then verifies if the global variable is
    equal to :param:`df_to_find_name` and finally, filters out any global
    variable name that starts with '_' (private variables).

    Parameters
    ----------
    df_to_find_name: pd.DataFrame
        The dataframe for which the name is to be found.

    Returns
    -------
    Optional[str]
        The name of the dataframe, if it exists.
    """
    if isinstance(df_to_find_name, pd.DataFrame) and "name" in df_to_find_name.attrs:
        return df_to_find_name.attrs["name"]
    return argname("df_to_find_name")


# noinspection PyBroadException
def save_to_db(
    df: pd.DataFrame,
    name: str | None = None,
    engine: None | str = None,
    float_precision: None | int = 4,
) -> pd.DataFrame:
    """
    Save a DataFrame to the database.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save
    name : str
        The name of the table to save the DataFrame to. If no name is provided,
        the DataFrame is saved to table with same name as the input DataFrame
    engine : str | None, optional
        The database engine to use. If None, the default engine is used.
    float_precision : Optional[int], defaults to 4.
        The amount of decimal places to round float columns from :param:`df`
        to. If None, no rounding is performed.

    Returns
    -------
    pd.DataFrame
        The DataFrame that was saved to the database.
    """
    try:
        name = name or get_df_name(df)
        if not hasattr(df, "to_sql"):
            raise ValueError(f'"df" must be a dataframe, not {type(df)}')
        if float_precision:
            df = df.round(float_precision)
        df.to_sql(name, engine, if_exists="replace")
        logging.info(
            "Saved table %s to local database with a total of %s rows",
            name,
            f"{df.shape[0]:,}",
        )
    except ValueError as error:
        logging.critical(error)

    except Exception as error:
        logging.critical(f"Could not save {name} to database. Error message: {error}")
    return df


def week_number(my_date: datetime) -> int | pd.NaT:
    """
    Determine week number that ``my_date``, falls on.

    Parameters
    ----------
    my_date: datetime
        A datetime object.

    Returns
    -------
    int | pd.NaT
        The week number or pd.NaT when my_date is not of type datetime.

    Notes
    -----
    If my_date is not convertible to a datetime object, the function returns
    pd.NaT
    """
    return pd.to_datetime(my_date, errors="coerce").isocalendar()[1]


def week_numbers(my_date_list: list) -> list:
    """
    Transform a list of datetime objects into a list of week numbers.

    Parameters
    ----------
    my_date_list: list
        A list of datetime objects.

    Returns
    -------
    list
        A list of week numbers.

    Examples
    --------
    >>> week_numbers([datetime(2019, 1, 5), datetime(2019, 2, 3), \
    datetime(2019, 7, 1), datetime(2019, 12, 31)])
    [1, 5, 27, 1]

    >>> week_numbers([datetime(2019, 1, 1), datetime(2019, 1, 2), \
    datetime(2019, 1, 3), datetime(2019, 1, 4)])
    [1, 1, 1, 1]

    >>> week_numbers([datetime(2018, 12, 31), datetime(2019, 1, 1), \
    datetime(2019, 1, 2), datetime(2019, 1, 3), datetime(2019, 1, 4)])
    [1, 1, 1, 1, 1]
    """
    return [week_number(i) for i in my_date_list]


def get_numeric_cols(_df: pd.DataFrame) -> list[str]:
    """
    Return a list of numeric column names in a pandas dataframe.

    Function is useful to identify columns that represent week numbers.

    Parameters
    ----------
    _df : pd.DataFrame
        A pandas dataframe.

    Returns
    -------
    list[str]
        A list of numeric column names in a pandas dataframe.

    Raises
    ------
    IndexError
        If no numeric column names are found.
    """
    numeric_cols = [col for col in _df.columns if str(col).isnumeric()]

    if len(numeric_cols) != 0:
        return numeric_cols
    raise IndexError(f"No numeric column found. Columns: {_df.columns}")


def round_numeric_columns(
    df: pd.DataFrame | pd.Series, n: int = 2
) -> pd.DataFrame | pd.Series:
    """
    Round numeric columns to “n“ decimal places.

    Parameters
    ----------
    df : pd.DataFrame | pd.Series
        Dataframe or pandas series to round numeric column(s)
    n : int, defaults to 2
        Amount of decimal places to round to.
        Must be greater than or equal to 0

    Returns
    -------
    pd.DataFrame | pd.Series
        Dataframe or pandas series, with rounded numeric column(s).
    """
    if isinstance(df, pd.Series):
        return (
            df.fillna(-1)
            .apply(
                lambda x: int(x)
                if n == 0 and not pd.isna(x) or x == ""
                else round(x, n)
            )
            .replace(-1, "")
        )
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: round(x, n))
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert integer columns to optimize memory usage of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to optimize.

    Returns
    -------
    pd.DataFrame
        The optimized DataFrame.
    """
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float columns to optimize memory usage of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to optimize.

    Returns
    -------
    pd.DataFrame
        The optimized DataFrame.
    """
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: list[str]) -> pd.DataFrame:
    """
    Convert object columns to optimize memory usage of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to optimize.
    datetime_features : List[str]
        A list of columns of the DataFrame that are datetime features.

    Returns
    -------
    pd.DataFrame
        The optimized DataFrame.
    """
    for col in df.select_dtypes(include=["object"]):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize(df: pd.DataFrame, datetime_features=None):
    """
    Convert column data types to optimize memory usage of pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to optimize.
    datetime_features : List[str]
        A list of columns of the DataFrame that are datetime features.

    Returns
    -------
    pd.DataFrame
        The optimized DataFrame.
    """
    if datetime_features is None:
        datetime_features = []
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def update_bo_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update "bo_remaining" and "bo_qty" columns with their min values by "SO".

    Function is executed after solution space creation, and is used to update
    the total available quantities, after alignment criteria discount.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be updated.

    Returns
    -------
    pd.DataFrame
        The updated dataframe.

    Examples
    --------
    >>> # noinspection PyShadowingNames
    >>> df = pd.DataFrame({
    ...     'so': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'bo_remaining': [1, 2, 3, 4, 5, 6]
    ... })
    >>> update_bo_remaining(df) # doctest: +NORMALIZE_WHITESPACE
       so  bo_remaining  bo_qty
    0  A              1       1
    1  A              1       1
    2  A              1       1
    3  B              4       4
    4  B              4       4
    5  B              4       4
    """
    df["bo_remaining"] = df["bo_qty"] = df.groupby("so")["bo_remaining"].transform(
        lambda x: x.min()
    )
    return df


def drop_duplicates_and_log(
    df: pd.DataFrame, custom_message: str = "", *args, **kwargs
):
    """
    Drop duplicates and log the number of duplicates removed.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be updated.
    custom_message : str, optional
        A custom message to be logged. Defaults to ''
    *args : List[str]
        Arguments of :meth:`pandas.DataFrame.drop_duplicates`.
    **kwargs : Dict[str, Any]
        The keyword arguments of :meth:`pandas.DataFrame.drop_duplicates`.

    Returns
    -------
    pd.DataFrame
        The updated dataframe.

    Examples
    --------
    >>> # noinspection PyShadowingNames
    >>> df = pd.DataFrame({'a': [1, 1, 1, 2, 2, 2], 'b': [1, 2, 3, 1, 2, 3]})
    >>> df
       a  b
    0  1  1
    1  1  2
    2  1  3
    3  2  1
    4  2  2
    5  2  3
    >>> drop_duplicates_and_log(df, subset=['a']
    ... ) # doctest: +NORMALIZE_WHITESPACE
        a	b
    0	1	1
    3	2	1
    """
    num_rows = df.shape[0]
    df = df.drop_duplicates(*args, **kwargs)
    num_duplicates = num_rows - df.shape[0]

    if num_duplicates > 0:
        if custom_message:
            custom_message += (
                ". "
                if not any(
                    [
                        custom_message.endswith(". "),
                        custom_message.endswith("."),
                    ]
                )
                else ""
            )
        try:
            name = get_df_name(df)
        except (ValueError, Exception):
            name = "dataframe"
        logging.warning(
            f"{custom_message}Dropped {num_duplicates:,} duplicates from"
            f" {name} (Percentual {100 * num_duplicates / num_rows:.2f}%)"
        )
    return df


@doc(drop_duplicates_and_log)
@register_dataframe_method
def drop_dup_and_log(
    df: pd.DataFrame,
    name: str = "df",
    custom_message: str = "",
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Pandas dataframe accessor for method :func:`drop_duplicates_and_log`."""
    if name:
        df.attrs["name"] = str(name)
    return drop_duplicates_and_log(df, *args, custom_message=custom_message, **kwargs)


@doc(pd.DataFrame.dropna)
@register_dataframe_method
def drop_na_and_log(
    df: pd.DataFrame,
    custom_message: str = "",
    df_name: str = "",
    *args,
    **kwargs,
) -> pd.DataFrame:
    if df_name:
        df.attrs["name"] = str(df_name)
    return _drop_na_and_log(df, *args, custom_message=custom_message, **kwargs)


def _drop_na_and_log(
    df: pd.DataFrame,
    custom_message: str = "",
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Drop rows with NaN values and log the number of rows removed.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be updated
    *args : List[str]
        Arguments of :meth:`pandas.DataFrame.dropna`
    custom_message : str, optional
        A custom message to be logged. Defaults to ''
    **kwargs : Dict[str, Any]
        The keyword arguments of :meth:`pandas.DataFrame.dropna`.

    Returns
    -------
    pd.DataFrame
        The updated dataframe.
    """
    num_rows = df.shape[0]
    if "subset" in kwargs.keys() and not iterable_not_string(kwargs["subset"]):
        kwargs["subset"] = [kwargs["subset"]]
    df = df.dropna(*args, **kwargs)
    num_na = num_rows - df.shape[0]

    if num_na > 0:
        name = get_df_name(df)
        log_dropped_nan_values(num_na, num_rows, name, custom_message)
    return df


def log_dropped_nan_values(
    num_na: int, num_rows: int, name: str = "df", custom_message: str = ""
) -> None:
    """
    Logs the number of rows removed due to NaN values.

    By default, logs are written as warnings.

    Parameters
    ----------
    num_na : int
        The number of NaN values removed
    num_rows : int
        The total number of rows
    name : str, optional
        The name of the dataframe. Defaults to 'df'
    custom_message : str, optional
        A custom message to be logged. Defaults to ''
    """
    if custom_message:
        custom_message += (
            ". "
            if not any(
                [
                    custom_message.endswith(". "),
                    custom_message.endswith("."),
                ]
            )
            else ""
        )
    logging.warning(
        f"{custom_message}Dropped {num_na:,} rows with NaN values "
        f"from {name} (Percentual {100 * num_na / num_rows:.2f}%)"
    )


class FuzzyMatch:
    """Performs fuzzy match operation."""

    _return_types = {"all", "first", "last"}
    _return_type = "first"

    @property
    def return_type(self) -> Any:
        if self._return_type == "all":
            return lambda x: x
        elif self._return_type == "first":
            return lambda x: x[0]
        return lambda x: x[-1]

    @return_type.setter
    def return_type(self, value: str):
        if value not in self._return_types:
            raise ValueError(
                "%s is not a valid option. Valid options: %s"
                % value
                % ", ".join(str(option) for option in self._return_types)
            )
        self._return_type = value

    def match(
        self, possible_vals: Iterable, value: str, _return_type: str = None
    ) -> Any:
        """
        Performs a fuzzy match on the possible values.

        Parameters
        ----------
        possible_vals : Iterable
            The values to match against.
        value : str
            The value to match.
        _return_type : str
            The type of return value to return.

        Returns
        -------
        matched_value : Any
            The matched value.
        """
        if _return_type:
            self.return_type = _return_type
        if isinstance(possible_vals, str):
            possible_vals = [possible_vals]
        _possible_vals = {str(k): k for k in possible_vals}
        _value = list(
            map(
                lambda x: _possible_vals[x],
                difflib.get_close_matches(value, list(_possible_vals.keys())),
            )
        )
        try:
            return self.return_type(_value)
        except IndexError as error:
            logging.exception(
                error, extra={"value": value, "possible_vals": possible_vals}
            )
            raise IndexError(
                f'Could not find match for "<{value}>" at {possible_vals}',
            )


def fuzzy_match(possible_vals: Iterable, value: Any, return_type: str = "first") -> Any:
    """
    Fuzzy match a value to a list of possible values

    Parameters
    ----------
    possible_vals : Iterable
        List of possible values or value
    value : Any
        Value to fuzzy match
    return_type : str {'all', 'first', or 'last'}, optional.
        How to handle multiple matches. Defaults to 'first'.

            * 'all' - return list with all matches
            * 'first' - return first match
            * 'last' - return last match

    Returns
    -------
    Any
        The matched value.

    Raises
    ------
    ValueError
        If no value found.

    Examples
    --------
    >>> fuzzy_match([10, 20, 30], '0', 'last')
    10
    >>> fuzzy_match([10, 20, 30], '0', 'first')
    30
    >>> fuzzy_match([10, 20, 30], '0', 'all')
    [10, 20, 30]
    >>> fuzzy_match([10, 20, 30], 'foo', 'all') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    IndexError: Could not find match for "<foo>" at [10, 20, 30]
    """
    return FuzzyMatch().match(possible_vals, str(value), return_type)


def divide_keep_remainder(
    row: pd.Series,
    result_col: str,
    numerator_col: str,
    denominator: str | int,
) -> pd.Series:
    """
    Divide the numerator by the denominator and keep the remainder in the
    last division.

    Parameters
    ----------
    row : pd.Series
        The row from which to get the numerator and denominator.
    result_col : str
        The column name to which to write the result.
    numerator_col : str
        The column name from which to get the numerator.
    denominator : Union[str, int]
        The denominator.

    Returns
    -------
    pd.Series
        The row with the result.

    Examples
    --------
    >>> # noinspection PyShadowingNames
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df.apply(
    ...     divide_keep_remainder,
    ...     result_col='c',
    ...     numerator_col='b',
    ...     denominator='a',
    ...     axis=1,
    ... )  # doctest: +NORMALIZE_WHITESPACE
       a  b          c
    0  1  4        [4]
    1  2  5     [2, 3]
    2  3  6  [2, 2, 2]
    >>> df.apply(
    ...     divide_keep_remainder,
    ...     result_col='c',
    ...     numerator_col='b',
    ...     denominator=3,
    ...     axis=1,
    ... )  # doctest: +NORMALIZE_WHITESPACE
       a  b          c
    0  1  4  [1, 1, 2]
    1  2  5  [1, 1, 3]
    2  3  6  [2, 2, 2]
    >>> df.apply(
    ...     divide_keep_remainder,
    ...     result_col='c',
    ...     numerator_col='b',
    ...     denominator=3,
    ...     axis=1,
    ... ).explode(column='c')  # doctest: +NORMALIZE_WHITESPACE
       a  b  c
    0  1  4  1
    0  1  4  1
    0  1  4  2
    1  2  5  1
    1  2  5  1
    1  2  5  3
    2  3  6  2
    2  3  6  2
    2  3  6  2

    Notes
    -----
    Use ``pandas.DataFrame.explode(column=result_col)`` to convert list of
    values into rows.
    """
    numerator = row[numerator_col]
    denominator = get_denominator(row, denominator)
    division_no_remainder = numerator // denominator
    remainder = numerator % denominator
    division_with_remainder = division_no_remainder + remainder

    row[result_col] = [
        *[division_no_remainder for _ in range(denominator - 1)],
        division_with_remainder,
    ]
    return row


@dispatch(float)
def get_denominator(denominator: float | int) -> int:
    """
    If denominator is a float or int, then return the integer value.

    If denominator is a float, then round it to the nearest integer. If the
    rounded value is 0, then return 1.

    Parameters
    ----------
    denominator : Union[float, int]
        The denominator.

    Returns
    -------
    int
        The denominator as integer.
    """
    return int(max(round(denominator, 0), 1))


@dispatch(pd.Series, str)
def get_denominator(row: pd.Series, denominator: str) -> int:
    """
    If denominator is a column name, then get the value from the row.

    If the denominator is a float, then round it to the nearest integer. If the
    rounded value is 0, then return 1.

    Parameters
    ----------
    row : pd.Series
        The row from which to get the denominator.
    denominator : str
        The column name from which to get the denominator.

    Returns
    -------
    int
        The denominator.
    """
    return int(max(round(row[denominator], 0), 1))


def read_clean_file(filepath: str | Path, log: bool = True) -> pd.DataFrame:
    """
    Read an Excel or csv file, clean column names, and return a DataFrame.

    When reading csv files, if ";" is found inside column names, then reads
    the csv file again, but using ";" as delimiter.

    Parameters
    ----------
    filepath : str | Path
        The path to the Excel file
    log : bool, defaults to True
        Whether to display information about the read file inside the logs

    Returns
    -------
    pd.DataFrame
        The DataFrame.

    Raises
    ------
    ValueError
        If the filepath is not an Excel or csv file.
    FileNotFoundError
        If the filepath does not exist.
    """
    _filepath = Path(filepath)
    if _filepath.is_file():
        name = _filepath.with_suffix("").name
        if _filepath.suffix.startswith(".xls"):
            df = pd.read_excel(_filepath)
        elif _filepath.suffix == ".csv":
            df = _read_csv(_filepath)
        else:
            raise ValueError(f"Unsupported file type: {_filepath.suffix}")
        return _prepare_df(df, log, name)
    raise FileNotFoundError(f"File not found: {filepath}")


def _read_csv(_filepath: Path) -> pd.DataFrame:
    """
    Read a csv file and return a DataFrame.

    Parameters
    ----------
    _filepath : Path
        The path to the csv file.

    Returns
    -------
    pd.DataFrame
        The DataFrame.
    """
    df = pd.read_csv(_filepath)
    if ";" in df.columns[0]:
        df = pd.read_csv(_filepath, delimiter=";")
    return df


def _prepare_df(df: pd.DataFrame, log: bool = True, name: str = "") -> pd.DataFrame:
    """
    Clean column names, and return a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame
    log : bool, defaults to True
        Whether to display information about the read file inside the logs
    name : str, defaults to ''
        The name of the dataframe.

    Returns
    -------
    pd.DataFrame
        The DataFrame.
    """
    df = df.pipe(clean_names)
    log_msg = "Read table"
    if name:
        df.attrs["name"] = name
        log_msg = f"Read file: {name}"
    if log:
        nrows, ncols = df.shape
        logging.info("%s with %s rows and %s columns", log_msg, nrows, ncols)
    return df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
