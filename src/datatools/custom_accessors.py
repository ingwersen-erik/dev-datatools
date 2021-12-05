#
#  MIT License
#
#  Copyright (c) 2021 Erik Ingwersen
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
""""
Module defines dataframe accessors that extend pandas dataframe
functionalities.

These are the added functionalities:
    * :fun:`safe_merge` - merge dataframes with different index data types,
      by first converting both sides indexes to same allowed dtype.
"""
from __future__ import annotations

import logging

import pandas as pd

# noinspection PyProtectedMember
from pandas._typing import FrameOrSeriesUnion
from pandas._typing import IndexLabel
from pandas._typing import Suffixes
from pandas.core.common import maybe_make_list
from pandas.errors import MergeError
from supply.dev.lp_model import engine
from varname import argname

from datatools.core.dtype_cast import maybe_make_list
from datatools.core.dtype_cast import normalize_dtypes
from datatools.pandas_register import register_dataframe_method


@register_dataframe_method
def to_sqlite(
    df: pd.DataFrame,
    table_name: str | None = None,
    con=engine,
    if_exists="replace",
    index=False,
    chunksize=None,
):
    """Insert dataframe into sqlite database."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to insert.")
    else:
        if table_name is None:
            table_name = argname("df")
            logging.warning(
                "table_name is None. Automatic name evaluator found %s name "
                "to be used instead.",
                table_name,
            )
        logging.info(
            "Saving %s to sqlite database. Total rows: %s", table_name, df.shape[0]
        )
        df = df.apply(lambda col: col.dt.date if hasattr(col, "dt") else col)
        df.to_sql(
            table_name,
            con,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
        )


@register_dataframe_method
def from_sqlite(table_name: str, con=engine) -> pd.DataFrame:
    """Load dataframe from sqlite database."""
    return pd.read_sql_table(table_name, con)


@register_dataframe_method
def safe_merge(
    left: pd.DataFrame,
    right: FrameOrSeriesUnion,
    how: str = "inner",
    on: IndexLabel | None = None,
    left_on: IndexLabel | None = None,
    right_on: IndexLabel | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ("_x", "_y"),
    copy: bool = True,
    indicator: bool = False,
    validate: str | None = None,
    allow_inflation: bool = True,
) -> pd.DataFrame:
    """
    Merge DataFrame objects, ensuring that the merge keys have the same data
    type.

    Almost the entire function comes from ``pandas merge``` function, directly.
    Due to some likely future changes to this method, ``@doc`` wrapper from
    ``pandas._utils.wrappers`` not in use.

    Parameters
    ----------
    left : pd.DataFrame
        Left object to merge with
    right : FrameOrSeriesUnion
        Right object to merge with
    how : str {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’
        Indicate how to apply the merge. Possible values:
            * 'left': use only keys from the left frame, like SQL left
              outer join; preserve key order.
            * 'right': use only keys from right frame, like an SQL right
              outer join; preserve key order.
            * 'outer': use union of keys from both frames, like an SQL
              full outer join; sort keys lexicographically.
            * 'inner': use intersection of keys from both frames, like an
              SQL inner join; preserve the order of the left keys.
            * 'cross': creates the cartesian product from both frames,
              preserves the order of the left keys.
    on : IndexLabel | None, default None
        Column or index level names to join on. These must be found in both
        DataFrames. If on is `None` and not merging on indexes then
        this defaults to the intersection of the columns in both DataFrames.
    left_on : IndexLabel | None, default None
        Column or index level names apply join, for the left DataFrame.
        If not empty, function assumes that these are the column indices
        from :param:`left_on`
    right_on : IndexLabel | None = None
        Column or index level names apply join, for the right DataFrame.
        If not empty, function assumes that these are the column indices
        from :param:`right_on`
    left_index : bool, default False
        Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the amount of keys in the other DataFrame (either the
        index or several columns).
    right_index : bool, default False
        Use the index from the right DataFrame as the join key
    sort : bool, default False
        Sort the join keys lexicographically in the result DataFrame. If
        False, the order of the join keys depends on the join type (how
        keyword)
    suffixes : Suffixes, default is (“_x”, “_y”)
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in left and
        right respectively. Pass a value of None instead of a string to
        indicate that the column name from left or right should be left
        as-is, with no suffix. At least one of the values must not be None
    copy : bool, default True
        If False, avoid copy if possible
    indicator : bool | str, default False
        If True, adds a column to the output DataFrame called “_merge” with
        information on the source of each row. The column can be given a
        different name by providing a string argument. The column will have a
        Categorical type with the value of “left_only” for observations whose
        merge key only appears in the left DataFrame, “right_only” for
        observations whose merge key only appears in the right DataFrame,
        and “both” if the observation's merge key is found in both DataFrames
    validate : str | None, optional
        If specified, checks if merge is of specified type.
            * "one_to_one" or "1:1": check if merge keys are unique in both
              left and right datasets.
            * "one_to_many" or "1:m": check if merge keys are unique in the
              left dataset.
            * "many_to_one" or "m:1": check if merge keys are unique in right
              dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.
    allow_inflation : bool, default False
        If "False", allows for merge to dupplicate rows. Otherwise,
        drops dupplicates after the following rules:
            * If how == "left": drop duplicate keys from :param:`right`
              dataframe.
            * If how == "right": drop duplicate keys from :param:`left`
              dataframe.

    Returns
    -------
    pd.DataFrame
        A DataFrame of the two merged objects.

    Raises
    ------
    MergeError
        * If :param:`on` and :param:`left_on` or :param:`right_on` are defined.
        * If :param:`on` and :param:`left_index`, :param:`right_index` defined.
        * If :param:`left_on` and :param:`right_index` are defined,
          or :param:`right_on` and :param:`left_index` are defined.
    ValueError
        * If :param:`left_on` and :param:`right_on` are defined, but both
          attributes do not contain the same amount of columns.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'], 'B': [20, 30, 40]})
    >>> df2 = pd.DataFrame({'a': ['A0', 'A1', 'A3'], 'b': [20, 50, 40]})
    >>> df1.safe_merge(df2, left_on=['A', 'B'], right_on=['a', 'b'], how='inner')
        A	B	a	b
    0	A0	20	A0	20
    >>> df2 = pd.DataFrame({'a': ['A0', 'A1', 'A3'], 'b': ['20', '50', '40']})
    >>> df1.merge(
    ... df2, left_on=['A', 'B'], right_on=['a', 'b'], how='inner'
    ... ) #  doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: You are trying to merge on int64 and object columns...
    >>> df1 = pd.DataFrame({'a': ['A0', 'A1', 'A2'], 'b': [20, 30, 30]})
    >>> df2 = pd.DataFrame({'A': ['A0', 'A1', 'A3'], 'B': ['20', '30', '40']})
    >>> df1.safe_merge(df2, left_on=['a', 'b'], right_on=['A', 'B'], how='inner')
        a	b	A	B
    0	A0	20	A0	20
    1	A1	30	A1	30
    """
    if on is not None:
        on = maybe_make_list(on)
        if any([left_on, right_on]):
            MergeError(
                'Can only pass argument "on" OR "left_on" and "right_on", '
                "not a combination of both."
            )
        left_on = right_on = on
    if left_on is not None and right_on is not None:
        left_on, right_on = maybe_make_list(left_on), maybe_make_list(right_on)
        if len(left_on) != len(right_on):
            raise ValueError("len(right_on) must equal len(left_on)")
        if any([left_index, right_index]):
            raise MergeError(
                'Can only pass argument "left_on" and "right_on"'
                ' OR "left_index" and "right_index", not a combination of both.'
            )
        right = right.rename(
            columns={
                old_name: new_name for old_name, new_name in zip(right_on, left_on)
            }
        )

        left, right = normalize_dtypes(left, right)

        right = right.rename(
            columns={
                old_name: new_name for old_name, new_name in zip(left_on, right_on)
            }
        )
        if not validate and not allow_inflation:
            validate = get_validation_relationship(how)

        if validate:
            left, right = eval_validation(left, right, left_on, right_on, validate)

        return left.merge(
            right,
            how=how,
            left_on=left_on,
            right_on=right_on,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )

    if all([left_index, right_index]):
        return left.merge(
            right,
            how=how,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )
    raise ValueError("Unsupported argument combination")


def get_validation_relationship(how: str) -> str:
    """
    Get the validation relationship for the parameter :param:`how` of the
    :ref:`safe_merge` function. This function is only used,
    if :param:`allow_inflation` is set to False.

    Parameters
    ----------
    how : str {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}
        Value with merge strategy to be converted.

    Returns
    -------
    str
        The validation strategy of the parameter :param:`how`

    Raises
    ------
    ValueError
        If :param:`how` is not a valid merge strategy.
    """
    if how == "left":
        return "many_to_one"
    elif how == "right":
        return "one_to_many"
    elif how == "inner":
        return "one_to_one"
    raise ValueError(f'Unsupported "how" argument {how}')


def eval_validation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: IndexLabel,
    right_on: IndexLabel,
    validate: str,
):
    """
    Evaluate the validation relationship between left and right and ensures
    that the left and right dataframes are compatible with the validation
    strategy.

    Parameters
    ----------
    left : pd.DataFrame
        Left dataframe
    right : pd.DataFrame
        Right dataframe
    left_on : IndexLabel
        Left index label
    right_on : IndexLabel
        Right index label
    validate : str {'one_to_one', 'one_to_many', 'many_to_one'}
        Validation strategy.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Left and right dataframes after validation.
    """
    if validate == "one_to_many":
        left = left.drop_dup_and_log(subset=left_on)
    elif validate == "many_to_one":
        right = right.drop_dup_and_log(subset=right_on)
    elif validate == "one_to_one":
        left = left.drop_dup_and_log(subset=left_on)
        right = right.drop_dup_and_log(subset=right_on)
    else:
        raise ValueError(f'Unsupported "validate" argument {validate}')
    return left, right
