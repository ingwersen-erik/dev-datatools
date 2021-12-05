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


from __future__ import annotations

import logging

from typing import Optional

import pandas as pd


def rename_back(
    df: pd.DataFrame,
    attr_column_map: str | None = "column_map",
    errors: str | None = "ignore",
) -> pd.DataFrame:
    """
    Rename columns back to their original names. Function tries to do that,
    by relying on a potentially saved attribute, that contains a dictionary
    with the original and their new names.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to rename.
    attr_column_map: str, optional
        The attribute name that suposedly stores the column map.
        By default, "column_map"
    errors : str {'ignore', 'raise'}, optional
        The error handling strategy to use when old column names are not
        inside the dataframe attributes. By default, 'ignore' is used.

    Returns
    -------
    pd.DataFrame
        The dataframe with the columns renamed, when attribute
        :param:`column_map` exists.

    Raises
    ------
    AttributeError
        When attribute :param:`column_map` does not exist.

    Examples
    --------
    >>> _df = pd.DataFrame(
    ...     {
    ...          'order_material': ['A', 'B', 'C'],
    ...          'site': ['X', 'Y', 'Z'],
    ...          pd.to_datetime('2021-10-10'): [0, 10, 0],
    ...     }
    ... )
    >>> _df.attrs['column_map'] = {'order_material': 'material'}
    >>> rename_back(_df)
      material site  2021-10-10 00:00:00
    0        A    X                    0
    1        B    Y                   10
    2        C    Z                    0
    """
    column_map = df.attrs.get(attr_column_map)
    if column_map is None and errors == "raise":
        raise AttributeError(
            f"Tried to rename columns, but no {attr_column_map} attribute found"
        )
    if column_map:
        logging.info("Renaming columns back to original names")
        column_map = {v: k for k, v in column_map.items()}
        df = df.rename(columns=column_map)
    return df


def fmt_colnames(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Beautifies the column names of a given dataframe.

    Formatting Options
    ------------------
    * Convert column names to uppercase
    * Replaces underscores with spaces "_" -> " "
    * Converts any datetime columns to dates

    Parameters
    ----------
    _df : pd.DataFrame
        The dataframe to rename the columns for.

    Returns
    -------
    pd.DataFrame
        The dataframe with renamed columns.

    Examples
    --------
    >>> # noinspection PyShadowingNames
    >>> _df = pd.DataFrame(
    ...     {
    ...         'order_material': ['A', 'B', 'C'],
    ...         'site': ['X', 'Y', 'Z'],
    ...         '2021/10/10': [0, 10, 0]
    ...     }
    ... )
    >>> fmt_colnames(_df)
      ORDER MATERIAL SITE  2021-10-10
    0              A    X           0
    1              B    Y          10
    2              C    Z           0
    """
    df_original_names = rename_back(_df)
    if df_original_names is not None:
        return df_original_names

    return _df.rename(
        columns={
            original_column: pd.to_datetime(original_column, errors="ignore").strftime(
                "%Y-%m-%d"
            )
            for original_column in _df.columns
            if isinstance(
                pd.to_datetime(original_column, errors="ignore"), pd.Timestamp
            )
        }
    ).rename(
        columns={
            original_column: str(original_column).upper().replace("_", " ")
            for original_column in _df.columns
        }
    )
