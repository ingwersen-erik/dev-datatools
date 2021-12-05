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
"""
Functions for casting data types on arrays and dataframes.

Module also provides some inference functions based on data types.

Module Functions
----------------
- :func:`astype_td64_unit_conversion`
    Cast a pandas.Timedelta to a numpy.timedelta64
- :func:`astype_nansafe`
    Cast a numpy.ndarray to a numpy.ndarray with nan-safe casting
- :func:`astype_float_to_int_nansafe`
    Cast objects to integer with nan-safe casting
- :func:`maybe_infer_to_datetimelike`
    Infer dtype from pandas.Series if possible
- :func:`maybe_cast_to_datetime`
    Cast a numpy.ndarray to datetime64[ns], if possible
- :func:`normalize_dtypes`
    Normalize columns in common from two dataframes to a common type
- :func:`find_common_type`
    Normalize a list of dtypes to a common type

Notes
-----
Some functions implement a special design pattern called **dispatch**.
Basically, dispatch allows you to define different versions of the same
function, for different input variables types. This is useful for
implementing functions that accept generalized arguments and need to be able
to handle different types of arguments. For an example, of how it works,
see the :func:`normalize_dtypes`_ function below.
"""
from __future__ import annotations

import logging
import warnings

from collections import abc
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar
from typing import cast
from typing import overload

import numpy as np
import pandas as pd

from multipledispatch import dispatch

# noinspection PyProtectedMember
from numpy import dtype
from pandas._libs import lib

# noinspection PyProtectedMember
from pandas._libs.tslibs import OutOfBoundsDatetime

# noinspection PyProtectedMember
from pandas._libs.tslibs.timedeltas import array_to_timedelta64

# noinspection PyProtectedMember
from pandas._typing import ArrayLike
from pandas._typing import DtypeObj
from pandas.core.dtypes.cast import _disallow_mismatched_datetimelike
from pandas.core.dtypes.cast import ensure_nanosecond_dtype
from pandas.core.dtypes.cast import sanitize_to_nanoseconds
from pandas.core.dtypes.common import ensure_object
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.common import is_complex_dtype
from pandas.core.dtypes.common import is_datetime64_dtype
from pandas.core.dtypes.common import is_datetime64tz_dtype
from pandas.core.dtypes.common import is_dtype_equal
from pandas.core.dtypes.common import is_float_dtype
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.common import is_string_dtype
from pandas.core.dtypes.common import is_timedelta64_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.errors import IntCastingNaNError

# noinspection PyProtectedMember
from pandas.util._exceptions import find_stack_level


if TYPE_CHECKING:
    from pandas.core.arrays import DatetimeArray
    from pandas.core.arrays import ExtensionArray
    from pandas.core.arrays import IntervalArray
    from pandas.core.arrays import PeriodArray
    from pandas.core.arrays import TimedeltaArray

_int8_max = np.iinfo(np.int8).max
_int16_max = np.iinfo(np.int16).max
_int32_max = np.iinfo(np.int32).max
_int64_max = np.iinfo(np.int64).max

NumpyArrayT = TypeVar("NumpyArrayT", bound=np.ndarray)


def astype_td64_unit_conversion(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    By panda's convention, converting to non-nano timedelta64
    returns an int64-dtyped array with ``ints`` representing multiples
    of the desired timedelta unit. This is essentially division.

    Parameters
    ----------
    values : np.ndarray[timedelta64[ns]]
    dtype : np.dtype
        timedelta64 with unit not-necessarily nano
    copy : bool

    Returns
    -------
    np.ndarray
    """
    if is_dtype_equal(values.dtype, dtype):
        if copy:
            return values.copy()
        return values

    # otherwise, we are converting to non-nano
    result = values.astype(dtype, copy=False)  # avoid double-copying
    result = result.astype(np.float64)

    mask = isna(values)
    np.putmask(result, mask, np.nan)
    return result


@overload
def astype_nansafe(
    arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...
) -> np.ndarray:
    ...


@overload
def astype_nansafe(
    arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...
) -> ExtensionArray:
    ...


def astype_nansafe(
    arr: np.ndarray, dtype: DtypeObj, copy: bool = True, skipna: bool = False
) -> ArrayLike:
    """
    Cast the elements of an array to a given ``dtype``, with multiple
    conversion strategies.

    The biggest difference between the casted array, and the original array, is
    that it now supports "NaN" for the integer-like dtype objects.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if e.g. the item sizes
        don't align.
    skipna: bool, default False
        Whether we should skip ``NaN`` when casting as a string-type.
        Defaults to False.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """
    if arr.ndim > 1:
        flat = arr.ravel()
        result = astype_nansafe(flat, dtype, copy=copy, skipna=skipna)
        # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]"
        # has no attribute "reshape"
        return result.reshape(arr.shape)  # type: ignore[union-attr]

    # We get here with 0-dim from sparse
    arr = np.atleast_1d(arr)

    # dispatch on extension dtype if needed
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)

    elif not isinstance(dtype, np.dtype):  # pragma: no cover
        raise ValueError("dtype must be np.dtype or ExtensionDtype")

    if arr.dtype.kind in ["m", "M"] and (
        issubclass(dtype.type, str)
        # error: Non-overlapping equality check (left operand type: "dtype[
        # Any]", right operand type: "Type[object]")
        or dtype == object  # type: ignore[comparison-overlap]
    ):
        from pandas.core.construction import ensure_wrapped_if_datetimelike

        arr = ensure_wrapped_if_datetimelike(arr)
        return arr.astype(dtype, copy=copy)

    if issubclass(dtype.type, str):
        return lib.ensure_string_array(arr, skipna=skipna, convert_na_value=False)

    elif is_datetime64_dtype(arr):
        # Non-overlapping equality check (left operand type: "dtype[Any]", right
        # operand type: "Type[signedinteger[Any]]")
        if dtype == np.int64:  # type: ignore[comparison-overlap]
            return _astype_nansafe(arr, dtype)
        # allow frequency conversions
        if dtype.kind == "M":
            return arr.astype(dtype)

        raise TypeError(
            f"cannot astype a datetime-like from [{arr.dtype}] to [{dtype}]"
        )

    elif is_timedelta64_dtype(arr):
        # error: Non-overlapping equality check (left operand type: "dtype[
        # Any]", right operand type: "Type[signedinteger[Any]]")
        if dtype == np.int64:  # type: ignore[comparison-overlap]
            return _astype_nansafe(arr, dtype)
        elif dtype.kind == "m":
            return astype_td64_unit_conversion(arr, dtype, copy=copy)

        raise TypeError(f"cannot astype a timedelta from [{arr.dtype}] to [{dtype}]")

    elif np.issubdtype(arr.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        return astype_float_to_int_nansafe(arr, dtype, copy)

    elif is_object_dtype(arr):

        # work around NumPy brokenness, #1987
        if np.issubdtype(dtype.type, np.integer):
            return lib.astype_intsafe(arr, dtype)

        # if we have a datetime/timedelta array of objects
        # then coerce to a proper dtype and recall astype_nansafe

        elif is_datetime64_dtype(dtype):
            from pandas import to_datetime

            return astype_nansafe(
                to_datetime(arr).values,
                dtype,
                copy=copy,
            )
        elif is_timedelta64_dtype(dtype):
            from pandas import to_timedelta

            # noinspection PyProtectedMember
            return astype_nansafe(to_timedelta(arr)._values, dtype, copy=copy)

    if dtype.name in ("datetime64", "timedelta64"):
        msg = (
            f"The '{dtype.name}' dtype has no unit. Please pass in "
            f"'{dtype.name}[ns]' instead."
        )
        raise ValueError(msg)

    if copy or is_object_dtype(arr.dtype) or is_object_dtype(dtype):
        # An explicit copy, or required since NumPy can't view from / to object.
        return arr.astype(dtype, copy=True)
    return arr.astype(dtype, copy=copy)


def _astype_nansafe(
    arr: np.ndarray,
    dtype: DtypeObj,
) -> ArrayLike:
    """
    Warns and returns a copy of the array with ``dtype`` cast with ``NaN``

    Parameters
    ----------
    arr : np.ndarray
    dtype : data-type or ndarray sub-class, optional
        Data-type descriptor of the returned view, e.g., float32 or int16.
        Omitting it results in the view having the same data-type as `a`.
        This argument can also be specified as an ndarray sub-class, which
        then specifies the dtype for the returned object (this is equivalent to
        setting the ``type`` parameter).

    Returns
    -------
    ndarray or subclass
        Array interpretation of `arr`.

    Raises
    ------
    ValueError
        The dtype was a `datetime64`/`timedelta64` dtype, but it had no unit.
    """
    warnings.warn(
        f"casting {arr.dtype} values to int64 with .astype(...) "
        "is deprecated and will raise in a future version. "
        "Use .view(...) instead.",
        FutureWarning,
        # stack level chosen to be correct when reached via
        # Series.astype
        stacklevel=find_stack_level(),
    )
    if isna(arr).any():
        raise ValueError("Cannot convert NaT values to integer")
    return arr.view(dtype)


def astype_float_to_int_nansafe(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    Astype with check preventing conversion of NaN to a meaningless integer.

    Parameters
    ----------
    values : np.ndarray
        The values to be converted
    dtype : np.dtype
        The dtype to convert to
    copy : bool
        Whether to copy the values.
    """
    if not np.isfinite(values).all():
        raise IntCastingNaNError(
            "Cannot convert non-finite values (NA or inf) to integer"
        )
    return values.astype(dtype, copy=copy)


def _maybe_infer_to_datetimelike(
    value: np.ndarray,
) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    Maybe infer object as a datetime-like array.

    We might have an array (or single object) that is datetime like.
    Object might lack dtype, and in such cases no change to the value occurs,
    unless we find a datetime/timedelta set.

    This is pretty strict in that a datetime/timedelta is **REQUIRED**, besides
    possible nulls, or string-like values.

    Parameters
    ----------
    value : np.ndarray[object]
        The array/value to be maybe inferred to a datetimelike.

    Returns
    -------
    np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray
        Object maybe inferred to a datetimelike. Can be one of the following
        index objects:

            * `np.ndarray`
                Multidimensional, homogeneous array of fixed-size.
            * `DatetimeArray`
                tz-naive or tz-aware datetime data.
            * `TimedeltaArray`
                ExtensionArray for timedelta data.
            * `PeriodArray`
                Pandas ExtensionArray for storing Period data.
            * `IntervalArray`
                An immutable index of intervals that are closed on the same side
    """
    if not isinstance(value, np.ndarray) or value.dtype != object:
        raise TypeError(type(value))  # pragma: no cover

    v = np.array(value, copy=False)

    shape = v.shape
    if v.ndim != 1:
        v = v.ravel()

    if not len(v):
        return value

    # noinspection PyShadowingNames
    def try_datetime(v: np.ndarray) -> ArrayLike:
        # Coerce to datetime64, datetime64tz, or in "corner cases",
        # object[datetimes]
        from pandas.core.arrays.datetimes import sequence_to_datetimes

        try:
            dta = sequence_to_datetimes(v, require_iso8601=True)
        except (ValueError, TypeError):
            # e.g. <class 'numpy.timedelta64'> is not convertible to datetime
            return v.reshape(shape)
        else:
            return dta.reshape(shape)

    # noinspection PyShadowingNames
    def try_timedelta(v: np.ndarray) -> np.ndarray:
        # safe coerce to timedelta64 will try first with a string & object
        # conversion
        try:
            td_values = array_to_timedelta64(v).view("m8[ns]")
        except (ValueError, OverflowError):
            return v.reshape(shape)
        else:
            return td_values.reshape(shape)

    inferred_type, seen_str = lib.infer_datetimelike_array(ensure_object(v))
    if inferred_type in ["period", "interval"]:
        return lib.maybe_convert_objects(  # type: ignore[return-value]
            v, convert_period=True, convert_interval=True
        )

    if inferred_type == "datetime":
        value = try_datetime(v)  # type: ignore[assignment]
    elif inferred_type == "timedelta":
        value = try_timedelta(v)
    elif inferred_type == "nat":
        # if all NaT, return as datetime
        if isna(v).all():
            value = try_datetime(v)  # type: ignore[assignment]
        else:
            value = try_timedelta(v)
            if lib.infer_dtype(value, skipna=False) in ["mixed"]:
                # cannot skip missing values, as NaT implies that the string
                # is actually a datetime
                value = try_datetime(v)  # type: ignore[assignment]

    if value.dtype.kind in ["m", "M"] and seen_str:
        warnings.warn(
            f"Inferring {value.dtype} from data containing strings is "
            f"deprecated and will be removed in a future version. "
            f"To retain the old behavior explicitly pass "
            f"Series(data, dtype={value.dtype})",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    return value


# noinspection PyProtectedMember
def maybe_cast_to_datetime(
    value: ExtensionArray | np.ndarray | list, dtype: DtypeObj | None
) -> ExtensionArray | np.ndarray:
    """
    Maybe cast the array/value to a datetimelike dtype.

    TRY to cast the array/value to a datetimelike dtype, converting float
    nan to iNaT. We allow a list *only* when dtype is not None.

    Parameters
    ----------
    value : ExtensionArray, np.ndarray, list
        The value to cast
    dtype : dtype, None
        The dtype to try to cast to.

    Returns
    -------
    ExtensionArray | np.ndarray
        The array with values **possibly** (but not certainly) casted.

    Raises
    ------
    TypeError
        if value is not a list like object.
    OutOfBoundsDatetime
        if the value is astyped datetime and is out of bounds for the dtype.
    ValueError
        if the value is a list and the dtype is None.
    """
    from pandas.core.arrays.datetimes import sequence_to_datetimes
    from pandas.core.arrays.timedeltas import TimedeltaArray

    if not is_list_like(value):
        raise TypeError(f"Value must be list-like, not {value}")

    if is_timedelta64_dtype(dtype):
        dtype = cast(np.dtype, dtype)
        dtype = ensure_nanosecond_dtype(dtype)
        return TimedeltaArray._from_sequence(value, dtype=dtype)

    if dtype is not None:
        is_datetime64 = is_datetime64_dtype(dtype)
        is_datetime64tz = is_datetime64tz_dtype(dtype)

        vdtype = getattr(value, "dtype", None)

        if is_datetime64 or is_datetime64tz:
            dtype = ensure_nanosecond_dtype(dtype)

            value = np.array(value, copy=False)

            # we have an array of datetime or timedelta & nulls
            if value.size or not is_dtype_equal(value.dtype, dtype):
                _disallow_mismatched_datetimelike(value, dtype)

                try:
                    if is_datetime64:
                        dta = sequence_to_datetimes(value)

                        if dta.tz is not None:
                            warnings.warn(
                                "Data is timezone-aware. Converting "
                                "timezone-aware data to timezone-naive by "
                                "passing dtype='datetime64[ns]' to "
                                "DataFrame or Series is deprecated and will "
                                "raise in a future version. Use "
                                "`pd.Series(values).dt.tz_localize(None)` "
                                "instead.",
                                FutureWarning,
                                stacklevel=find_stack_level(),
                            )
                            # equiv: dta.view(dtype)
                            # Note: NOT equivalent to dta.astype(dtype)
                            dta = dta.tz_localize(None)

                        value = dta
                    elif is_datetime64tz:
                        dtype = cast(DatetimeTZDtype, dtype)
                        is_dt_string = is_string_dtype(value.dtype)
                        dta = sequence_to_datetimes(value)
                        if dta.tz is not None:
                            value = dta.astype(dtype, copy=False)
                        elif is_dt_string:
                            value = dta.tz_localize(dtype.tz)
                        else:
                            if getattr(vdtype, "kind", None) == "M":
                                warnings.warn(
                                    "In a future version, constructing a "
                                    "Series from datetime64[ns] data and a "
                                    "DatetimeTZDtype will interpret the data "
                                    "as wall-times instead of  UTC times, "
                                    "matching the behavior of DatetimeIndex. "
                                    "To treat the data as UTC times, "
                                    "use pd.Series(data).dt.tz_localize('UTC')"
                                    ".tz_convert(dtype.tz) or "
                                    "pd.Series(data.view('int64'),dtype=dtype)",
                                    FutureWarning,
                                    stacklevel=find_stack_level(),
                                )

                            value = dta.tz_localize("UTC").tz_convert(dtype.tz)
                except OutOfBoundsDatetime:
                    raise
                except ValueError:
                    pass

        elif getattr(vdtype, "kind", None) in ["m", "M"]:
            return astype_nansafe(value, dtype)  # type: ignore[arg-type]

    elif isinstance(value, np.ndarray):
        if value.dtype.kind in ["M", "m"]:
            # catch a datetime/timedelta that is not of ns variety
            # and no coercion specified
            value = sanitize_to_nanoseconds(value)

        elif value.dtype == object:
            value = _maybe_infer_to_datetimelike(value)

    elif isinstance(value, list):
        # We only get here with dtype=None, which we do not allow
        raise ValueError(
            "maybe_cast_to_datetime allows a list *only* if dtype is not None"
        )
    # At this point we have converted or raised every possible list that we had
    return cast(ArrayLike, value)


# TODO: Fix cases where it fails to find a common dtype, for values of type
#       Object
def find_common_type(
    types: list[DtypeObj] | DtypeObj,
) -> list[dtype | ExtensionDtype] | dtype | object | Any:
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : list[DtypeObj] | DtypeObj
        A list of possible dtypes that can be considered for conversion.
        Or a single dtype, thus not requiring to find a common type.

    Returns
    -------
    list[dtype | ExtensionDtype] | dtype | object | Any
        pandas extension or numpy dtype

    Raises
    ------
    ValueError
        Raises ValueError if :param:`types` is empty

    See Also
    --------
    The function `numpy.find_common_type <numpy.find_common_type>`_
    contains the Numpy's implementation of this function behavior.
    """
    if not types:
        raise ValueError("No types given.")
    elif not isinstance(types, Iterable):
        types = [types]

    first = types[0]

    # Workaround for find_common_type([np.dtype('datetime64[ns]')] * 2)
    # => object
    if all(is_dtype_equal(first, t) for t in types[1:]):
        return first

    # Get unique types (dict.fromkeys is used as order-preserving set())
    types = list(dict.fromkeys(types).keys())

    if any(isinstance(t, ExtensionDtype) for t in types):
        for _type in types:
            if isinstance(_type, ExtensionDtype):
                res = _type._get_common_dtype(types)
                if res is not None:
                    return res
        return np.dtype("object")

    # Take the lowest unit
    if all(is_datetime64_dtype(t) for t in types):
        return np.dtype("datetime64[ns]")
    if all(is_timedelta64_dtype(t) for t in types):
        return np.dtype("timedelta64[ns]")

    # Don't mix bool, int, float or complex
    # this is different from what numpy does, when it casts bools represented
    # by floats or ints to int
    has_bools = any(is_bool_dtype(t) for t in types)

    if has_bools:
        for t in types:
            if is_integer_dtype(t) or is_float_dtype(t) or is_complex_dtype(t):
                return np.dtype("object")

    return np.find_common_type(types, [])  # type: ignore[arg-type]


@dispatch(pd.DataFrame, pd.DataFrame)
def normalize_dtypes(
    xdf: pd.DataFrame,
    ydf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize the underlying column dtypes of two input dataframes.

    Function determines which columns exist on both dataframes, and only
    converts those ``pd.Series`` of values to the same dtype.

    :func:`find_common_type` finds the common data type supported to column
    on both input dataframes.

    Parameters
    ----------
    xdf : pd.DataFrame
        The first dataframe to be updated.
    ydf : pd.DataFrame
        The second dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The two dataframes with the same data types.
    """
    return normalize_dtypes(xdf, ydf, [*xdf.columns, *ydf.columns])


# TODO: After applying bugfix to :func:`find_common_type`, remove the if/else
#       that converts Objects to str (also Objects), and convert ``astype_dict``
#       to dict comprehension.
@dispatch(pd.DataFrame, pd.DataFrame, abc.Iterable)
def normalize_dtypes(
    xdf: pd.DataFrame, ydf: pd.DataFrame, use_cols: abc.Iterable
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize the underlying column dtypes of the given dataframes.

    Function finds columns in common (by column name), using each
    dataframe's columns and the user-provided list of columns to normalize
    data type. This dispatch won't normalize the data types of all the columns
    with matching names.

    Parameters
    ----------
    xdf : pd.DataFrame
        The first dataframe to be updated.
    ydf : pd.DataFrame
        The second dataframe.
    use_cols : abc.Iterable
        User-defined list of column names to normalize.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The two dataframes with the same data types.
    """
    if not iterable_not_string(use_cols):
        use_cols = [use_cols]

    cols_to_consider = list(set(xdf.columns) & set(use_cols) & set(ydf.columns))

    astype_dict = {}
    for col in cols_to_consider:
        common_type = find_common_type([xdf[col].dtype, ydf[col].dtype])
        astype_dict[col] = str if common_type == object else common_type
    if astype_dict:
        logging.info(
            "Converting the following columns underlying data types: %s",
            astype_dict,
        )
    return xdf.astype(astype_dict), ydf.astype(astype_dict)


def iterable_not_string(obj) -> bool:
    """
    Check if :param:`obj` is iterable, and also, not a string.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    is_iter_not_string : bool
        Whether `obj` is a non-string iterable.

    Examples
    --------
    >>> iterable_not_string([1, 2, 3])
    True
    >>> iterable_not_string("foo")
    False
    >>> iterable_not_string(1)
    False

    Notes
    -----
    Python strings, like lists and other list-like objects, are also Iterable.
    Therefore, applying ``isinstance(obj, Iterable)`` to possible ``string``,
    variables, to determine which objects to convert to list won't work as
    one might expect. Therefore, this function not only reduces the quantity of
    boilerplate code, but also reduces the risk of errors.

    Examples
    --------
    >>> from typing import Iterable
    >>> lst = [['1', '2', '5', '8'], '12', '15']
    >>> for l in lst:
    ...     print(l if isinstance(l, Iterable) else '')
    ['1', '2', '5', '8']
    12
    15
    >>> for l in lst:
    ...     print(l if iterable_not_string(l) else '')
    ['1', '2', '5', '8']
    """
    return isinstance(obj, abc.Iterable) and not isinstance(obj, str)


def maybe_make_list(obj: Any) -> Any | list[Any]:
    """
    Convert ``obj`` to ``list``, if ``obj`` is ``Iterable``,
    and more speciffically a ``string``.

    Function works similarly to :func:`iterable_not_string`, but instead of
    returning True or False, it returns the object as a list, whenever it is
    possible.

    Parameters
    ----------
    obj : Any
        The object to be evaluated and possibly converted to list.

    Returns
    -------
    Any | list[Any]
        The original object, or a list of the original object.

    Examples
    --------
    >>> x = '12345'
    >>> list(x)
    ['1', '2', '3', '4', '5']
    >>> maybe_make_list(x)
    ['12345']

    >>> maybe_make_list(None)

    >>> maybe_make_list(['1', '2', '3'])
    ['1', '2', '3']
    """
    return [obj] if obj is not None and not isinstance(obj, (tuple, list)) else obj
