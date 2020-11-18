# This file stores utility functions to be used for data analytics
import pandas as pd
from scipy import stats, integrate
import logging
from os import path
import c3aidatalake
from math import nan
from . import constants

# The threshold of number of skipped days in get_impact before a warning is generated
IMPACT_DAYS_SKIP_WARNING_THRESHOLD = 10

def get_impact(data, date, pre_window, post_window):
    """
    Given a pandas Series, calculate the "impact score" of the values 
    centered at the specified origin date, such as for the purpose of finding 
    the effects of some government policies.

    "impact score" is defined as [area(lr_post) - area(lr_pre)] / area(lr_pre)
    where
        - area() is the area under the line given
        - lr_pre is the line obtained by doing linear regression on the data
          in the 'pre_window' days time before 'date'
        - lr_post is the line obtained by doing linear regression on the data
          in the 'post_window' days time after 'date'

    If some values in data are missing, the data for that day will be skipped
    and next available day, going in the direction away from the origin date,
    will be used instead.

    data must have enough data points, after taking into account the origin date,
    pre_window, post_window, and any missing data.
    -----
    data: The pandas Series containing numeric values, with dates on the row index
    date: The origin date. As string or datetime.
    pre_window: The number of days before the origin date to calculate linear regression on.
    post_window: The number of days after the origin date to calculate linear regression on.
    """

    # Verify that we have proper date type on the DataFrame index
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        raise TypeError(
            "Series index must be a datetime type. Call set_index on the date column first.")

    date = pd.to_datetime(date)
    data = data.sort_index()

    logging.info(f"get_impact: Getting impact score for {data.name} on {date.date()}")

    # Each column of the DataFrame is worked on separately, as we treat
    # missing rows separately from each other
    date_after = date + pd.DateOffset(days=1)

    pre_data = get_consecutive_rows(data, date, pre_window, 1)
    post_data = get_consecutive_rows(data, date_after, post_window, 0)
    if len(pre_data) < 2 or len(post_data) < 2:
        logging.warning(f"get_impact: Not enough data points to calculate impact!")
        return nan
    if len(pre_data) < pre_window or len(post_data) < post_window:
        logging.warning(f"get_impact: Using less data points than specified in pre_window/post_window! (pre:{len(pre_data)}, post:{len(post_data)})")
    pre_days_skipped = (date - pre_data.index.min()).days - pre_window + 1
    post_days_skipped = (post_data.index.max() - date_after).days - post_window + 1
    if pre_days_skipped > IMPACT_DAYS_SKIP_WARNING_THRESHOLD or post_days_skipped > IMPACT_DAYS_SKIP_WARNING_THRESHOLD:
        logging.warning(f"get_impact: There are large gaps in data used for linear regression! (pre_skipped:{pre_days_skipped}, post_skipped:{post_days_skipped})")

    pre_x = (pre_data.index - date).days
    pre_y = pre_data.to_numpy()
    post_x = (post_data.index - date).days
    post_y = post_data.to_numpy()

    pre_regress = stats.linregress(pre_x, pre_y)
    post_regress = stats.linregress(post_x, post_y)
    pre_regress_line_func = lambda x: max(0, (pre_regress.slope * x) + pre_regress.intercept)
    post_regress_line_func = lambda x: (post_regress.slope * x) + post_regress.intercept
    impact_area_func = lambda x: post_regress_line_func(x) - pre_regress_line_func(x)

    post_begin = min(post_x)
    post_end = max(post_x)
    impact_area = integrate.quad(impact_area_func, a=post_begin, b=post_end)[0]
    total_area = integrate.quad(pre_regress_line_func, a=post_begin, b=post_end)[0]
    logging.debug(f"get_impact: impact_area={impact_area}, total_area={total_area}")
    if total_area == 0:
        total_area += 1
        impact_area += 1
    impact_score = impact_area / total_area
    return impact_score


def get_consecutive_rows(data, origin, num, direction):
    """
    Given an index sorted pandas Series, return num number of 
    consecutive rows in the given direction from the starting 
    origin point on the index, inclusively, with a valid datapoint.

    A valid datapoint is a non-NaN numeric value. 0-value is valid.
    -----
    data: The pandas dataframe. Must be sorted on the index.
    origin: The index item to start from
    num: The number of data points to return
    direction: The direction of the points from the origin. 0 = Ascending, 1 = Descending
    """

    data = data.dropna()

    try:
        if direction == 0:
            origin_loc = data.index.get_loc(origin, method="bfill")
            return data.iloc[origin_loc:origin_loc+num]
        elif direction == 1:
            origin_loc = data.index.get_loc(origin, method="ffill")
            lower_bound = origin_loc + 1 - num
            # A negative lower bound results in empty series
            # So bound lower bound to zero here
            if lower_bound < 0:
                lower_bound = 0
            return data.iloc[lower_bound:origin_loc+1]
        else:
            raise ValueError("Invalid value for direction parameter")
    except KeyError:
        # If origin is out of range, return empty data
        return data.iloc[0:0]


def cached_evalmetrics(evalmetrics_api, evalmetrics_spec, csv_dir, clean_func=None):
    """
    Download and return data from C3.AI Evalmetrics API.
    DataFrame is saved to CSV files based on location ids in the spec.
    Existing local data is returned if found.
    A function can be provided to transform the DataFrame before saving and returning.
    -----
    evalmetrics_spec: spec dictionary to provide to evalmetrics api
    csv_dir: directory to read and write csv files
    clean_func: optional function to transform dataframe before returning and saving
    """
    result_data = pd.DataFrame()
    location_ids = evalmetrics_spec["ids"]
    expressions = evalmetrics_spec["expressions"]

    for location_id in location_ids:
        csv_path = path.join(csv_dir, location_id + ".csv")
        evalmetrics_spec["ids"] = [location_id]
        # Re-assign the expressions property because the call to c3aidatalake.evalmetrics modifies it
        # causing calls in second loop and after to return invalid results.
        evalmetrics_spec["expressions"] = expressions

        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logging.debug(f"Local data found for {location_id} at {csv_path}")
            result_data = result_data.join(df, how="outer")

        except FileNotFoundError:
            logging.debug(
                f"No local data found for {location_id}. Fetching live data.")
            new_df = c3aidatalake.evalmetrics(
                evalmetrics_api,
                {
                    "spec": evalmetrics_spec
                },
                get_all=True
            )
            if clean_func is not None:
                new_df = clean_func(new_df)
            new_df.to_csv(csv_path)
            result_data = result_data.join(new_df, how="outer")

    return result_data
