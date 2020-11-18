import pandas as pd
from math import nan
from . import constants, utils

# Mobility-related function definitions

DATA_PATH = "~/code/data/mobility"

def get_clean_data(location_ids):
    """
    Returns cleaned mobility data for the specified location ids
    as a dictionary with keys as the movement category data.
    If data is not available locally, it will be downloaded.
    -----
    location_ids: C3.AI location ids
    """
    raw_data = utils.cached_evalmetrics("outbreaklocation", {
        "ids": location_ids,
        "expressions": [
            "Google_GroceryMobility",
            "Google_TransitStationsMobility",
            "Google_ParksMobility",
            "Google_ResidentialMobility",
            "Google_RetailMobility",
            "Google_WorkplacesMobility"
        ],
        "start": constants.IH_START_DATE,
        "end": constants.IH_END_DATE,
        "interval": "DAY",
    }, DATA_PATH, clean_data)
    return group_by_movement_category(raw_data)


def clean_data(data):
    """
    Given a mobility data set returned by the C3.AI API,
    set proper index and convert to correct data types
    -----
    data: a mobility DataFrame returned by the C3.AI API
    """
    data = data.set_index('dates')
    for col in data.columns:
        data[col] = pd.to_numeric(data[col])
    data = map_missing_to_nan(data)
    return data


def map_missing_to_nan(data):
    """
    Given a mobility data set returned by the C3.AI API,
    map any missing data to NaN values and remove the 'missing' column.

    Assume dates column is set as the index and data types
    are already properly set.

    Any 'missing' value over 0 is considered completely missing
    -----
    data: a mobility DataFrame
    """
    data_values = data.loc[:, data.columns.str.endswith('.data')]
    data_missing = data.loc[:, data.columns.str.endswith('.missing')]
    for col_name, series in data_missing.iteritems():
        data_col_name = col_name[:-7] + "data"
        for index, val in series.iteritems():
            if val > 0:
                data_values.at[index, data_col_name] = nan
    return data_values

def group_by_movement_category(data):
    """
    Given a cleaned mobility dataframe, split the dataframe into
    multiple dataframes, each for its own categry but retaining
    locations as the columns
    -----
    data: the mobility dataframe output from clean_data
    """
    result_dict = {}
    for col_name, series in data.iteritems():
        country, category, val_type = col_name.split(".")
        category = category[7:-8] # Remove Google_ prefix & Mobility suffix
        if category not in result_dict:
            result_dict[category] = pd.DataFrame(index=data.index)
        result_dict[category][country] = series
    return result_dict

def aggregate_weekly_decay(data, start_date, num_weeks=5):
    """
    Given a data set returned by the C3.AI API and a starting date, 
    group the data into weeks from the starting date and aggregating the mean for each.

    The values are also normalized to be the % change 
    from the average of the week prior to the startning date.

    The interval for the data set must be in DAY.
    -----
    data: C3.AI data (as Pandas dataframe)
    start_date: Starting date of the data as string. All data prior will be discarded.
    num_weeks: Number of weeks after the starting date to aggregate for.
    """

    # 0. Prepare variables and convert to proper data types
    start_date = pd.to_datetime(start_date)
    # Make date our index to make it easier to work with
    data = data.set_index('dates')
    for col in data.columns:
        data[col] = pd.to_numeric(data[col])

    # 1. Find the average of the week prior to the starting week
    week_before_start = data.loc[start_date -
                                 pd.DateOffset(days=7):start_date - pd.DateOffset(days=1)]
    baseline_mean = week_before_start.loc[:, week_before_start.columns.str.endswith(
        '.data')].mean()

    # 2. Aggregate data from the date range we're interested in
    data = data.loc[start_date:start_date +
                    pd.DateOffset(days=num_weeks * 7 - 1)]
    # Group by weeks and average each
    data = data.groupby(pd.Grouper(freq='7D')).mean()

    # 3. Transform data based on baseline (only for 'data' columns)
    # Separate the data and missing columns first
    data_values = data.loc[:, data.columns.str.endswith('.data')]
    data_missing = data.loc[:, data.columns.str.endswith('.missing')]
    # Transform the data
    data_values = data_values / baseline_mean * 100
    # Combine back to one dataframe
    final_data = data_values.merge(
        data_missing, left_index=True, right_index=True)
    # Undo index to maintain original data structure
    final_data = final_data.reset_index()

    return final_data
