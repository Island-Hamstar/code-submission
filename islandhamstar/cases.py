from . import constants, utils
import pandas as pd
from math import nan

DATA_PATH = "~/code/data/cases"


def get_clean_data(location_ids):
    """
    Returns cleaned JHU cases data for the specified location ids.
    If data is not available locally, it will be downloaded.
    -----
    location_ids: C3.AI location ids
    """
    raw_data = utils.cached_evalmetrics("outbreaklocation", {
        "ids": location_ids,
        "expressions": [
            "JHU_ConfirmedCases",
            "JHU_ConfirmedDeaths",
            "JHU_ConfirmedRecoveries",
        ],
        "start": constants.IH_START_DATE,
        "end": constants.IH_END_DATE,
        "interval": "DAY",
    }, DATA_PATH, clean_data)
    return cases_only(raw_data)


def cases_only(data):
    """
    Filter out cases dataframe to only contain confirmed cases
    """
    filtered_data = data.filter(like='JHU_ConfirmedCases', axis='columns')
    return filtered_data.rename(columns=lambda x: x.split(".")[0])


def clean_data(data):
    """
    Given a case report data set returned by the C3.AI API,
    set proper index and convert to correct data types
    -----
    data: a case report DataFrame returned by the C3.AI API
    """
    data = data.set_index('dates')
    for col in data.columns:
        data[col] = pd.to_numeric(data[col])
    data = map_missing_to_nan(data)
    return data


def map_missing_to_nan(data):
    """
    Given a case report data set returned by the C3.AI API,
    map any missing data to NaN values and remove the 'missing' column.

    Assume dates column is set as the index and data types
    are already properly set.

    Any 'missing' value over 0 is considered completely missing
    -----
    data: a case report DataFrame
    """
    data_values = data.loc[:, data.columns.str.endswith('.data')]
    data_missing = data.loc[:, data.columns.str.endswith('.missing')]
    for col_name, series in data_missing.iteritems():
        data_col_name = col_name[:-7] + "data"
        for index, val in series.iteritems():
            if val > 0:
                data_values.at[index, data_col_name] = nan
    return data_values
