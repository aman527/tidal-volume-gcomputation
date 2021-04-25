from pathlib import Path

import numpy as np
import pandas as pd

CWD = Path().absolute()
END_HOUR = 96

df = pd.read_csv(CWD / 'driving_pressure_dataset.csv')

# ---- create / rename useful columns ----
df['driving_pressure'] = df['plateau_pressure'] - df['peep_set']
# whether the driving pressure was measured at a given time point
df['driving_pressure_meas'] = (df['plateau_pressure_meas'] == True) & (df['peep_set_meas'] == True)
# whether the driving pressure was measured at time t - 1
df['last_driving_pressure_meas'] = (df['last_plateau_pressure_meas'] == True) & (df['last_peep_set_meas'] == True)
# rename tidal volume columns
df['imputed_TV_standardized_meas'] = df['tidal_volume_set_meas']
df['last_imputed_TV_standardized_meas'] = df['last_tidal_volume_set_meas']

def apply_by_stay( _handle_group, column_name: str):
    """Create a new column resulting from the elementwise application of a given function to the inputted column."""
    if (len(df[column_name]) <= 1):
        return pd.Series([np.nan])
    # First, split column by stay_id
    grouped = df.groupby('stay_id', axis='rows')[column_name]
    # Each group is a tuple, so extract the pd.Series from each group
    grouped = map(lambda group: group[1], grouped)
    # Apply the function to each group
    last_groups = [_handle_group(g) for g in grouped]
    # Concatenate the list of groups into one final column, resetting indices
    last_column = pd.concat(last_groups, ignore_index=True)
    return last_column


def _get_last(group):
    """# Convert each row of a given group into storing the LAST measurement (ie the value at time t - 1)."""
    # Inputs tuple where elemnent 1 is a pd.Series, prepends NaN to said Series.
    # pd.concat isn't used because it is VERY slow when used multiple times
    last_group = [np.nan]
    last_group[1:] =  group[:len(group)-1]
    last_group = pd.Series(last_group)
    return last_group


def _get_time_since_last_meas(group):
    """# Get time since last measurement, where group is a column of bools representing whether a meas was taken."""
    # If there are no measurements, just return a list of NaNs
    if (group.sum() == 0):
        return pd.Series(np.full(len(group), np.nan))
    # Otherwise, determine the indices of all measurements 
    ones = group.loc[group == True].index    
    # Get a list of the difference in time values between each measurement
    steps_between_meas = np.concatenate([ones[1:], [len(group)]]) - ones
    # Values of ts_last should be NaN before the first measurement
    pre_meas_1 = np.full(ones[0], np.nan)
    # post_meas_1 is now a nested list of range objects
    post_meas_1 = [np.arange(i) for i in steps_between_meas]
    # flatten the nested list
    post_meas_1 = np.concatenate(post_meas_1)
    # concatenate the pre- and post- measurement 1 values
    new_group = np.concatenate((pre_meas_1, post_meas_1))
    return pd.Series(new_group)


def time_since_prev(column):
    """Create new column tracking the time since the previous measurement.

    Previous differs from last in one key way: when a value is measured at the current time point, LAST = 0. 
    But PREVIOUS lists the time since the measurement before the current one.
    """
    ts_last_col = column + '_ts_last'
    meas_col = column + '_meas'
    # Condition to replace with 'previous' value: if a measurement was taken at the current timestep and if study hour is > 1
    condition = (df[meas_col] == True) & (df['study_hour'] > 1)
    # Create a numpy array that tracks what the 'previous' value WOULD be at any given time point
    prev_value = np.concatenate([[0], df[ts_last_col]])[:-1] + 1
    # If the condition is satisfied, replace the 'time since last' value with the new, 'time since previous', value
    new_column = np.where(condition, prev_value, df[ts_last_col])
    return pd.Series(new_column)


for p in ('driving_pressure', 'plateau_pressure'):
    # Create column tracking change in measurement from time = t - 1 to time = t
    df[p + '_change'] = df[p] - df['last_' + p]
    # Create column storing the value of the last measurement
    df['last_' + p + '_change'] = apply_by_stay(_get_last, p + '_change')
    # Create column storing the time since last measurement
    df[p + '_ts_last'] = apply_by_stay(_get_time_since_last_meas, p + '_meas')
    # Create column storing the time since the previous measurement
    df[p+ '_ts_prev'] = time_since_prev(p)

df['peep_set_change'] = df['peep_set'] - df['last_peep_set']
df['tidal_volume_set_change'] = df['tidal_volume_set'] - df['last_tidal_volume_set']
