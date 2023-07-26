# This is a sample Python script.

import numpy as np
import pandas as pd
import statistics
import math

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# cache = TTLCache(maxsize=1000, ttl=60)

patient_file_url = 'data/patients.csv'
output_event_file_url = 'data/output_events.csv'
chart_event_file_url = 'data/chart_events.csv'
lab_event_file_url = 'data/lab_events.csv'


def read_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(file_path)
        # Now you can use the 'df' DataFrame to work with your CSV data
        return df
    except FileNotFoundError:
        print("Error: The file could not be found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or does not contain any data.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def write_to_csv(file_path, result_dict):
    df = pd.DataFrame([result_dict],
                      columns=['patient_id', 'admission_id', 'gender', 'dob', 'time_series', 'cum_urine_output',
                               'min_bp', 'mean_bp', 'max_dp', 'var_bp', 'std_bp', 'min_temp', 'mean_temp', 'max_temp',
                               'var_temp', 'std_temp', 'lactate_diff'])
    df.to_csv(file_path, mode='a', index=False, header=False)


def filter_data_by_patient_id(df, patient_id):
    if patient_id in df['subject_id'].values:
        filtered_df = filter_rows_by_col_val(df, 'subject_id', patient_id)
        return filtered_df


def process_patient_details():
    # Read all the required files

    patient_df = read_csv(patient_file_url)
    output_event_df = read_csv(output_event_file_url)
    chart_event_df = read_csv(chart_event_file_url)
    lab_event_df = read_csv(lab_event_file_url)

    # clean up of non available patients (3 patients are not available in output_event_file)
    filtered_patient_df = []
    output_patient_id_list = output_event_df['subject_id'].values
    for index, row in patient_df.iterrows():
        if row['subject_id'] in output_patient_id_list:
            filtered_patient_df.append(row)
        else:
            continue

    # result data dictionary to be saved in csv file

    for row in filtered_patient_df:

        # get patient_id, gender and dob from the patient data frame
        patient_id = row['subject_id']
        gender = row['gender']
        dob = row['dob']

        # filter all the output events for each patient id
        output_event_filtered_df = filter_data_by_patient_id(output_event_df, patient_id)

        # filter all the chart events for each patient_id
        chart_event_filtered_df = filter_data_by_patient_id(chart_event_df, patient_id)

        # # filter all the lab events for each patient_id
        lab_event_filtered_df = filter_data_by_patient_id(lab_event_df, patient_id)
        #
        # # filtered data frames containing blood pressure from each patient's chart
        patient_bp_filtered_df = filter_rows_by_col_val(chart_event_filtered_df, 'valueuom', 'mmHg')

        # filtered data frames containing temperature from each patient's chart
        patient_temp_filtered_df = filter_rows_by_col_val_list(chart_event_filtered_df, 'valueuom',
                                                               ['?F', 'Deg. C', 'Deg. F'])

        # filtered data frame containing lactate in lab reports
        patient_lactate_filtered = filter_rows_by_col_val(lab_event_filtered_df, 'valuenum', 'mmol/L')

        # sorted all the filtered data frames in ascending order of chart time to get start and end timestamp
        sorted_output_event_df = sort_by_col(output_event_filtered_df, 'charttime')
        sorted_bp_filtered_df = sort_by_col(patient_bp_filtered_df, 'charttime')
        sorted_temp_filtered_df = sort_by_col(patient_temp_filtered_df, 'charttime')
        sorted_lactate_filtered_df = sort_by_col(patient_lactate_filtered, 'charttime')

        admission_id = 0
        if len(sorted_output_event_df) > 0:
            admission_id = sorted_output_event_df['hadm_id'].iloc[0]

        start_time = sorted_output_event_df['charttime'].iloc[0]
        end_time = sorted_output_event_df['charttime'].iloc[-1]

        # Check if start_time is before or equal to end_time
        if start_time > end_time:
            raise ValueError("start_time must be before or equal to end_time.")

        # time series list between start and end time
        time_series_list = pd.date_range(start=start_time, end=end_time, freq='1H')

        for time in time_series_list:
            result_dict = {
                'patient_id': patient_id,
                'admission_id': admission_id,
                'gender': gender,
                'dob': dob,
                'time_series': time
            }
            # current timestamp
            current_timestamp = pd.to_datetime(time)

            # next one-hour timestamp
            next_timestamp = pd.to_datetime(time) + pd.Timedelta(hours=1)

            # cumulative urine output
            filtered_urine_measures_timestamps = filter_timestamp_avail_in_lookup(sorted_output_event_df,
                                                                                  current_timestamp,
                                                                                  next_timestamp)
            cumulative_urine_output = np.cumsum(filtered_urine_measures_timestamps['value'].values)

            set_cumulative_urine_output(cumulative_urine_output, result_dict)

            # min/mean/max/variance/ std of blood pressure
            filtered_bp_measures_df = filter_timestamp_avail_in_lookup(sorted_bp_filtered_df, current_timestamp,
                                                                       next_timestamp)
            bp_values = filtered_bp_measures_df['value'].values

            set_bp_measurement(bp_values, result_dict)

            # min/mean/max/variance/ std deviation of body temperature
            filtered_temp_measures_timestamps = filter_timestamp_avail_in_lookup(sorted_temp_filtered_df,
                                                                                 current_timestamp,
                                                                                 next_timestamp)
            temp_values = filtered_temp_measures_timestamps['value'].values
            set_temp_measurement(temp_values, result_dict)

            # current and previous lactate difference
            filtered_timestamps_for_lactate_events = filter_timestamp_avail_in_lookup(sorted_lactate_filtered_df,
                                                                                      current_timestamp,
                                                                                      next_timestamp)
            lactate_values = filtered_timestamps_for_lactate_events['value'].values
            set_lactate_measurement(lactate_values, result_dict)

            # write to csv file
            write_to_csv('results/results.csv', result_dict)


def set_bp_measurement(bp_values, result_dict):
    bp_values_float = [float(value) for value in bp_values]
    if len(bp_values_float) > 0:
        calculate_all_mathematical_cal(result_dict, bp_values_float, 'min_bp', 'mean_bp', 'max_bp', 'var_bp',
                                       'std_bp')
    else:
        data_not_avail_value(result_dict, 'min_bp', 'mean_bp', 'max_bp', 'var_bp', 'std_bp')


def set_temp_measurement(temp_values, result_dict):
    temp_values_float = [float(value) for value in temp_values]
    if len(temp_values_float) > 0:
        calculate_all_mathematical_cal(result_dict, temp_values_float, 'min_temp', 'mean_temp', 'max_temp',
                                       'var_temp', 'std_temp')
    else:
        data_not_avail_value(result_dict, 'min_temp', 'mean_temp', 'max_temp', 'var_temp', 'std_temp')


def set_lactate_measurement(lactate_values, result_dict):
    lactate_values_float = []

    for value in lactate_values:
        try:
            lactate_values_float.append(float(value))
        except ValueError:
            pass

    # diff_from_prev_to_recent = lactate_values[0] - lactate_values[-1]
    if len(lactate_values_float) >= 2:
        result_dict['lactate_diff'] = round(lactate_values_float[-1] - lactate_values_float[0], 2)
    else:
        result_dict['lactate_diff'] = 0


def set_cumulative_urine_output(cumulative_urine_output, result_dict):
    column_name = 'cum_urine_output'
    cum_sum = 0
    if len(cumulative_urine_output) > 0:
        cum_sum = cum_sum + cumulative_urine_output[-1]
        if cum_sum == 'nan':
            result_dict[column_name] = str(0) + " " + "ml"
        else:
            result_dict[column_name] = str(cum_sum) + " " + "ml"
    else:
        result_dict[column_name] = str(0) + " " + "ml"


def data_not_avail_value(result_dict, min_val, mean_val, max_val, var_val, std_val):
    result_dict[min_val] = 0
    result_dict[mean_val] = 0
    result_dict[max_val] = 0
    result_dict[var_val] = 0
    result_dict[std_val] = 0


def calculate_all_mathematical_cal(result_dict, values, col1, col2, col3, col4, col5):
    min_value = round(min(values), 2)
    result_dict[col1] = min_value

    mean_value = round(sum(values) / len(values), 2)
    result_dict[col2] = mean_value

    max_value = round(max(values), 2)
    result_dict[col3] = max_value

    if len(values) > 1:
        variance = round(statistics.variance(values), 2)
        result_dict[col4] = variance

    std = round(calculate_standard_deviation(values), 2)
    result_dict[col5] = std


def filter_timestamp_avail_in_lookup(df, current, next_timestamp):
    filtered_timestamps_df = df[
        (pd.to_datetime(df['charttime']) >= current) & (
                pd.to_datetime(df['charttime']) < next_timestamp)]
    return filtered_timestamps_df


def sort_by_col(df, col):
    if df is not None:
        return df.sort_values(by=col, ascending=True)


def calculate_standard_deviation(values):
    # Calculate the mean
    mean_value = sum(values) / len(values)

    # Calculate the squared differences from the mean
    squared_diff = [(x - mean_value) ** 2 for x in values]

    # Calculate the variance
    variance_value = sum(squared_diff) / len(values)

    # Calculate the standard deviation as the square root of the variance
    std_deviation = math.sqrt(variance_value)
    return std_deviation


def filter_rows_by_col_val_list(df, col_name, col_value_list):
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in the CSV file.")

    filtered_df = df[df[col_name].isin(col_value_list)]

    return filtered_df


def filter_rows_by_col_val(df, column_name, column_value):
    # check whether column name exist in the data frame or not
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")

    # Check if the filter value exists in the column
    if column_value in df[column_name].values:
        # If the filter value exists, apply the filter
        filtered_df = df[df[column_name] == column_value]
    else:
        # Return a copy of the original DataFrame
        filtered_df = df.copy()
    return filtered_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_patient_details()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
