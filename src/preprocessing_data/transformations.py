import pandas as pd


def map_into_numeric(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Map categorical values in a DataFrame to numeric values based on provided mappings."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if not isinstance(mappings, dict):
        raise TypeError("The input 'mappings' must be a dictionary.")

    for col_name in mappings:
        df = df.replace({col_name: mappings[col_name]})
    return df


def apply_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to the DayOfWeek column."""
    if 'DayOfWeek' not in df.columns:
        print("Warning: 'DayOfWeek' column not found. Cannot apply one-hot encoding.")
        return df

    df_encoded = df.copy()
    df_encoded['DayOfWeek'] = df_encoded['DayOfWeek'].astype(int)

    day_names = {
        1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday',
        5: 'Friday', 6: 'Saturday', 7: 'Sunday'
    }

    day_dummies = pd.get_dummies(df_encoded['DayOfWeek'], prefix='Day')

    new_column_names = {}
    for col in day_dummies.columns:
        try:
            day_num_str = col.split('_')[1]
            day_num = int(float(day_num_str))
            day_name = day_names.get(day_num)
            if day_name:
                new_column_names[col] = f'DayIs{day_name}'
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse column name {col}: {e}")
            continue

    day_dummies = day_dummies.rename(columns=new_column_names)
    df_encoded = df_encoded.drop(columns=['DayOfWeek'])
    df_encoded = pd.concat([df_encoded, day_dummies], axis=1)

    print(f"Applied one-hot encoding to days of week. Added {day_dummies.shape[1]} new columns.")

    return df_encoded