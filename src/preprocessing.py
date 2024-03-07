from src.data import load_data, map_into_numeric, is_data_full, get_X_y


def preprocess_data(file_name: str, state_holiday_mapping: dict, prc_test_samples: float=.05):
    df = load_data(file_name)
    n_samples_test = int(prc_test_samples * len(df))
    train_df = df[:-n_samples_test]
    test_df = df[-n_samples_test:]

    train_df = map_into_numeric(train_df, {'StateHoliday': state_holiday_mapping})
    train_df = train_df.groupby('Date').mean()
    train_df.drop(['Store'], axis=1, inplace=True)
    assert is_data_full(train_df)

    test_df = map_into_numeric(test_df, {'StateHoliday': state_holiday_mapping})
    test_df = test_df.groupby('Date').mean()
    test_df.drop(['Store'], axis=1, inplace=True)
    assert is_data_full(test_df)

    X_train, y_train = get_X_y(train_df, input_size=1)
    X_test, y_test = get_X_y(test_df, input_size=1)

    return X_train, y_train, X_test, y_test