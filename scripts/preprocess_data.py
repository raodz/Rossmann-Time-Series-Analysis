from src.data import load_data, map_into_numeric, is_data_full, get_X_y

data_file = 'train.csv'

df = load_data(data_file)
n_samples_test = int(0.15*len(df))
train_df = df[:-n_samples_test]
test_df = df[-n_samples_test:]

state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}
# All types of holidays are mapped into 1

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
