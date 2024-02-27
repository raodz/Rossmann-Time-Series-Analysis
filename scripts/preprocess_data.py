from src.data import load_data, map_into_numeric, is_data_full

data_file = 'train.csv'

data = load_data(data_file)

state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}
# All types of holidays are mapped into 1
data = map_into_numeric(data, {'StateHoliday': state_holiday_mapping})

data = data.groupby('Date').mean()

assert is_data_full(data)


