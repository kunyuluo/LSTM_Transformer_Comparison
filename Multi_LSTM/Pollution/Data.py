import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def check_linearity(
        df: pd.DataFrame,
        column_name_1: str,
        column_name_2: str,
        switch_table: bool = False,
        log_transform: bool = False):
    """
    Check the linearity of the data.
    """
    data_1 = df[column_name_1]
    data_2 = df[column_name_2]

    if log_transform:
        data_1 = np.log(data_1)
        # data_2 = np.log(data_2)

    x_label = column_name_1
    y_label = column_name_2

    if switch_table:
        data_1, data_2 = data_2, data_1
        x_label, y_label = y_label, x_label

    plt.scatter(data_1, data_2, s=2)
    plt.title('Linearity between \'{}\' and \'{}\''.format(column_name_1, column_name_2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


dataset = pd.read_csv('pollution.csv', parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

dataset['pollution'].fillna(0, inplace=True)

# drop the first 24 hours
dataset = dataset[24:]

# summarize first 5 rows
print(dataset.head(5))
dataset.to_csv('new_pollution.csv')
# check_linearity(dataset, 'pm2.5', 'DEWP', switch_table=True)
