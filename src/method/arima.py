import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from src.util.util import proj_root_dir


# 三指数平滑方法
def run_method():
    # config
    plt.style.use('bmh')
    sns.set_style("whitegrid")
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    warnings.filterwarnings("ignore")
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # 导入数据集：
    data = pd.read_csv(str(proj_root_dir / 'data/data_for_tsa.csv'))
    data['date'] = pd.to_datetime(data['date'])
    print(data.head())

    train = data[data['date'] <= '2016-03-27']
    test = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]

    # plot data
    fig, ax = plt.subplots(figsize=(25, 5))
    train.plot(x='date', y='demand', label='Train', ax=ax)
    test.plot(x='date', y='demand', label='Test', ax=ax);

    predictions = pd.DataFrame()
    predictions['date'] = test['date']
    stats = pd.DataFrame(columns=['Model Name', 'Execution Time', 'RMSE'])

    # 开始调用具体方法
    t0 = time.time()
    model_name = 'ARIMA'
    arima_model = auto_arima(train['demand'], start_p=0, start_q=0,
                             max_p=14, max_q=3,
                             seasonal=False,
                             d=None, trace=True, random_state=2020,
                             error_action='ignore',  # we don't want to know if an order does not work
                             suppress_warnings=True,  # we don't want convergence warnings
                             stepwise=True)
    arima_model.summary()

    # train
    arima_model.fit(train['demand'])
    t1 = time.time() - t0
    # predict
    predictions[model_name] = arima_model.predict(n_periods=28)
    # visualize
    fig, ax = plt.subplots(figsize=(25, 4))
    train[-28:].plot(x='date', y='demand', label='Train', ax=ax)
    test.plot(x='date', y='demand', label='Test', ax=ax);
    predictions.plot(x='date', y=model_name, label=model_name, ax=ax);
    # evaluate
    score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))
    print('RMSE for {}: {:.4f}'.format(model_name, score))

    stats = stats.append({'Model Name': model_name, 'Execution Time': t1, 'RMSE': score}, ignore_index=True)

    print("stats: %s" % (stats,))
    plt.show()


if __name__ == '__main__':
    run_method()
