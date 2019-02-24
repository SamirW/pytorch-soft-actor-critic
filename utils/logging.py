"""From DK"""

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_key_from_log(path, key, index, flip=False):
    with open(path) as f:
        content = f.read().splitlines()

    data = []
    for line in content:
        if key in line:
            target_data = line.split()[index]
            if target_data[0] == "[":
                target_data = target_data[1:-1]

            if flip:
                data.append(-float(target_data))
            else:
                data.append(float(target_data))

    assert len(data) > 0

    return data


def to_panda(dict_df):
    return pd.DataFrame(data=dict_df)


def normalize(data, flip=False):
    if flip:
        data = [-sample for sample in data]

    # Normalize between 0 and 1
    max_v = np.max(data)
    min_v = np.min(data)

    data = (np.asarray(data) - min_v) / (max_v - min_v)
    return data


def tsplot(ax, datas, legends, title, filename, ylim_min, ylim_max):
    sns.set_style("ticks")

    for i_data, data in enumerate(datas):
        x = np.arange(data.shape[1]) * 50
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - std, mean + std)

        ax.fill_between(x, error[0], error[1], alpha=0.2)
        ax.plot(x, mean, label=legends[i_data])
        ax.margins(x=0)

    # plt.xlim(0, min([len(data[0, :]) for data in datas]))
    if ylim_min is not None and ylim_max is not None:
        plt.ylim(ylim_min, ylim_max)
    plt.xlabel("Train Iteration", size=13)
    plt.ylabel("Avg Evaluation Reward", size=13)
    plt.title(title, size=13)
    plt.legend()

    plt.savefig(filename)

def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(config, log_dir):
    log = {}                                                                                                                                      
    set_logger(
        logger_name=config.log_name, 
        log_file=str(log_dir / config.log_name))
    log[config.log_name] = logging.getLogger(config.log_name)

    # Log arguments
    for (name, value) in vars(config).items():
        log[config.log_name].info("{}: {}".format(name, value))

    return log