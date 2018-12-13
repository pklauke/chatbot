# !/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os

import numpy as np
import pandas as pd

from src.data import datasets


def download_datasets():
    """Download datasets if not already done."""
    if not os.path.exists("__data__/cornell/movie_conversations.txt") \
            or not os.path.exists("__data__/cornell/movie_lines.txt"):
        subprocess.call(['scripts/download_cornell.sh'])
    if not os.path.isdir('__data__/opensubs'):
        subprocess.call(['scripts/download_opensubs.sh'])


def prepare_datasets(target_filename='data'):
    """Prepares the data sets. This includes downloading if necessary and combining them into one data set with shape
    (nrows, 2). That data set will be stored in feather format.

    :param target_filename: Filename of the combined data sets.
    :return: None
    """
    data_cornell = np.array(datasets.readCornellData('__data__/cornell/', max_len=1000000))
    data_opensubs = np.array(datasets.readOpensubsData('__data__/opensubs/', max_len=1000000))

    data = np.concatenate([data_cornell, data_opensubs], axis=0)
    del data_cornell, data_opensubs

    pd.DataFrame(data, columns=('question', 'answer')).to_feather('__data__/'+target_filename+'.feather')
