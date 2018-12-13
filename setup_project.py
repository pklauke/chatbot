# !/usr/bin/env python
# -*- coding: utf-8 -*-

from src.data.load import download_datasets, prepare_datasets


if __name__ == '__main__':
    download_datasets()
    prepare_datasets()
