# -*- coding: utf-8 -*-
import os, math, time, copy, random, warnings, json, re
from typing import List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
