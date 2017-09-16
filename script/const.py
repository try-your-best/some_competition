# -*- coding: utf-8 -*-

from enum import Enum
import pandas as pd
import numpy as np
import gc
import time
import random

UID = 'user_id'
PID = 'product_id'
OID = 'order_id'
DID = 'department_id'
AID = 'aisle_id'

UP = 'user_product'
UA = 'user_aisle'
UD = 'user_department'
LABEL = 'label'

ORDER_PRODUCT_PATH = '../input/order_products__prior_extend.csv'
ORDER_PATH = '../input/orders.csv'
INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
MODEL_DIR = '../model/'
PREDICT_DIR = '../prediction/'
STACKING_DIR = '../stacking/'
REORDER_INPUT_DIR = '../reorder_size_input/'
NONE_ORDER_INPUT_DIR = '../none_order_input/'
NEW_FEATURE_DIR = '../../imba-master/data/'

RAND_SEED = 0


class DataSetType(Enum):
	Train = 0
	Validate = 1
	Test = 2