from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.optimization import (
    get_linear_schedule_with_warmup, get_constant_schedule)
from torch.optim import AdamW
from data import Data
from evaluate import evaluate, calculate_accuracy_f1, get_labels_from_file
from model import BertForClassification, BertClassifierv2, BertClassifierv3, BertClassifierv4
from utils import get_csv_logger, get_path
from vocab import build_vocab
from transformers import AutoTokenizer, BertTokenizer
# from plt import pic


MODEL_MAP = {
    'bert': BertClassifierv3
}

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

from train import Trainer


# 1. Load data
from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace
import pandas as pd

with open(r"/root/private_data/cail/config/bert_config.json") as fin:
    config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

model = BertClassifierv4(config)
model.load_state_dict(torch.load(r"/root/private_data/cail/model/bert/BERT/model.bin", map_location="cuda"))
model.to("cuda")
model.eval()

tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)

df = pd.read_csv(r"/root/private_data/cail/valid.csv")

all_preds = []
for _, row in df.iterrows():

    # 编码文本
    encoded = tokenizer(
        row['a'], row['b'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    token_type_ids = encoded["token_type_ids"].to("cuda")

    # 模型预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        pred = logits.argmax(dim=1).item()

    all_preds.append(pred)

# 添加预测列并保存
df["prediction"] = all_preds
df.to_csv("valid_with_predictions.csv", index=False)