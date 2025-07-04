'''
from typing import List
import jieba
import torch

import pandas as pd

from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm

import transformers
transformers.logging.set_verbosity_error()

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class Tokenizer:
    """Tokenizer for Chinese given vocab.txt.

    Attributes:
        dictionary: Dict[str, int], {<word>: <index>}
    """
    # def __init__(self, vocab_file='vocab.txt'):
    #     """Initialize and build dictionary.

    #     Args:
    #         vocab_file: one word each line
    #     """
    #     self.dictionary = {'[PAD]': 0, '[UNK]': 1}
    #     count = 2
    #     with open(vocab_file, encoding='utf-8') as fin:
    #         for line in fin:
    #             word = line.strip()
    #             self.dictionary[word] = count
    #             count += 1

    # def __len__(self):
    #     return len(self.dictionary)

    # @staticmethod
    # def tokenize(sentence: str) -> List[str]:
    #     """Cut words for a sentence.

    #     Args:
    #         sentence: sentence

    #     Returns:
    #         words list
    #     """
    #     return jieba.lcut(sentence)

    # def convert_tokens_to_ids(
    #         self, tokens_list: List[str]) -> List[int]:
    #     """Convert tokens to ids.

    #     Args:
    #         tokens_list: word list

    #     Returns:
    #         index list
    #     """
    #     return [self.dictionary.get(w, 1) for w in tokens_list]
    def __init__(self, pretrained_name=r'/root/private_data/cail/model/bert/bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    def __len__(self):
        return self.tokenizer.vocab_size

    def tokenize(self, sentence: str) -> List[str]:
        return self.tokenizer.tokenize(sentence)

    def convert_tokens_to_ids(self, tokens_list: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens_list)

    def encode(self, sentence: str, add_special_tokens=True) -> List[int]:
        return self.tokenizer.encode(sentence, add_special_tokens=add_special_tokens)


class Data:
    """Data processor for BERT and RNN model for SMP-CAIL2020-Argmine.

    Attributes:
        model_type: 'bert' or 'rnn'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
                    Tokenizer for rnn
    """
    def __init__(self,
                 vocab_file=r'model\bert\bert-base-chinese',
                 max_seq_len: int = 512,
                 model_type: str = 'bert'):
        """Initialize data processor for SMP-CAIL2020-Argmine.

        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert' or 'rnn'
                If model_type == 'bert', use BertTokenizer as tokenizer
                Otherwise, use Tokenizer as tokenizer
        """
        self.model_type = model_type
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer(vocab_file)
        else:  # rnn
            self.tokenizer = Tokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def load_file(self,
                  file_path='SMP-CAIL2020-train.csv',
                  train=True) -> TensorDataset:
        """Load SMP-CAIL2020-Argmine train file and construct TensorDataset.

        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
            RNN model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length)
        """
        sc_list, bc_list, label_list = self._load_file(file_path, train)
        if self.model_type == 'bert':
            dataset = self._convert_sentence_pair_to_bert_dataset(
                sc_list, bc_list, label_list)
        else:  # rnn
            dataset = self._convert_sentence_pair_to_rnn_dataset(
                sc_list, bc_list, label_list)
        return dataset

    def load_train_and_valid_files(self, train_file, valid_file):
        """Load all files for SMP-CAIL2020-Argmine.

        Args:
            train_file, valid_file: files for SMP-CAIL2020-Argmine

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')
        print('Loading train records for valid...')
        valid_set_train = self.load_file(train_file, False)
        print(len(valid_set_train), 'train records loaded.')
        print('Loading valid records...')
        valid_set_valid = self.load_file(valid_file, False)
        print(len(valid_set_valid), 'valid records loaded.')
        return train_set, valid_set_train, valid_set_valid

    def _load_file(self, filename, train: bool = True):
        """Load SMP-CAIL2020-Argmine train/test file.

        For train file,
        The ratio between positive samples and negative samples is 1:4
        Copy positive 3 times so that positive:negative = 1:1

        Args:
            filename: SMP-CAIL2020-Argmine file
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            sc_list, bc_list, label_list with the same length
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: List[int], list of labels
        """
        data_frame = pd.read_csv(filename)
        sc_list, bc_list, label_list = [], [], []
        for row in data_frame.itertuples(index=False):
            candidates = row[3:8]
            answer = int(row[-1]) if train else None
            sc_tokens = self.tokenizer.tokenize(row[2])
            for i, _ in enumerate(candidates):
                bc_tokens = self.tokenizer.tokenize(candidates[i])
                if train:
                    if i + 1 == answer:
                        # Copy positive sample 4 times
                        for _ in range(len(candidates) - 1):
                            sc_list.append(sc_tokens)
                            bc_list.append(bc_tokens)
                            label_list.append(1)
                    else:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(0)
                else:  # test
                    sc_list.append(sc_tokens)
                    bc_list.append(bc_tokens)
        return sc_list, bc_list, label_list

    def _convert_sentence_pair_to_bert_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentence pairs to dataset for BERT model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            tokens = ['[CLS]'] + s1_list[i] + ['[SEP]']
            segment_ids = [0] * len(tokens)
            tokens += s2_list[i] + ['[SEP]']
            segment_ids += [1] * (len(s2_list[i]) + 1)
            if len(tokens) > self.max_seq_len:
                # tokens = tokens[:self.max_seq_len]
                # segment_ids = segment_ids[:self.max_seq_len]

                    # 保留特殊token
                special_tokens = ['[CLS]', '[SEP]']
                sep_indices = [j for j, token in enumerate(tokens) if token == '[SEP]']
                
                # 分别处理两个句子
                s1_tokens = tokens[1:sep_indices[0]]  # 不包括[CLS]和第一个[SEP]
                s2_tokens = tokens[sep_indices[0]+1:sep_indices[1]]  # 两个[SEP]之间的token
                
                # 合并两个句子用于TextRank
                combined_tokens = s1_tokens + s2_tokens
                
                # 使用TextRank抽取重要token
                important_tokens = self._textrank_extract_tokens(
                    combined_tokens, 
                    target_len=self.max_seq_len - 3  # 减去[CLS]和两个[SEP]
                )
                
                # 重新构建tokens和segment_ids
                # 尝试保持原有的句子结构
                s1_important = [token for token in important_tokens if token in s1_tokens]
                s2_important = [token for token in important_tokens if token in s2_tokens]
                
                # 如果抽取的token不够，按原比例分配
                if len(s1_important) + len(s2_important) < len(important_tokens):
                    remaining = len(important_tokens) - len(s1_important) - len(s2_important)
                    s1_ratio = len(s1_tokens) / len(combined_tokens)
                    s1_add = int(remaining * s1_ratio)
                    s2_add = remaining - s1_add
                    
                    # 从原句子中补充token
                    s1_remaining = [t for t in s1_tokens if t not in s1_important][:s1_add]
                    s2_remaining = [t for t in s2_tokens if t not in s2_important][:s2_add]
                    
                    s1_important.extend(s1_remaining)
                    s2_important.extend(s2_remaining)
                
                # 重新构建
                tokens = ['[CLS]'] + s1_important + ['[SEP]'] + s2_important + ['[SEP]']
                segment_ids = [0] * (len(s1_important) + 2) + [1] * (len(s2_important) + 1)
                
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (self.max_seq_len - tokens_len)
            segment_ids += [0] * (self.max_seq_len - tokens_len)
            input_mask += [0] * (self.max_seq_len - tokens_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)

    def _textrank_extract_tokens(self, tokens, target_len):
        """使用TextRank方法抽取重要的token"""
        if len(tokens) <= target_len:
            return tokens
        
        # 创建token的embedding（简化版本，实际可以使用预训练的词向量）
        # 这里使用简单的共现矩阵作为相似度
        token_to_idx = {token: i for i, token in enumerate(set(tokens))}
        unique_tokens = list(token_to_idx.keys())
        
        # 构建共现矩阵
        window_size = 2
        cooccurrence_matrix = np.zeros((len(unique_tokens), len(unique_tokens)))
        
        for i in range(len(tokens)):
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if i != j:
                    idx1 = token_to_idx[tokens[i]]
                    idx2 = token_to_idx[tokens[j]]
                    cooccurrence_matrix[idx1][idx2] += 1
        
        # 使用PageRank算法
        # 将共现矩阵转换为概率矩阵
        row_sums = cooccurrence_matrix.sum(axis=1)
        transition_matrix = np.divide(cooccurrence_matrix, row_sums[:, np.newaxis], 
                                     out=np.zeros_like(cooccurrence_matrix), 
                                     where=row_sums[:, np.newaxis]!=0)
        
        # 简化的PageRank实现
        damping = 0.85
        max_iter = 100
        tol = 1e-6
        
        n = len(unique_tokens)
        ranks = np.ones(n) / n
        
        for _ in range(max_iter):
            new_ranks = (1 - damping) / n + damping * transition_matrix.T.dot(ranks)
            if np.linalg.norm(new_ranks - ranks) < tol:
                break
            ranks = new_ranks
        
        # 获取每个token的重要性分数
        token_scores = {}
        for token, idx in token_to_idx.items():
            token_scores[token] = ranks[idx]
        
        # 按重要性排序并保持原有顺序
        important_tokens = []
        token_importance = [(token, token_scores[token]) for token in tokens]
        
        # 按重要性分数排序
        sorted_tokens = sorted(set(tokens), key=lambda x: token_scores[x], reverse=True)
        
        # 选择前target_len个重要token，但保持原有顺序
        selected_tokens = set(sorted_tokens[:target_len])
        important_tokens = [token for token in tokens if token in selected_tokens]
        
        # 如果还是太长，直接截断
        if len(important_tokens) > target_len:
            important_tokens = important_tokens[:target_len]
        
        return important_tokens


    def _convert_sentence_pair_to_rnn_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentences pairs to dataset for RNN model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
        """
        all_s1_ids, all_s2_ids = [], []
        all_s1_lengths, all_s2_lengths = [], []
        for i in tqdm(range(len(s1_list)), ncols=80):
            tokens_s1, tokens_s2 = s1_list[i], s2_list[i]
            all_s1_lengths.append(min(len(tokens_s1), self.max_seq_len))
            all_s2_lengths.append(min(len(tokens_s2), self.max_seq_len))
            if len(tokens_s1) > self.max_seq_len:
                tokens_s1 = tokens_s1[:self.max_seq_len]
            if len(tokens_s2) > self.max_seq_len:
                tokens_s2 = tokens_s2[:self.max_seq_len]
            s1_ids = self.tokenizer.convert_tokens_to_ids(tokens_s1)
            s2_ids = self.tokenizer.convert_tokens_to_ids(tokens_s2)
            if len(s1_ids) < self.max_seq_len:
                s1_ids += [0] * (self.max_seq_len - len(s1_ids))
            if len(s2_ids) < self.max_seq_len:
                s2_ids += [0] * (self.max_seq_len - len(s2_ids))
            all_s1_ids.append(s1_ids)
            all_s2_ids.append(s2_ids)
        all_s1_ids = torch.tensor(all_s1_ids, dtype=torch.long)
        all_s2_ids = torch.tensor(all_s2_ids, dtype=torch.long)
        all_s1_lengths = torch.tensor(all_s1_lengths, dtype=torch.long)
        all_s2_lengths = torch.tensor(all_s2_lengths, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths,
                all_label_ids)
        # test
        return TensorDataset(
            all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths)


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')


if __name__ == '__main__':
    test_data()
'''

"""
1. Tokenizer (used for RNN model):
    from data import Tokenizer
    vocab_file = 'vocab.txt'
    sentence = '我饿了，想吃东西了。'
    tokenizer = Tokenizer(vocab_file)
    tokens = tokenizer.tokenize(sentence)
    # ['我', '饿', '了', '，', '想', '吃', '东西', '了', '。']
    ids = tokenizer.convert_tokens_to_ids(tokens)
2. Data:
    from data import Data
    # For training, load train and valid set
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    datasets = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv', 'SMP-CAIL2020-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    datasets = data.load_all_files(
        'SMP-CAIL2020-train.csv', 'SMP-CAIL2020-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For testing, load test set
    data = Data('model/bert/vocab.txt', model_type='bert')
    test_set = data.load_file('SMP-CAIL2020-test.csv', train=False)
"""
from typing import List
import jieba
import torch

import pandas as pd

from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm


class Tokenizer:
    """Tokenizer for Chinese given vocab.txt.

    Attributes:
        dictionary: Dict[str, int], {<word>: <index>}
    """
    def __init__(self, vocab_file='vocab.txt'):
        """Initialize and build dictionary.

        Args:
            vocab_file: one word each line
        """
        self.dictionary = {'[PAD]': 0, '[UNK]': 1}
        count = 2
        with open(vocab_file, encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                self.dictionary[word] = count
                count += 1

    def __len__(self):
        return len(self.dictionary)

    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        """Cut words for a sentence.

        Args:
            sentence: sentence

        Returns:
            words list
        """
        return jieba.lcut(sentence)

    def convert_tokens_to_ids(
            self, tokens_list: List[str]) -> List[int]:
        """Convert tokens to ids.

        Args:
            tokens_list: word list

        Returns:
            index list
        """
        return [self.dictionary.get(w, 1) for w in tokens_list]


class Data:
    """Data processor for BERT and RNN model for SMP-CAIL2020-Argmine.

    Attributes:
        model_type: 'bert' or 'rnn'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
                    Tokenizer for rnn
    """
    def __init__(self,
                 vocab_file=r'model\bert\bert-base-chinese',
                 max_seq_len: int = 512,
                 model_type: str = 'bert'):
        """Initialize data processor for SMP-CAIL2020-Argmine.

        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert' or 'rnn'
                If model_type == 'bert', use BertTokenizer as tokenizer
                Otherwise, use Tokenizer as tokenizer
        """
        self.model_type = model_type
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer(vocab_file)
        else:  # rnn
            self.tokenizer = Tokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def load_file(self,
                  file_path='SMP-CAIL2020-train.csv',
                  train=True) -> TensorDataset:
        """Load SMP-CAIL2020-Argmine train file and construct TensorDataset.

        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
            RNN model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length)
        """
        sc_list, bc_list, label_list = self._load_file(file_path, train)
        if self.model_type == 'bert':
            dataset = self._convert_sentence_pair_to_bert_dataset(
                sc_list, bc_list, label_list)
        else:  # rnn
            dataset = self._convert_sentence_pair_to_rnn_dataset(
                sc_list, bc_list, label_list)
        return dataset

    def load_train_and_valid_files(self, train_file, valid_file):
        """Load all files for SMP-CAIL2020-Argmine.

        Args:
            train_file, valid_file: files for SMP-CAIL2020-Argmine

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')
        print('Loading train records for valid...')
        valid_set_train = self.load_file(train_file, False)
        print(len(valid_set_train), 'train records loaded.')
        print('Loading valid records...')
        valid_set_valid = self.load_file(valid_file, False)
        print(len(valid_set_valid), 'valid records loaded.')
        return train_set, valid_set_train, valid_set_valid

    def _load_file(self, filename, train: bool = True):
        """Load SMP-CAIL2020-Argmine train/test file.

        For train file,
        The ratio between positive samples and negative samples is 1:4
        Copy positive 3 times so that positive:negative = 1:1

        Args:
            filename: SMP-CAIL2020-Argmine file
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            sc_list, bc_list, label_list with the same length
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: List[int], list of labels
        """
        data_frame = pd.read_csv(filename)
        sc_list, bc_list, label_list = [], [], []
        for row in data_frame.itertuples(index=False):
            candidates = row[3:8]
            answer = int(row[-1]) if train else None
            sc_tokens = self.tokenizer.tokenize(row[2])
            for i, _ in enumerate(candidates):
                bc_tokens = self.tokenizer.tokenize(candidates[i])
                if train:
                    if i + 1 == answer:
                        # Copy positive sample 4 times
                        for _ in range(len(candidates) - 1):
                            sc_list.append(sc_tokens)
                            bc_list.append(bc_tokens)
                            label_list.append(1)
                    else:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(0)
                else:  # test
                    sc_list.append(sc_tokens)
                    bc_list.append(bc_tokens)
        return sc_list, bc_list, label_list

    def _convert_sentence_pair_to_bert_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentence pairs to dataset for BERT model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            tokens = ['[CLS]'] + s1_list[i] + ['[SEP]']
            segment_ids = [0] * len(tokens)
            tokens += s2_list[i] + ['[SEP]']
            segment_ids += [1] * (len(s2_list[i]) + 1)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
                segment_ids = segment_ids[:self.max_seq_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (self.max_seq_len - tokens_len)
            segment_ids += [0] * (self.max_seq_len - tokens_len)
            input_mask += [0] * (self.max_seq_len - tokens_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)

    def _convert_sentence_pair_to_rnn_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentences pairs to dataset for RNN model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
        """
        all_s1_ids, all_s2_ids = [], []
        all_s1_lengths, all_s2_lengths = [], []
        for i in tqdm(range(len(s1_list)), ncols=80):
            tokens_s1, tokens_s2 = s1_list[i], s2_list[i]
            all_s1_lengths.append(min(len(tokens_s1), self.max_seq_len))
            all_s2_lengths.append(min(len(tokens_s2), self.max_seq_len))
            if len(tokens_s1) > self.max_seq_len:
                tokens_s1 = tokens_s1[:self.max_seq_len]
            if len(tokens_s2) > self.max_seq_len:
                tokens_s2 = tokens_s2[:self.max_seq_len]
            s1_ids = self.tokenizer.convert_tokens_to_ids(tokens_s1)
            s2_ids = self.tokenizer.convert_tokens_to_ids(tokens_s2)
            if len(s1_ids) < self.max_seq_len:
                s1_ids += [0] * (self.max_seq_len - len(s1_ids))
            if len(s2_ids) < self.max_seq_len:
                s2_ids += [0] * (self.max_seq_len - len(s2_ids))
            all_s1_ids.append(s1_ids)
            all_s2_ids.append(s2_ids)
        all_s1_ids = torch.tensor(all_s1_ids, dtype=torch.long)
        all_s2_ids = torch.tensor(all_s2_ids, dtype=torch.long)
        all_s1_lengths = torch.tensor(all_s1_lengths, dtype=torch.long)
        all_s2_lengths = torch.tensor(all_s2_lengths, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths,
                all_label_ids)
        # test
        return TensorDataset(
            all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths)


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')


if __name__ == '__main__':
    test_data()
