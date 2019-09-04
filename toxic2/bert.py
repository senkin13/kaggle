from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import json
import random
import logging
import argparse
from tqdm import tqdm, trange
import pandas as pd
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertLayer, BertIntermediate, BertAttention, \
    BertOutput, BertEmbeddings, BertPooler
import copy

from pytorch_pretrained_bert.optimization import BertAdam

from pytorch_pretrained_bert.optimization import BertAdam
from toolz.itertoolz import partition_all, concatv
from joblib import Parallel, delayed

import time
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_CSV='train_0607.csv'
VALID_CSV="val_5_0607.csv"
TEST_CSV='test_0607.csv'

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

weights = pd.read_csv('../input/' + TRAIN_CSV, usecols=['weight']).values
loss_weight = 1.0 / weights.mean()

def custom_loss_old(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2

def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,1:],targets[:,2:])
    return bce_loss_1 + bce_loss_2


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        print("BertEncoder layer init...", layer)
        # 按照config里设置的encoder层数复制n个encoder层
        # self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        encoders = []
        for _ in range(config.num_hidden_layers):
            encoders.append(copy.deepcopy(layer))
       
        # 第13个encoder
        encoders.append(copy.deepcopy(layer))
        self.layer = nn.ModuleList(encoders)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        # print("encoder self.layer",len(self.layer))
        for layer_module in self.layer:  # 遍历所有的encoder层，获得每一层的hidden_states
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        print("BertModel init bert weights")
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, y_aux=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.y_aux = y_aux


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, y_aux):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.y_aux = y_aux


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        train = pd.read_csv(input_file,
                            usecols=['id', 'target', 'comment_text', 'male', 'female', 'homosexual_gay_or_lesbian',
                                     'christian', 'jewish',
                                     'muslim', 'black', 'white', 'psychiatric_or_mental_illness', 'severe_toxicity',
                                     'obscene', 'identity_attack', 'insult', 'threat'])

        y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
        print("y_aux_train", y_aux_train.shape)
        # Overall
        # weights = np.ones((len(train),)) / 4
        # # Subgroup
        # weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
        # # Background Positive, Subgroup Negative
        # weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
        #              (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(
        #                  np.int)) > 1).astype(bool).astype(np.int) / 4
        # # Background Negative, Subgroup Positive
        # weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
        #              (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(
        #                  np.int)) > 1).astype(bool).astype(np.int) / 4
        # y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
        weights = pd.read_csv('../input/' + TRAIN_CSV, usecols=['weight']).values
        print ('load weights ok')
        y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights[:len(train)].reshape(-1,)]).T
        print("y_train", y_train.shape)
        # 遍历csv文件,每行一个turple
        # print("quotechar",quotechar)
        start = time.time()
        text_list = train['comment_text'].astype(str).values
        # targets=train['target'].values
        # 为Y增加一列y_aux_train
        for i, text in enumerate(text_list):
            # dicts.append((text, targets[i]))
            # print(y_aux_train[i].shape)
            dicts.append((text, y_train[i], y_aux_train[i]))
        # print("for loop time:{}".format(time.time()-start))
        # for row in df.itertuples(index=True, name='Pandas'):
        #     dicts.append((getattr(row, "comment_text"), getattr(row, "target")))
        print("for loop time:{}".format(time.time() - start))
        return dicts

    @classmethod
    def _read_test_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        df = pd.read_csv(input_file)
        df['comment_text'] = df['comment_text'].astype(str)
        # 遍历csv文件,每行一个turple
        for row in df.itertuples(index=True, name='Pandas'):
            # print(np.asarray([0,0,0,0,0,0]).shape)
            dicts.append((getattr(row, "comment_text"), 0, np.asarray([0, 0, 0, 0, 0, 0])))

        return dicts


class Head(nn.Module):
    """The MLP submodule"""

    def __init__(self, bert_hidden_size: int):
        super().__init__()

        self.bert_hidden_size = bert_hidden_size * (1 + 2 + 1)

        self.fc = nn.Sequential(

            nn.Dropout(0.1),  # 0.1

            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            nn.ReLU(),

            nn.Linear(self.bert_hidden_size, 1)

        )

    def forward(self, bert_outputs):
        return self.fc(bert_outputs.squeeze(1))


class Head2(nn.Module):
    """The MLP submodule"""

    def __init__(self, bert_hidden_size: int):
        super().__init__()

        self.bert_hidden_size = bert_hidden_size * (1 + 2 + 1)

        self.fc = nn.Sequential(

            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            nn.BatchNorm1d(self.bert_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(self.bert_hidden_size, 1)

        )

    def forward(self, bert_outputs):
        return self.fc(bert_outputs.squeeze(1))


class ToxicModel(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str, device: torch.device):
        super().__init__()
        self.device = device
        if bert_model in ("bert-base-uncased", "bert-base-cased", '../bert_feature/models/pytorch-bert-base-uncased'):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            # raise ValueError("Unsupported BERT model.")
            if bert_model.__contains__('base'):
                self.bert_hidden_size = 768
            elif bert_model.__contains__('large'):
                self.bert_hidden_size = 1024
#             elif bert_model.__contains__('toxic'):#为在老toxic上的预训练模型做特别处理
#                 self.bert_hidden_size = 768
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.head = Head2(self.bert_hidden_size).to(device)
        # self.linear_aux_out = nn.Linear(2, 6)
        self.linear_aux_out = nn.Linear(self.bert_hidden_size, 6)  # 6
        # self.ssa = StructuredSelfAttention(self.bert_hidden_size )
        # self.apply(self.init_bert_weights)

    def avg_pooling(self, encoder_tensor):
        """
        Inputs:
            `encoder_tensor`: encoded-hidden-state -> a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]

        Outputs:
            `global_avg_pool_cnn`: pooled tensor -> [batch_size, hidden_size]
        """
        global_avg_pool_cnn = torch.mean(encoder_tensor, 1)
        return global_avg_pool_cnn

    def max_pooling(self, encoder_tensor):
        """
        Inputs:
            `encoder_tensor`: encoded-hidden-state -> a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]

        Outputs:
            `global_avg_pool_cnn`: pooled tensor -> [batch_size, hidden_size]
        """
        global_max_pool_cnn, _ = torch.max(encoder_tensor, 1)
        return global_max_pool_cnn

    def avg_max_pooling(self, encoder_tensor):
        """
        Inputs:
            `encoder_tensor`: encoded-hidden-state -> a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]

        Outputs:
            `cat pooled tensor`: [batch_size, hidden_size*2]
        """
        global_max_pool_cnn, _ = torch.max(encoder_tensor, 1)  # batch x hidden
        global_avg_pool_cnn = torch.mean(encoder_tensor, 1)
        return torch.cat([global_max_pool_cnn, global_avg_pool_cnn], dim=1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                 output_all_encoded_layers=True)

        pooled_output = pooled_output.to(self.device)
        # 最后一层encoder的pooling -> [batch_size, hidden_size]
        pooled_last_1 = self.avg_pooling(encoded_layer[-1])

        # 最后一层encoder的pooling -> [batch_size, hidden_size]
        pooled_last_11 = self.max_pooling(encoded_layer[-1])
        # print('pooled_last_11', pooled_last_11.size())
        # 倒数第二层encoder的CLS标志
        pooled_last_12 = encoded_layer[-2][:, 0]

        cat_sentence_embedding = torch.cat([pooled_last_1, pooled_last_11, pooled_last_12, pooled_output],
                                           dim=1)
        logits = self.head(cat_sentence_embedding)
        if labels is not None:

            aux_result = self.linear_aux_out(pooled_output)
            out = torch.cat([logits, aux_result], 1)
            loss = custom_loss(out, labels)

            return loss
        else:
            # print("label is null...")
            return logits


class MyPro(DataProcessor):
    '''自定义数据读取方法，针对csv文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, TRAIN_CSV)), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, VALID_CSV)), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(

            self._read_test_csv(os.path.join(data_dir, TEST_CSV)), 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, dicts, set_type):
        examples = []
        for (i, (text_a, label, y_aux)) in enumerate(dicts):
            # print(label.shape)
            # print(y_aux.shape)
            guid = "%s-%s" % (set_type, i)

            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, y_aux=y_aux))  # qin hui add
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    '''Loads a data file into a list of `InputBatch`s.

    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法

    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):  # qin hui add
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # print("example.label",example.label)
        # label_id = label_map[example.label]
        label_id = example.label  # [label_id,weight]
        # print("label_id",label_id.shape)
        #         if ex_index < 5 and show_exp:
        #             logger.info("*** Example ***")
        #             logger.info("guid: %s" % (example.guid))
        #             logger.info("tokens: %s" % " ".join(
        #                     [str(x) for x in tokens]))
        #             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #             logger.info(
        #                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #             logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id, y_aux=example.y_aux))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, oof_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[oof_name])


def compute_bpsn_auc(df, subgroup, label, oof_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[oof_name])


def compute_bnsp_auc(df, subgroup, label, oof_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[oof_name])


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def calculate_overall_auc(df, oof_name):
    true_labels = df['target']
    predicted_labels = df[oof_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def val(model, processor, data_dir, max_seq_length, eval_batch_size, label_list, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    eval_examples = processor.get_dev_examples(data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        np.hstack([np.asarray([f.label_id for f in eval_features]), np.asarray([f.y_aux for f in eval_features])]),
        dtype=torch.float32)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=4,
                                 pin_memory=True, )

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    # gt=[]
    # predict=[]
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = torch.sigmoid(logits).cpu().numpy()

            # predict = np.hstack((predict, pred[:, 1]))
            # print(pred)
            predict = np.hstack((predict, pred[:, 0]))
            # predict.extend(pred[:, 1])

            gt = np.hstack((gt, label_ids[:, 0].cpu().numpy()))

            # gt.extend( label_ids[:, 0].cpu().numpy())

    # gt=np.asarray(gt).squeeze()
    # predict=np.asarray(predict)
    print(gt.shape, predict.shape)
    print("log loss", np.mean(metrics.log_loss(gt, predict)))
    # print("acc", np.mean(metrics.accuracy_score(gt, predict)))
    # print("f1", np.mean(metrics.f1_score(gt, predict, average=None)))
    auc = np.mean(metrics.roc_auc_score(gt, predict, average=None))
    # print("auc",auc)
    print('auc score in text set is {}'.format(auc))
    oof_name = 'predicted_target'
    validate_df = pd.read_csv(os.path.join(data_dir, VALID_CSV))
    validate_df['comment_text'].astype(str).fillna('na', inplace=True)
    for col in identity_columns:
        validate_df[col] = np.where(validate_df[col] >= 0.5, True, False)  # 必须转成True,False
    validate_df[oof_name] = predict
    validate_df['target'] = gt.astype(np.int32)
    # validate_df.to_csv('valid_test_cust_loss.csv',index=False)
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, oof_name, 'target')
    f_auc = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, oof_name))
    print("final auc", f_auc)
    return f_auc


def test(model, processor, data_dir, max_seq_length, eval_batch_size, label_list, tokenizer, device):
    '''模型测试

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    test_examples = processor.get_test_examples(data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    print(np.asarray([f.label_id for f in test_features]).shape)
    print(np.asarray([f.y_aux for f in test_features]).shape)
    all_label_ids = torch.tensor(
        np.hstack([np.asarray([f.label_id for f in test_features]).reshape((-1, 1)),
                   np.asarray([f.y_aux for f in test_features])]),
        dtype=torch.float32)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1,
                                 pin_memory=True, )

    model.eval()
    predict = np.zeros((0,), dtype=np.float32)  # 取标签

    # gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, _ in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = torch.sigmoid(logits).cpu().numpy()
            predict = np.hstack((predict, pred[:, 0]))

    nrows = len(test_examples)
    sub = pd.read_csv('../input/sample_submission.csv', nrows=nrows)
    # print(predict.shape)
    sub.prediction = predict
    # print("logit",logits.size())
    # sub.prediction = logits.cpu().numpy()
    sub.to_csv("submission.csv".format(TRAIN_CSV), index=False)
    # return auc


# copy from https://www.kaggle.com/gdoteof/pytorch-bert-baseline-wd-epochs-layers/notebook
def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


# copy from https://www.kaggle.com/gdoteof/pytorch-bert-baseline-wd-epochs-layers/notebook
def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


# copy from https://www.kaggle.com/gdoteof/pytorch-bert-baseline-wd-epochs-layers/notebook
def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


# copy from https://www.kaggle.com/gdoteof/pytorch-bert-baseline-wd-epochs-layers/notebook
def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def main():
    # prameters
    data_dir = '../input/'
    bert_model = '../bert-large-wwm-uncased' # 把自己large wwm模型路径取代这个bert_model
#     bert_model = 'bert-base-uncased'
#     bert_model = './oldtoxic'#使用在老toxic上训练好的预训练模型权重.下载路径:https://www.kaggle.com/qinhui1999/old-toxic-bert-v2
    task_name = 'MyPro'
    output_dir = 'checkpoints/'
    model_save_pth = 'checkpoints/bert_large_wwm.pth'
    max_seq_length = 220
    do_train = True
    do_eval = True
    do_lower_case = True
    train_batch_size = 56
    eval_batch_size = 200
    learning_rate = 1e-5
    num_train_epochs = 1
    warmup_proportion = 0.05
    no_cuda = False
    local_rank = -1
    seed = 42
    gradient_accumulation_steps = 8
    optimize_on_cpu = False
    fp16 = False
    save_checkpoints_steps = 50000
    loss_scale = 128

    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro}

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if fp16:
            logger.info("16-bits training currently not supported in distributed training")
            fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(local_rank != -1))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
        print('The checkpoint directory is aleady existed...')
    else:
        os.makedirs(output_dir, exist_ok=True)

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    # print("tokenizer",tokenizer)
    train_examples = None
    num_train_steps = None
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    # Prepare model
    # model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2,
    #             cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank))
    model = ToxicModel(bert_model, device)
    # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
    # set_trainable(model.bert, False)
    # 锁定embedding层
    #     set_trainable(model.bert.embeddings, False)
    # set_trainable(model.bert.encoder.layer[11], True)
    # set_trainable(model.head, True)
    # model.load_state_dict(torch.load('checkpoints/bert_classification_2epoch.pth')['state_dict'])
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) for n, param in
                           model.named_parameters()]
    elif optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) for n, param in
                           model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if do_train:

        if os.path.exists('train.token_new_cleaned_wwm.npy'):
            train_features = np.load('train.token_new_cleaned_wwm.npy',allow_pickle=True)
        else:
            parallel = Parallel(300, backend="multiprocessing", verbose=5)
            train_features = list(concatv(*parallel(
                delayed(convert_examples_to_features)(example, label_list, max_seq_length, tokenizer) for example in
                list(partition_all(300, train_examples)))))
            train_features = np.asarray(train_features)
            np.save('train.token_new_cleaned_wwm', train_features)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        torch.cuda.empty_cache()
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        print('y_aux', np.asarray([f.y_aux for f in train_features]).shape)
        all_label_ids = torch.tensor(np.hstack(
            [np.asarray([f.label_id for f in train_features]), np.asarray([f.y_aux for f in train_features])]),
            dtype=torch.float32)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, num_workers=2,
                                      pin_memory=True, )
        
        #model.load_state_dict(torch.load('checkpoints/bert_large_wwm.pth')['state_dict'])
        # model.load_state_dict(torch.load('checkpoints/0_80000_iterations.pth')['state_dict'])

        model.train()
        best_score = 0
        flags = 0
        torch.cuda.empty_cache()
        ''' 
        model.load_state_dict(torch.load('checkpoints/0_20000_iterations.pth')['model'])
        optimizer.load_state_dict(torch.load('checkpoints/0_20000_iterations.pth')['optimizer'])
        old_iter = int(torch.load('checkpoints/0_20000_iterations.pth')['iteration'])
        '''
        old_iter = -1

        for i_epoch in trange(int(num_train_epochs), desc="Epoch"):
            torch.cuda.empty_cache()
            iteration = 0  # counter
            save_point = save_checkpoints_steps  # 10000
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if iteration <= old_iter: iteration += 1; continue
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                torch.cuda.empty_cache()
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if fp16 and loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * loss_scale
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16 or optimize_on_cpu:
                        if fp16 and loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            loss_scale = loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                #Save model
                if iteration % save_point == 0 and iteration > 0:
                    checkpoint = {
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}
                
                    checkpoint_path = os.path.join(
                        output_dir, '{}_{}_iterations.pth'.format(i_epoch, iteration))
                
                    torch.save(checkpoint, checkpoint_path)
                    logging.info('Model saved to {}'.format(checkpoint_path))
                    val(model, processor, data_dir, max_seq_length, eval_batch_size, label_list, tokenizer, device)
                
                iteration += 1


    checkpoint = {
        'state_dict': model.state_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()

    }
    torch.save(checkpoint, model_save_pth)
    val(model, processor, data_dir, max_seq_length, eval_batch_size, label_list, tokenizer, device)

    test(model, processor, data_dir, max_seq_length, eval_batch_size, label_list, tokenizer, device)


if __name__ == '__main__':
    main()
