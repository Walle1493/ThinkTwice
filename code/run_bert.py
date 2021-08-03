# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open
# from typing import OrderedDict
# from transformers45.models import albert
from collections import OrderedDict

import copy
import numpy as np
from packaging.version import parse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
# from model import BertForQuestionAnswering
# from transformers45.models.albert.modeling_albert import AlbertForMultiTask
# from transformers import AlbertTokenizer
from model import BertForQuestionAnswering2, BertForQuestionAnswering3
from transformers import BertTokenizer

import pdb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 enter_position=None,
                 ans_in_para_index=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.enter_position = enter_position
        self.ans_in_para_index=ans_in_para_index

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 span_na=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.span_na = span_na


def reconstruct_train_examples(train_examples, qas_id_topk_dict):
    # 为train_examples重新构建
    # (1) 拼接doc_tokens
    # (2) 平移start_position和end_position
    # (3) 将enter_position置为[]，表示该短文本只有一个段落，后面直接截断
    train_examples_new = copy.deepcopy(train_examples)
    for example_index, example in enumerate(train_examples_new):
        span_to_logits_dict = qas_id_topk_dict[example_index]   # doc_span_index -> (logtis, span_na)
        doc_span_indexes = list(span_to_logits_dict.keys())   # 取出k个doc_span_index
        logits_span_nas = list(span_to_logits_dict.values()) # [(logits, span_na), (logits, span_na), ...]
        split_doc_tokens = [] # 存放doc_span_index对应的doc_tokens
        doc_tokens = example.doc_tokens
        enter_position = example.enter_position
        for idx in range(len(doc_span_indexes)):
        # for doc_span_index in doc_span_indexes: # 对于每个dco_span_index，找到它的开始和结束位置(word)
            doc_span_index = doc_span_indexes[idx]  # 遍历
            logits_span_na = logits_span_nas[idx]   # 遍历
            if logits_span_na[1] == 0:  # 可回答，需要更改答案位置
                if doc_span_index == 0:
                    rel_offset = example.start_position
                    # end_offset = example.end_position
                else:
                    rel_offset = example.start_position - enter_position[doc_span_index - 1]    # 开始位置相对开头的偏移
                    # if doc_span_index >= len(enter_position):
                    #     off1 = example.end_position - enter_position[doc_span_index - 1]
                    #     off2 = len(doc_tokens) - enter_position[doc_span_index - 1]
                    #     end_offset = min(off1, off2)
                    # else:
                    #     off1 = example.end_position - enter_position[doc_span_index - 1]
                    #     off2 = enter_position[doc_span_index] - enter_position[doc_span_index - 1] - 1
                    #     end_offset = min(off1, off2)
                split_length = len(split_doc_tokens)    # 新doc_token已有的长度
                new_start_position = split_length + rel_offset
                new_end_position = new_start_position + (example.end_position - example.start_position)
                # new_end_position = split_length + end_offset
                example.start_position = new_start_position
                example.end_position = new_end_position
            if enter_position == []:    # 原文只有一句
                doc_start = 0
                doc_end = len(doc_tokens) - 1
            elif doc_span_index >= len(enter_position):   # 最后一句
                doc_start = enter_position[-1]
                doc_end = len(doc_tokens) - 1   # (包含)
            elif doc_span_index == 0:   # 第一句
                doc_start = 0
                doc_end = enter_position[doc_span_index] - 1
            else:   # 其他情况
                doc_start = enter_position[doc_span_index - 1]
                doc_end = enter_position[doc_span_index] - 1
            split_doc_tokens.extend(doc_tokens[doc_start: doc_end + 1]) # 用extend直接加上k句 (1)
        example.doc_tokens = split_doc_tokens   # (1)
        example.enter_position = [] # (3)
    
    return train_examples_new


def reconstruct_eval_examples(eval_examples, qas_id_topk_dict, qas_id_top10_dict):
    # 为eval_examples重新构建
    eval_examples_new = copy.deepcopy(eval_examples)
    hits_num = [0] * 10
    dev_num = [0] * 10
    hits_rate = [0] * 10
    for example_index, example in enumerate(eval_examples_new):
        span_to_logits_dict = qas_id_topk_dict[example_index]   # doc_span_index -> logtis
        doc_span_indexes = list(span_to_logits_dict.keys())   # 取出k个doc_span_index
        span_to_logits_dict_10 = qas_id_top10_dict[example_index]
        doc_span_indexes_10 = list(span_to_logits_dict_10.keys())   # 10个doc_span_index(从大到小排)
        split_doc_tokens = [] # 存放doc_span_index对应的doc_tokens
        doc_tokens = example.doc_tokens
        enter_position = example.enter_position
        for doc_span_index in doc_span_indexes: # 对于每个dco_span_index，找到它的开始和结束位置(word)
            if enter_position == []:    # 原文只有一句
                doc_start = 0
                doc_end = len(doc_tokens) - 1
            elif doc_span_index >= len(enter_position):   # 最后一句
                doc_start = enter_position[-1]
                doc_end = len(doc_tokens) - 1   # (包含)
            elif doc_span_index == 0:   # 第一句
                doc_start = 0
                doc_end = enter_position[doc_span_index] - 1
            else:   # 其他情况
                doc_start = enter_position[doc_span_index - 1]
                doc_end = enter_position[doc_span_index] - 1
            split_doc_tokens.extend(doc_tokens[doc_start: doc_end + 1]) # 用extend直接加上k句
        example.doc_tokens = split_doc_tokens
        example.enter_position = []
        
        # 计算命中率-选择的k个句子包含答案
        # if example.ans_in_para_index in doc_span_indexes:
        #     hits_num += 1
        # dev_num += 1
        # 对k=1-10的情况都算hit@k
        for k in range(1, 11):
            # ans_in weiyu doc_index_10
            if example.ans_in_para_index in doc_span_indexes_10[:k]:
                hits_num[k-1] += 1
            dev_num[k-1] += 1
    for k in range(1, 11):
        hits_rate[k-1] = hits_num[k-1] / dev_num[k-1]

    # logger.info("hits_rate: %f", 1.0 * hits_num / dev_num)
        # eval_features_gen = (f for f in eval_features)  # 将eval_features改成生成器
        # while True: # 遍历eval_features
        #     try:
        #         feature = next(eval_features_gen)
        #         if feature.example_index == example_index:
        #             # 在feature中找到与doc_span_indexes对应的doc_tokens(word tokens)
        #             if feature.doc_span_index in doc_span_indexes:
        #                 tokens = feature.tokens # feature中的tokens
        #                 token_to_orig_map = feature.token_to_orig_map   # token在word中的位置
        #                 long_doc_tokens = example.doc_tokens    # example中的word
        #                 pass    
        #     except:
        #         break
    
    return eval_examples_new, hits_rate


def split_paragraphs(input_data):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def is_enter(c):
        if c == "\n":
            return True
        return False
    
    for entry in input_data:
        new_paragraphs = []
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            qas = paragraph["qas"]
            doc_tokens = [] # 按空格分隔的tokens
            char_to_word_offset = [] # 每个char所属的word位置（含空格）
            enter_char_position = [] # 记录换行符后一个word的位置，不包含末尾
            prev_is_whitespace = True

            # 寻找换行符
            for c_idx, c in enumerate(paragraph_text):
                if is_enter(c):
                    enter_char_position.append(c_idx)
            # 连续分布的第一个\n
            first_enter_char_position = []
            if enter_char_position:
                first_enter_char_position.append(enter_char_position[0])
            i = 1
            while i < len(enter_char_position):
                if enter_char_position[i] - enter_char_position[i - 1] == 1:
                    i += 1
                    continue
                else:
                    first_enter_char_position.append(enter_char_position[i])
                i += 1
            # 连续分布的最后一个\n
            last_enter_char_position = []
            if enter_char_position:
                last_enter_char_position.append(enter_char_position[-1])
            j = len(enter_char_position) - 1
            while j >= 0:
                if enter_char_position[j] - enter_char_position[j - 1] == 1:
                    j -= 1
                    continue
                else:
                    last_enter_char_position.insert(0, enter_char_position[j])
                j -= 1
            first_enter_char_position.append(len(paragraph_text))
            last_enter_char_position.append(len(paragraph_text))
            new_paragraph = {}  # 分段后的context + qas
            para_start_pos = 0  # 某段落的开始位置
            k = 0
            while k < len(first_enter_char_position):
                para_end_pos = first_enter_char_position[k] - 1 # 某段落的结束位置（包含）
                para = paragraph_text[para_start_pos: para_end_pos + 1] # 切分后的段落
                for qa in qas:
                    qa["is_impossible"] = True
                    for answer in qa["answers"]:
                        answer_start = answer["answer_start"]
                        answer_length = len(answer["text"])
                        if answer_start >= para_start_pos:
                            if answer_start + answer_length <= para_end_pos:
                                answer["answer_start"] = answer_start - para_start_pos  # 平移答案位置
                                qa["is_impossible"] = False # 答案落入段落中，说明可回答
                            # else:
                            #     print(qa["id"])
                            #     print(para_start_pos, para_end_pos)
                            #     print(answer_start, answer_length)
                            #     raise Exception("answer out of paragraph")
                para_start_pos = last_enter_char_position[k] + 1
                new_paragraph["context"] = para
                new_paragraph["qas"] = qas
                new_paragraphs.append(new_paragraph)
                k += 1
        entry["paragraphs"] = new_paragraphs
    
    return input_data

def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    # input_data = split_paragraphs(input_data)
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    
    def is_enter(c):
        if c == "\n":
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = [] # 按空格分隔的tokens
            char_to_word_offset = [] # 每个char所属的word位置（含空格）
            enter_position = [] # 记录换行符后一个word的位置，不包含末尾
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
                if is_enter(c):
                    enter_pos = len(doc_tokens) # 不+1，则记录后一个word的位置
                    if enter_pos not in enter_position:
                        enter_position.append(enter_pos)    # 记录换行符后面一个word的位置

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None   # 答案在doc_tokens中的开始索引
                end_position = None     # include
                orig_answer_text = None
                is_impossible = False
                ans_in_para_index = None    # new task for whether answer in k-paras
                # k_para_index = []
                if True:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if is_training and (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                
                # 为eval增加特殊属性，用于统计第一阶段命中率
                if not is_training:
                    for enter_index, enter_pos in enumerate(enter_position):
                        if start_position < enter_pos:
                            ans_in_para_index = enter_index # 赋予para索引
                            break
                    else:
                        ans_in_para_index = len(enter_position) # 若都不在，则返回最后一段，即总长
                    
                    start_position = None
                    end_position = None

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,  # 缩短成top_k句
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,  # 训练：平移变换
                    end_position=end_position,  # 训练：平移变换
                    is_impossible=is_impossible,
                    enter_position=enter_position,  # []
                    ans_in_para_index=ans_in_para_index)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # delete doc_stride
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []  # token在word中的位置
        orig_to_tok_index = []  # word在token中的位置
        all_doc_tokens = [] # token（用tokenizer分词）
        for (i, token) in enumerate(example.doc_tokens):
            span_na = example.is_impossible
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None   # 答案在all_doc_tokens中的开始索引
        tok_end_position = None     # include
        tok_enter_position = []   # 换行符在all_doc_tokens中的索引，需要包含末尾
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        for enter_position in example.enter_position:
            tok_enter_position.append(orig_to_tok_index[enter_position] - 1)    # 后面tok位置的前一个
        tok_enter_position.append(len(all_doc_tokens) - 1)  # 加入tok末尾的位置

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])
        doc_spans = []
        # start_offset = 0
        # while start_offset < len(all_doc_tokens):
        #     length = len(all_doc_tokens) - start_offset
        #     if length > max_tokens_for_doc:
        #         length = max_tokens_for_doc
        #     doc_spans.append(_DocSpan(start=start_offset, length=length))
        #     if start_offset + length == len(all_doc_tokens):
        #         break
        #     start_offset += min(length, doc_stride)
        # 分句：每1个句子属于一个片段
        doc_start = 0
        for doc_end in tok_enter_position:
            doc_length = min(doc_end - doc_start + 1, max_tokens_for_doc)
            doc_spans.append(_DocSpan(start=doc_start, length=doc_length))
            doc_start = doc_end + 1

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}  # 每个span中，doc的索引对应于word的索引
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length, "input_ids:{}\n, example_index:{}\n, doc_span_index:{}\n".format(input_ids, example_index, doc_span_index)
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_na = 1
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    span_na = 0
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < 2:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                    span_na=span_na))
            unique_id += 1
    return features


def convert_examples_to_features2(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        
        tok_to_orig_index = []  # for each token, store its word index
        orig_to_tok_index = []  # for each word, store its token index
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            span_na = example.is_impossible
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        
        tok_start_position = None   # answer token's strat position
        tok_end_position = None # answer token's end position(include)
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)   # try to maximum the context of answer span

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])
        doc_spans = []  # store some _DocSpan(s)
        start_offset = 0
        while start_offset < len(all_doc_tokens):   # create sliding chunks
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans): # sliding chunks
            tokens = []
            token_to_orig_map = {}  # for each token, find its index in original word
            token_is_max_context = {}   # check whether current token surrounds max context
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")  # [CLS] + [token] * query_len + [SEP]
            segment_ids.append(0)   # [0] * (1 + query_len + 1)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")  # [CLS] + [token] * query_len + [SEP] + [token] * doc_len + [SEP]
            segment_ids.append(1)   # [0] * (1 + query_len + 1) + [1] * (doc_len + 1)
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens) # 101-[CLS], 102-[SEP]
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:  # zero padding
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True  # check whether answer is out of current span
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_na = 1
                else:
                    doc_offset = len(query_tokens) + 2  # doc_token_start
                    start_position = tok_start_position - doc_start + doc_offset    # position in all tokens(query+doc)
                    end_position = tok_end_position - doc_start + doc_offset
                    span_na = 0
            if is_training and example.is_impossible:
                start_position = 0  # [CLS]
                end_position = 0    # [CLS]
                span_na = 1
            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                    span_na=span_na))
            unique_id += 1

    return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "span_na_logits"])
ClsResult = collections.namedtuple("ClsResult",
                                   ["unique_id", "example_index", "doc_span_index", "span_na_logits"])
TrainClsResult = collections.namedtuple("TrainClsResult",
                                   ["unique_id", "example_index", "doc_span_index", "span_na_logits", "span_na"])

def select_topk_for_train(train_examples, train_features, all_results, top_k):
    # qas_id_topk_dict = 
    # like this: {37: {2:0.003, 7:0.324, 0:-0.344, ... 4:0.001}, 21: {5:???, ..., 1:???}, ...}
    qas_id_topk_dict = OrderedDict()   # 存储上述数据结构(全部的logits)
    # 遍历所有result(unique_id, example_index, doc_span_index, span_na_logits)
    for result in all_results:
        example_index = result.example_index
        doc_span_index = result.doc_span_index
        span_na_logits = result.span_na_logits
        span_na = result.span_na
        if example_index not in qas_id_topk_dict:
            qas_id_topk_dict[example_index] = OrderedDict()
        # 该字典的value中，前一个为可回答的概率，后一个为是否不可回答
        qas_id_topk_dict[example_index][doc_span_index] = (span_na_logits[0], span_na)
    # 选择top k(筛选qas_id_topk_dict到只剩k个最大的，按原文顺序排列)
    for example_index in qas_id_topk_dict:
        index_to_logits_dict = qas_id_topk_dict[example_index]  # 每个example中的doc_span_index -> (cls_logits, span_na)
        sorted_dict = select_k_for_train(index_to_logits_dict, top_k) # 选取ans和前k-1或k个logits最大的并按原文顺序排序
        qas_id_topk_dict[example_index] = sorted_dict
    return qas_id_topk_dict


def select_k_for_train(index_to_logits_dict, top_k):
    """
    对于有答案文本：选择一个包含答案的段落P，加上k-1个最容易回答的段落，按原文顺序排列
    对于没有答案文本：选择k个最容易回答的段落，按原文顺序排列
    """
    dict_to_list = list(index_to_logits_dict.items())   # 将有序字典转换为列表
    sorted_list = sorted(dict_to_list, key=lambda x:x[1][0], reverse=True) # 按cls_logtis从大到小排序
    sorted_list = sorted(sorted_list, key=lambda x:x[1][1])   # 把span_na=0放在最前面
    select_k_big = sorted_list[:top_k]  # 选取前k个logits最高的doc_span_index->cls_logits
    psg_sorted = sorted(select_k_big, key=lambda x:x[0])    # 按照原文顺序排序
    sorted_dict = OrderedDict(psg_sorted)   # 转换为有序字典
    return sorted_dict


def select_topk(eval_examples, eval_features, all_results, top_k):
    # qas_id_topk_dict = 
    # (ordered_dict){example_index: (ordered_dict){doc_span_index(0), ..., doc_span_index(k)}, ...}
    # like this: {0: {0:0.003, 1:0.324, 2:-0.344, ... 7:0.001}, 1: {0:???, ..., 9:???}, ...}
    qas_id_topk_dict = OrderedDict()   # 存储上述数据结构(全部的logits)
    # 遍历所有result(unique_id, example_index, doc_span_index, span_na_logits)
    for result in all_results:
        example_index = result.example_index
        doc_span_index = result.doc_span_index
        span_na_logits = result.span_na_logits
        if example_index not in qas_id_topk_dict:
            qas_id_topk_dict[example_index] = OrderedDict()
        qas_id_topk_dict[example_index][doc_span_index] = span_na_logits
    # 选择top k(筛选qas_id_topk_dict到只剩k个最大的，按原文顺序排列)
    qas_id_top10_dict = copy.deepcopy(qas_id_topk_dict)
    for example_index in qas_id_topk_dict:
        index_to_logits_dict = qas_id_topk_dict[example_index]  # 每个example中的doc_span_index -> cls_logits
        sorted_dict, sorted_dict_10 = select_k(index_to_logits_dict, top_k) # 选取前k个logits最大的并按原文顺序排序
        qas_id_topk_dict[example_index] = sorted_dict
        qas_id_top10_dict[example_index] = sorted_dict_10
    return qas_id_topk_dict, qas_id_top10_dict


def select_k(index_to_logits_dict, top_k):
    dict_to_list = list(index_to_logits_dict.items())   # 将有序字典转换为列表
    sorted_list = sorted(dict_to_list, key=lambda x:x[1][0], reverse=True) # 按cls_logtis从大到小排序
    select_k_big = sorted_list[:top_k]  # 选取前k个logits最高的doc_span_index->cls_logits
    select_10_big = sorted_list[:10]    # 直接按从大到小的顺序，计算hit@k
    psg_sorted = sorted(select_k_big, key=lambda x:x[0])    # 按照原文顺序排序
    sorted_dict = OrderedDict(psg_sorted)   # 转换为有序字典
    sorted_dict_10 = OrderedDict(select_10_big)
    return sorted_dict, sorted_dict_10

def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, verbose_logging,
                     version_2_with_negative):
    # example索引到该example下多个feature的映射
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    # 每个feature到RawResult的映射{unique_id, start_logtis, end_logits, span_na_logits}
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index] # 获取每个example下的features
        prelim_predictions = [] # 存放每个可能答案
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)
        # 这里的prelim_predictions已经将example中所有可能答案排好序

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"])
        seen_predictions = {}
        nbest = []  # 丢入最好的20个答案(text, start_logit, end_logit)
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break   # 存满nbest_size个就结束
            feature = features[pred.feature_index]  # 获取当前feature
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]  # 答案token
                orig_doc_start = feature.token_to_orig_map[pred.start_index]    # 开始word
                orig_doc_end = feature.token_to_orig_map[pred.end_index]    # 结束word
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)] # 答案word
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                # tok_text = tok_text.replace("_", "")    # Albert_token
                # tok_text = tok_text.replace(" _", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True # 加入最终text（改动：text所在的sentence）
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        assert len(nbest) >= 1

        total_scores = []   # 放入start_logit + end_logit的20个总分
        best_non_null_entry = None  # 赋予最佳的可回答的predict
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry # 如果text=="", 左值的布尔值为False

        probs = _compute_softmax(total_scores)  # 将分数归一化为概率
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)  # 度量不可回答的可能性，值越大表示越不可回答
            scores_diff_json[example.qas_id] = score_diff
            all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, scores_diff_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer(s):

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers'] if normalize_answer(a['text'])]
                if not gold_answers:
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def evaluate_1(args, model, device, train_dataloader, train_examples, train_features):
    """为训练数据筛选k个段落，其中answer所在段落必选（如果可回答）"""
    all_results = []
    model.eval()

    for idx, (input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices, span_nas) in enumerate(train_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            # 当做预测(dev set)用
            return_dict = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            batch_start_logits, batch_end_logits, batch_span_na_logits = return_dict["start_logits"], return_dict["end_logits"], return_dict["span_na_logits"]
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            
            span_na_logits = batch_span_na_logits[i].detach().cpu()
            span_na_logits = F.softmax(span_na_logits, dim=-1)
            train_feature = train_features[example_index.item()]
            unique_id = int(train_feature.unique_id)
            # 用example_index + doc_span_index或unique_id标识一段话
            all_results.append(TrainClsResult(unique_id=unique_id,
                                         example_index=train_feature.example_index,
                                         doc_span_index=train_feature.doc_span_index,
                                         span_na_logits=span_na_logits,
                                         span_na=train_feature.span_na))
    
    qas_id_topk_dict = select_topk_for_train(train_examples, train_features, all_results, args.top_k)
    return qas_id_topk_dict

def evaluate_0(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
    # 目标：为每个example选择top k个段落
    all_results = []
    model.eval()
    # eval中span_na没用
    for idx, (input_ids, input_mask, segment_ids, example_indices, _) in enumerate(eval_dataloader):
        # eval_bsz条数据
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        # span_na = span_na.to(device)
        with torch.no_grad():
            return_dict = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            batch_start_logits, batch_end_logits, batch_span_na_logits = return_dict["start_logits"], return_dict["end_logits"], return_dict["span_na_logits"]
        for i, example_index in enumerate(example_indices):
            # 对每条数据
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            span_na_logits = batch_span_na_logits[i].detach().cpu().tolist()
            # 修改
            span_na_logits = batch_span_na_logits[i].detach().cpu()
            span_na_logits = F.softmax(span_na_logits, dim=-1)
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            # 用example_index + doc_span_index或unique_id标识一段话
            all_results.append(ClsResult(unique_id=unique_id,
                                         example_index=eval_feature.example_index,
                                         doc_span_index=eval_feature.doc_span_index,
                                         span_na_logits=span_na_logits))

    # preds(ordered_dict): qas_id -> best_pred
    # nbest_preds(ordered_dict): qas_id -> [(ordered_dict)pred, ... *20]
    # na_probs(ordered_dict): qas_id -> no answer prob
    # 目标：在每个example中找出top_k个paragraph，返回qas_id
    qas_id_topk_dict, qas_id_top10_dict = select_topk(eval_examples, eval_features, all_results, args.top_k)
    # 新增：并且返回不按原文排序的k个para
    return qas_id_topk_dict, qas_id_top10_dict
    preds, nbest_preds, na_probs = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging,
                         args.version_2_with_negative)
    
    if pred_only:
        if args.version_2_with_negative:
            for k in preds:
                if na_probs[k] > na_prob_thresh:
                    preds[k] = ''
        return {}, preds, nbest_preds

    if args.version_2_with_negative:
        qid_to_has_ans = make_qid_to_has_ans(eval_dataset)  # 统计每个question是否有答案
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]  # has ans
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]   # no ans
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        result = make_eval_dict(exact_thresh, f1_thresh)
        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
            merge_eval(result, has_ans_eval, 'HasAns')
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
            merge_eval(result, no_ans_eval, 'NoAns')
        find_all_best_thresh(result, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
        for k in preds:
            if na_probs[k] > result['best_f1_thresh']:
                preds[k] = ''
    else:
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        result = make_eval_dict(exact_raw, f1_raw)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    # logger.info("  hits_rate = %f", hits_num / dev_num)
    return result, preds, nbest_preds


def evaluate(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
    all_results = []
    # hits_num = 0
    # dev_num = 0
    model.eval()
    # remove span_na
    for idx, (input_ids, input_mask, segment_ids, example_indices) in enumerate(eval_dataloader):
        # eval_bsz条数据
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        # span_na = span_na.to(device)
        with torch.no_grad():
            return_dict = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            batch_start_logits, batch_end_logits, batch_span_na_logits = return_dict["start_logits"], return_dict["end_logits"], return_dict["span_na_logits"]
        for i, example_index in enumerate(example_indices):
            # 对每条数据
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            span_na_logits = batch_span_na_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         span_na_logits=span_na_logits))
            # if np.argmax(span_na_logits) == span_na[i]:
            #     hits_num += 1
            # dev_num += 1

    # preds(ordered_dict): qas_id -> best_pred
    # nbest_preds(ordered_dict): qas_id -> [(ordered_dict)pred, ... *20]
    # na_probs(ordered_dict): qas_id -> no answer prob
    preds, nbest_preds, na_probs = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging,
                         args.version_2_with_negative)
    if pred_only:
        if args.version_2_with_negative:
            for k in preds:
                if na_probs[k] > na_prob_thresh:
                    preds[k] = ''
        return {}, preds, nbest_preds

    if args.version_2_with_negative:
        qid_to_has_ans = make_qid_to_has_ans(eval_dataset)  # 统计每个question是否有答案
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]  # has ans
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]   # no ans
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        result = make_eval_dict(exact_thresh, f1_thresh)
        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
            merge_eval(result, has_ans_eval, 'HasAns')
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
            merge_eval(result, no_ans_eval, 'NoAns')
        find_all_best_thresh(result, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
        for k in preds:
            if na_probs[k] > result['best_f1_thresh']:
                preds[k] = ''
    else:
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        result = make_eval_dict(exact_raw, f1_raw)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    # logger.info("  hits_rate = %f", hits_num / dev_num)
    return result, preds, nbest_preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir_2):
        os.makedirs(args.output_dir_2)
    os.makedirs(args.data_binary_dir, exist_ok=True)
    os.makedirs(args.data_binary_dir_2, exist_ok=True)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir_2, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir_2, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    if args.do_train or (not args.eval_test):
        with open(args.dev_file) as f:
            dataset_json = json.load(f)
        eval_dataset = dataset_json['data']
        if args.do_preprocess:
            eval_examples = read_squad_examples(
                input_file=args.dev_file, is_training=False,
                version_2_with_negative=args.version_2_with_negative)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            logger.info("Saving eval features into file %s newsqa_eval_features.pkl", args.data_binary_dir)
            torch.save(eval_features, os.path.join(args.data_binary_dir, "newsqa_eval_%s.pkl" % args.max_seq_length))
        else:
            eval_examples = read_squad_examples(
                input_file=args.dev_file, is_training=False,
                version_2_with_negative=args.version_2_with_negative)
            eval_features = torch.load(os.path.join(args.data_binary_dir, "newsqa_eval_%s.pkl" % args.max_seq_length))
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_span_na = torch.tensor([f.span_na for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_span_na)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        # model1
        if args.do_preprocess:
            train_examples = read_squad_examples(
                input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
            train_features = convert_examples_to_features(
                    examples=train_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True)
            logger.info("Saving train features into file %s newsqa_train_features.pkl", args.data_binary_dir)
            torch.save(train_features, os.path.join(args.data_binary_dir, "newsqa_train_%s.pkl" % args.max_seq_length))
        else:
            train_examples = read_squad_examples(
                input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
            train_features = torch.load(os.path.join(args.data_binary_dir, "newsqa_train_%s.pkl" % args.max_seq_length))
        # if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
        #     # train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        #     pass
        # else:
        #     random.shuffle(train_features)
        # 先不打乱，等编好example_index后再打乱顺序
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_span_na = torch.tensor([f.span_na for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_example_index, all_span_na)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.first_stage:
            eval_step = max(1, len(train_batches) // args.eval_per_epoch)
            best_result = None
            lrs = [args.learning_rate] if args.learning_rate else \
                [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
            for lr in lrs:
                model = BertForQuestionAnswering2.from_pretrained(args.model)
                # model2 = BertForQuestionAnswering.from_pretrained(args.model)
                if args.fp16:
                    model.half()
                    # model2.half()
                model.to(device)
                # model2.to(device)
                if n_gpu > 1:
                    model = torch.nn.DataParallel(model)
                    # model2 = torch.nn.DataParallel(model2)
                param_optimizer = list(model.named_parameters())
                # param_optimizer2 = list(model2.named_parameters())
                param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
                # param_optimizer2 = [n for n in param_optimizer2 if 'pooler' not in n[0]]
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer
                                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer
                                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                # optimizer_grouped_parameters2 = [
                #     {'params': [p for n, p in param_optimizer2
                #                 if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                #     {'params': [p for n, p in param_optimizer2
                #                 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                # ]

                if args.fp16:
                    try:
                        from apex.optimizers import FP16_Optimizer
                        from apex.optimizers import FusedAdam
                    except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                        "to use distributed and fp16 training.")
                    optimizer = FusedAdam(optimizer_grouped_parameters,
                                        lr=lr,
                                        bias_correction=False,
                                        max_grad_norm=1.0)
                    # optimizer2 = FusedAdam(optimizer_grouped_parameters2,
                    #                       lr=lr,
                    #                       bias_correction=False,
                    #                       max_grad_norm=1.0)
                    if args.loss_scale == 0:
                        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                        # optimizer2 = FP16_Optimizer(optimizer2, dynamic_loss_scale=True)
                    else:
                        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
                        # optimizer2 = FP16_Optimizer(optimizer2, static_loss_scale=args.loss_scale)
                else:
                    optimizer = BertAdam(optimizer_grouped_parameters,
                                        lr=lr,
                                        warmup=args.warmup_proportion,
                                        t_total=num_train_optimization_steps)
                    # optimizer2 = BertAdam(optimizer_grouped_parameters2,
                    #                      lr=lr,
                    #                      warmup=args.warmup_proportion,
                    #                      t_total=num_train_optimization_steps)
                tr_loss = 0
                nb_tr_examples = 0
                nb_tr_steps = 0
                global_step = 0
                start_time = time.time()
                for epoch in range(int(args.num_train_epochs)):
                    model.train()
                    # model2.train()
                    logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                    if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                        random.shuffle(train_batches)
                    for step, batch in enumerate(train_batches):
                        if n_gpu == 1:
                            batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, segment_ids, start_positions, end_positions, example_index, span_na = batch
                        return_dict = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, start_positions=start_positions, end_positions=end_positions, span_na=span_na)
                        loss = return_dict["loss"]
                        # return_dict2 = model2(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, start_positions=start_positions, end_positions=end_positions, span_na=span_na)
                        if n_gpu > 1:
                            loss = loss.mean()
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1

                        if args.fp16:
                            optimizer.backward(loss)
                        else:
                            loss.backward()
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if args.fp16:
                                lr_this_step = lr * \
                                    warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = lr_this_step
                            optimizer.step()
                            optimizer.zero_grad()
                            global_step += 1

                        if (step + 1) % eval_step == 0:
                            logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

                            save_model = False
                            if args.do_eval:
                                # result, _, _ = \
                                #     evaluate(args, model, device, eval_dataset,
                                #              eval_dataloader, eval_examples, eval_features)
                                # 得到字典：选择k句最好的话
                                qas_id_topk_dict, qas_id_top10_dict = evaluate_0(args, model, device, eval_dataset,\
                                    eval_dataloader, eval_examples, eval_features)
                                # eval_dataset = ???
                                # eval_dataloader = ???
                                # eval_examples = ???
                                # eval_features = ???
                                eval_dataset_new = eval_dataset
                                eval_examples_new, hits_rate = reconstruct_eval_examples(eval_examples, qas_id_topk_dict, qas_id_top10_dict)
                                # 换行符为空，相当于直接截断
                                eval_features_new = convert_examples_to_features(
                                    examples=eval_examples_new,
                                    tokenizer=tokenizer,
                                    max_seq_length=args.max_seq_length,
                                    doc_stride=args.doc_stride,
                                    max_query_length=args.max_query_length,
                                    is_training=False)
                                
                                all_input_ids_new = torch.tensor([f.input_ids for f in eval_features_new], dtype=torch.long)
                                all_input_mask_new = torch.tensor([f.input_mask for f in eval_features_new], dtype=torch.long)
                                all_segment_ids_new = torch.tensor([f.segment_ids for f in eval_features_new], dtype=torch.long)
                                all_example_index_new = torch.arange(all_input_ids_new.size(0), dtype=torch.long)
                                eval_data_new = TensorDataset(all_input_ids_new, all_input_mask_new, all_segment_ids_new, all_example_index_new)
                                eval_sampler_new = SequentialSampler(eval_data_new)
                                eval_dataloader_new = DataLoader(eval_data_new, sampler=eval_sampler_new, batch_size=args.eval_batch_size)
                                # 保留hit@k，移除预测答案
                                result, _, _ = \
                                    evaluate(args, model, device, eval_dataset_new,
                                            eval_dataloader_new, eval_examples_new, eval_features_new)
                                for k in range(1, 11):
                                    logger.info("hits at %d = %f", k, hits_rate[k-1])
                                model.train()
                                # model2.train()
                                result['global_step'] = global_step
                                result['epoch'] = epoch
                                result['learning_rate'] = lr
                                result['batch_size'] = args.train_batch_size
                                if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                    best_result = result
                                    save_model = True
                                    logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                                (args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                            else:
                                save_model = True
                            if save_model:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                # model_to_save = model2.module if hasattr(model2, 'module') else model2
                                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                tokenizer.save_vocabulary(args.output_dir)
                                if best_result:
                                    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
                                        for key in sorted(best_result.keys()):
                                            writer.write("%s = %s\n" % (key, str(best_result[key])))
        else:
            model = BertForQuestionAnswering2.from_pretrained(args.output_dir)
            model.to(device)
        # model2
        # 重构训练集
        
        qas_id_topk_dict = evaluate_1(args, model, device, train_dataloader, train_examples, train_features)

        if args.do_preprocess_2:
            
            train_examples_2 = reconstruct_train_examples(train_examples, qas_id_topk_dict)
            # 直接截断（弃用滑动窗口）
            
            train_features_2 = convert_examples_to_features(
                    examples=train_examples_2,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length_2,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True)
            logger.info("saving stage 2...")
            torch.save(train_examples_2, os.path.join(args.data_binary_dir_2, "newsqa_train_examples_%s.pkl" % args.max_seq_length_2))
            torch.save(train_features_2, os.path.join(args.data_binary_dir_2, "newsqa_train_features_%s.pkl" % args.max_seq_length_2))
        else:
            train_examples_2 = torch.load(os.path.join(args.data_binary_dir_2, "newsqa_train_examples_%s.pkl" % args.max_seq_length_2))
            train_features_2 = torch.load(os.path.join(args.data_binary_dir_2, "newsqa_train_features_%s.pkl" % args.max_seq_length_2))
        
        # 构建数据
        all_input_ids_2 = torch.tensor([f.input_ids for f in train_features_2], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask for f in train_features_2], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.segment_ids for f in train_features_2], dtype=torch.long)
        all_start_positions_2 = torch.tensor([f.start_position for f in train_features_2], dtype=torch.long)
        all_end_positions_2 = torch.tensor([f.end_position for f in train_features_2], dtype=torch.long)
        all_span_na_2 = torch.tensor([f.span_na for f in train_features_2], dtype=torch.long)
        train_data_2 = TensorDataset(all_input_ids_2, all_input_mask_2, all_segment_ids_2,
                                   all_start_positions_2, all_end_positions_2, all_span_na_2)
        train_sampler_2 = RandomSampler(train_data_2)
        train_dataloader_2 = DataLoader(train_data_2, sampler=train_sampler_2, batch_size=args.train_batch_size)
        train_batches_2 = [batch for batch in train_dataloader_2]

        num_train_optimization_steps_2 = \
            len(train_dataloader_2) // args.gradient_accumulation_steps * args.num_train_epochs_2
        
        logger.info("***** Train Stage 2 *****")
        logger.info("  Num orig examples = %d", len(train_examples_2))
        logger.info("  Num split examples = %d", len(train_features_2))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps_2)

        eval_step_2 = max(1, len(train_batches_2) // args.eval_per_epoch)
        best_result_2 = None

        lr_2 = args.learning_rate
        model2 = BertForQuestionAnswering3.from_pretrained(args.model)
        if args.fp16:
            model2.half()
        model2.to(device)
        if n_gpu > 1:
            model2 = torch.nn.DataParallel(model2)
        param_optimizer_2 = list(model2.named_parameters())
        param_optimizer_2 = [n for n in param_optimizer_2 if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters_2 = [
            {'params': [p for n, p in param_optimizer_2
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer_2
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # fp16 先不做
        optimizer_2 = BertAdam(optimizer_grouped_parameters_2,
                            lr=lr_2,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps_2)

        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        global_step = 0
        start_time = time.time()

        for epoch in range(int(args.num_train_epochs_2)):
            model2.train()
            logger.info("Start epoch #{} (lr = {}) in stage 2...".format(epoch, lr_2))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches_2)
            for step, batch in enumerate(train_batches_2):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions, span_na = batch
                return_dict = model2(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, start_positions=start_positions, end_positions=end_positions, span_na=span_na)
                loss = return_dict["loss"]
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # 去掉fp16部分
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        lr_this_step = lr_2 * \
                            warmup_linear(global_step/num_train_optimization_steps_2, args.warmup_proportion)
                        for param_group in optimizer_2.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer_2.step()
                    optimizer_2.zero_grad()
                    global_step += 1

                if (step + 1) % eval_step_2 == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                        epoch, step + 1, len(train_batches_2), time.time() - start_time, tr_loss / nb_tr_steps))

                    save_model = False
                    if args.do_eval:
                        # result, _, _ = \
                        #     evaluate(args, model, device, eval_dataset,
                        #              eval_dataloader, eval_examples, eval_features)
                        # 得到字典：选择k句最好的话
                        qas_id_topk_dict, qas_id_top10_dict = evaluate_0(args, model, device, eval_dataset,\
                            eval_dataloader, eval_examples, eval_features)
                        # eval_dataset = ???
                        # eval_dataloader = ???
                        # eval_examples = ???
                        # eval_features = ???
                        eval_dataset_new = eval_dataset
                        eval_examples_new, hits_rate = reconstruct_eval_examples(eval_examples, qas_id_topk_dict, qas_id_top10_dict)
                        eval_features_new = convert_examples_to_features(
                            examples=eval_examples_new,
                            tokenizer=tokenizer,
                            max_seq_length=args.max_seq_length,
                            doc_stride=args.doc_stride,
                            max_query_length=args.max_query_length,
                            is_training=False)
                        
                        all_input_ids_new = torch.tensor([f.input_ids for f in eval_features_new], dtype=torch.long)
                        all_input_mask_new = torch.tensor([f.input_mask for f in eval_features_new], dtype=torch.long)
                        all_segment_ids_new = torch.tensor([f.segment_ids for f in eval_features_new], dtype=torch.long)
                        all_example_index_new = torch.arange(all_input_ids_new.size(0), dtype=torch.long)
                        eval_data_new = TensorDataset(all_input_ids_new, all_input_mask_new, all_segment_ids_new, all_example_index_new)
                        eval_sampler_new = SequentialSampler(eval_data_new)
                        eval_dataloader_new = DataLoader(eval_data_new, sampler=eval_sampler_new, batch_size=args.eval_batch_size)
                        result, _, _ = \
                            evaluate(args, model2, device, eval_dataset_new,
                                        eval_dataloader_new, eval_examples_new, eval_features_new)
                        for k in range(1, 11):
                            logger.info("hits at %d = %f", k, hits_rate[k-1])
                        model2.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr_2
                        result['batch_size'] = args.train_batch_size
                        if (best_result_2 is None) or (result[args.eval_metric] > best_result_2[args.eval_metric]):
                            best_result_2 = result
                            save_model = True
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr_2), epoch, result[args.eval_metric]))
                    else:
                        save_model = True
                    if save_model:
                        model_to_save = model2.module if hasattr(model2, 'module') else model2
                        output_model_file = os.path.join(args.output_dir_2, WEIGHTS_NAME)
                        output_config_file = os.path.join(args.output_dir_2, CONFIG_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir_2)
                        if best_result_2:
                            with open(os.path.join(args.output_dir_2, "eval_results.txt"), "w") as writer:
                                for key in sorted(best_result_2.keys()):
                                    writer.write("%s = %s\n" % (key, str(best_result_2[key])))

    # 需要评估第二个模型出来后的结果
    if args.do_eval:
        if args.eval_test:
            with open(args.test_file) as f:
                dataset_json = json.load(f)
            eval_dataset = dataset_json['data']
            eval_examples = read_squad_examples(
                input_file=args.test_file, is_training=False,
                version_2_with_negative=args.version_2_with_negative)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            all_span_na = torch.tensor([f.span_na for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_span_na)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
        model = BertForQuestionAnswering2.from_pretrained(args.output_dir)
        model2 = BertForQuestionAnswering3.from_pretrained(args.output_dir_2)
        if args.fp16:
            model.half()
            model2.half()
        model.to(device)
        model2.to(device)

        # na_prob_thresh = 1.0
        # if args.version_2_with_negative:
        #     eval_result_file = os.path.join(args.output_dir, "eval_results.txt")
        #     if os.path.isfile(eval_result_file):
        #         with open(eval_result_file) as f:
        #             for line in f.readlines():
        #                 if line.startswith('best_f1_thresh'):
        #                     na_prob_thresh = float(line.strip().split()[-1])
        #                     logger.info("na_prob_thresh = %.6f" % na_prob_thresh)

        # repeat again
        na_prob_thresh = 1.0
        if args.version_2_with_negative:
            eval_result_file_2 = os.path.join(args.output_dir_2, "eval_results.txt")
            if os.path.isfile(eval_result_file_2):
                with open(eval_result_file_2) as f:
                    for line in f.readlines():
                        if line.startswith('best_f1_thresh'):
                            na_prob_thresh = float(line.strip().split()[-1])
                            logger.info("na_prob_thresh = %.6f" % na_prob_thresh)

        qas_id_topk_dict, qas_id_top10_dict = evaluate_0(args, model, device, eval_dataset,\
                                eval_dataloader, eval_examples, eval_features)
        eval_dataset_new = eval_dataset
        eval_examples_new, hits_rate = reconstruct_eval_examples(eval_examples, qas_id_topk_dict, qas_id_top10_dict)
        eval_features_new = convert_examples_to_features(
            examples=eval_examples_new,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length_2,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        all_input_ids_new = torch.tensor([f.input_ids for f in eval_features_new], dtype=torch.long)
        all_input_mask_new = torch.tensor([f.input_mask for f in eval_features_new], dtype=torch.long)
        all_segment_ids_new = torch.tensor([f.segment_ids for f in eval_features_new], dtype=torch.long)
        all_example_index_new = torch.arange(all_input_ids_new.size(0), dtype=torch.long)
        eval_data_new = TensorDataset(all_input_ids_new, all_input_mask_new, all_segment_ids_new, all_example_index_new)
        eval_sampler_new = SequentialSampler(eval_data_new)
        eval_dataloader_new = DataLoader(eval_data_new, sampler=eval_sampler_new, batch_size=args.eval_batch_size)
        
        result, preds, _ = \
            evaluate(args, model2, device, eval_dataset_new,
                     eval_dataloader_new, eval_examples_new, eval_features_new,
                     na_prob_thresh=na_prob_thresh,
                     pred_only=args.eval_test)
        with open(os.path.join(args.output_dir_2, "predictions.json"), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model checkpoints and predictions will be written.")
        parser.add_argument("--train_file", default=None, type=str,
                            help="SQuAD json for training. E.g., train-v1.1.json")
        parser.add_argument("--dev_file", default=None, type=str,
                            help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
        parser.add_argument("--test_file", default=None, type=str)
        parser.add_argument("--eval_per_epoch", default=5, type=int,
                            help="How many times it evaluates on dev set per epoch")
        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, "
                                 "how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")
        parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_test", action='store_true', help='Wehther to run eval on the test set.')
        parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
        parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
        parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--eval_metric", default='f1', type=str)
        parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                                 "of training.")
        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json "
                                 "output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. "
                                 "This is needed because the start "
                                 "and end predictions are not conditioned on one another.")
        parser.add_argument("--verbose_logging", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal SQuAD evaluation.")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale', type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        parser.add_argument('--version_2_with_negative', action='store_true',
                            help='If true, the SQuAD examples contain some that do not have an answer.')
        parser.add_argument("--top_k", type=int, default=1, help="first read to select top k paragraphs")
        parser.add_argument("--max_seq_length_2", type=int, default=512, help="seqlen1=256, seqlen2=512")
        parser.add_argument("--output_dir_2", default=None, type=str, required=True)
        parser.add_argument("--num_train_epochs_2", default=4.0, type=float)
        parser.add_argument("--do_preprocess", default=False, action="store_true", help="make data binary format")
        parser.add_argument("--data_binary_dir", default="", type=str)
        parser.add_argument("--first_stage", default=False, action="store_true")
        parser.add_argument("--do_preprocess_2", default=False, action="store_true")
        parser.add_argument("--data_binary_dir_2", default="", type=str)
        args = parser.parse_args()

        main(args)
