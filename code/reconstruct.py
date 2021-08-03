from collections import OrderedDict
import copy


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
