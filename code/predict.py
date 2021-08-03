import collections
from utils import _get_best_indexes, _compute_softmax, make_qid_to_has_ans, get_raw_scores, \
    apply_no_ans_threshold, make_eval_dict, merge_eval, find_all_best_thresh
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from reconstruct import select_topk_for_train, select_topk
import logging
import torch
import torch.nn.functional as F


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "span_na_logits"])
ClsResult = collections.namedtuple("ClsResult",
                                   ["unique_id", "example_index", "doc_span_index", "span_na_logits"])
TrainClsResult = collections.namedtuple("TrainClsResult",
                                   ["unique_id", "example_index", "doc_span_index", "span_na_logits", "span_na"])


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


def filter_for_train(args, model, device, train_dataloader, train_examples, train_features):
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

def filter_for_eval(args, model, device, eval_dataset, eval_dataloader,
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
