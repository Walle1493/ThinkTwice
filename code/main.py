import torch
import logging
import random
import numpy as np
import os
from transformers import BertTokenizer
import json
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from bert_model import BertForQuestionAnsweringCLS, BertForQuestionAnsweringQA
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
import time
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
import argparse
from predict import filter_for_eval, filter_for_train, evaluate
from preprocess import read_squad_examples, convert_examples_to_features
from reconstruct import reconstruct_train_examples, reconstruct_eval_examples


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
                model = BertForQuestionAnsweringCLS.from_pretrained(args.model)
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
                                qas_id_topk_dict, qas_id_top10_dict = filter_for_eval(args, model, device, eval_dataset,\
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
            model = BertForQuestionAnsweringCLS.from_pretrained(args.output_dir)
            model.to(device)
        # model2
        # 重构训练集
        
        qas_id_topk_dict = filter_for_train(args, model, device, train_dataloader, train_examples, train_features)

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
        model2 = BertForQuestionAnsweringQA.from_pretrained(args.model)
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
                        qas_id_topk_dict, qas_id_top10_dict = filter_for_eval(args, model, device, eval_dataset,\
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
            
        model = BertForQuestionAnsweringCLS.from_pretrained(args.output_dir)
        model2 = BertForQuestionAnsweringQA.from_pretrained(args.output_dir_2)
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

        qas_id_topk_dict, qas_id_top10_dict = filter_for_eval(args, model, device, eval_dataset,\
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
