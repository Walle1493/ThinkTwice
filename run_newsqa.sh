# # hierarchical_demo
# python hierarchical/run_bert.py \
#   --do_train \
#   --do_eval \
#   --eval_test \
#   --model bert-base-uncased \
#   --train_file ~/Data/newsqa/newsqa-squad2-dataset/scaled-newsqa-train.json \
#   --dev_file ~/Data/newsqa/newsqa-squad2-dataset/scaled-newsqa-dev.json \
#   --test_file ~/Data/newsqa/newsqa-squad2-dataset/scaled-newsqa-test.json \
#   --train_batch_size 24 \
#   --eval_batch_size 32  \
#   --learning_rate 2e-5 \
#   --num_train_epochs 1 \
#   --num_train_epochs_2 1 \
#   --max_seq_length 256 \
#   --max_seq_length_2 512 \
#   --doc_stride 128 \
#   --eval_metric best_f1 \
#   --output_dir outputs/newsqa/try_retr \
#   --output_dir_2 outputs/newsqa/try_read \
#   --data_binary_dir data_binary/try_retr \
#   --data_binary_dir_2 data_binary/try_read \
#   --version_2_with_negative \
#   --do_lower_case \
#   --top_k 5 \
#   # --do_preprocess_2 \
#   # --first_stage \
#   # --do_preprocess \


# hierarchical
python hierarchical/run_bert.py \
  --do_train \
  --do_eval \
  --eval_test \
  --model bert-base-uncased \
  --train_file ~/Data/newsqa/newsqa-squad2-dataset/squad-newsqa-train.json \
  --dev_file ~/Data/newsqa/newsqa-squad2-dataset/squad-newsqa-dev.json \
  --test_file ~/Data/newsqa/newsqa-squad2-dataset/squad-newsqa-test.json \
  --train_batch_size 24 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --num_train_epochs_2 3 \
  --max_seq_length 256 \
  --max_seq_length_2 512 \
  --doc_stride 128 \
  --eval_metric best_f1 \
  --output_dir outputs/newsqa/bert_retr \
  --output_dir_2 outputs/newsqa/bert_read \
  --data_binary_dir data_binary/newsqa_retr \
  --data_binary_dir_2 data_binary/newsqa_read \
  --version_2_with_negative \
  --do_lower_case \
  --top_k 5 \
  --do_preprocess_2 \
  --first_stage \
  # --do_preprocess \