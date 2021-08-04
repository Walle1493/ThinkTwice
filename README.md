# ThinkTwice

ThinkTwice is a retriever-reader architecture for solving long-text machine reading comprehension. It is based on the paper: ***ThinkTwice: A Two-Stage Method for Long-Text Machine Reading Comprehension***. Authors are [Mengxing Dong](https://github.com/Walle1493), Bowei Zou, [Jin Qian](https://github.com/jaytsien), [Rongtao Huang](https://github.com/WhaleFallzz) and Yu Hong from Soochow University and Institute for Infocomm Research. The paper will be published in NLPCC 2021 soon.

## Contents

- [Background](https://github.com/Walle1493/ThinkTwice#background)
- [Requirements](https://github.com/Walle1493/ThinkTwice#install)
- [Dataset](https://github.com/Walle1493/ThinkTwice#Dataset)
- [Train](https://github.com/Walle1493/ThinkTwice#Train)
- [License](https://github.com/Walle1493/ThinkTwice#License)

## Background

Our idea is mainly inspired by the way humans think: We first read a lengthy document and remain several slices which are important to our task in our mind; then we are gonna capture the final answer within this limited information.

The goals for this repository are:

1. A **complete code** for NewsQA. This repo offers an implement for dealing with long text MRC dataset NewsQA; you can also try this method on other datsets like TriviaQA, Natural Questions yourself.
2. A comparison **description**. The performance on ThinkTwice has been listed in the paper.
3. A public space for **advice**. You are welcomed to propose an issue in this repo.

## Requirements

Clone this repo at your local server. Install necessary libraries listed below.

```bash
git clone git@github.com:Walle1493/ThinkTwice.git
pip install requirements.txt
```

You may install several libraries on yourself.

## Dataset

You need to prepare data in a squad2-like format. Since [NewsQA](https://github.com/Maluuba/newsqa) ([click here seeing more](https://github.com/Maluuba/newsqa)) is similar to SQuAD-2.0, we don't offer the script in this repo. The demo data format is showed below:

```json
"version": "1",
"data": [
    {
        "type": "train",
        "title": "./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story",
        "paragraphs": [
            {
                "context": "NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted a wealthy...",
                "qas": [
                    {
                        "question": "What was the amount of children murdered?",
                        "id": "./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story01",
                        "answers": [
                            {
                                "answer_start": 294,
                                "text": "19"
                            }
                        ],
                        "is_impossible": false
                    },
                    {
                        "question": "When was Pandher sentenced to death?",
                        "id": "./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story02",
                        "answers": [
                            {
                                "answer_start": 261,
                                "text": "February"
                            }
                        ],
                        "is_impossible": false
                    }
                ]
            }
        ]
    }
]
```

P.S.: You are supposed to make a change when dealing with other datasets like [TriviaQA](https://github.com/mandarjoshi90/triviaqa) or [Natural Questions](https://github.com/google-research-datasets/natural-questions), because we split passages by '\n' character in NewsQA, while not all the same in other datasets.

## Train

The training step (including test module) depends mainly on these parameters. We trained our two-stage model on 4 GPUs with 12G 1080Ti in 60 hours.

```bash
python code/main.py \
  --do_train \
  --do_eval \
  --eval_test \
  --model bert-base-uncased \
  --train_file ~/Data/newsqa/newsqa-squad2-dataset/squad-newsqa-train.json \
  --dev_file ~/Data/newsqa/newsqa-squad2-dataset/squad-newsqa-dev.json \
  --test_file ~/Data/newsqa/newsqa-squad2-dataset/squad-newsqa-test.json \
  --train_batch_size 256 \
  --train_batch_size_2 24 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --num_train_epochs_2 3 \
  --max_seq_length 128 \
  --max_seq_length_2 512 \
  --doc_stride 128 \
  --eval_metric best_f1 \
  --output_dir outputs/newsqa/retr \
  --output_dir_2 outputs/newsqa/read \
  --data_binary_dir data_binary/retr \
  --data_binary_dir_2 data_binary/read \
  --version_2_with_negative \
  --do_lower_case \
  --top_k 5 \
  --do_preprocess \
  --do_preprocess_2 \
  --first_stage \
```

In order to improve efficiency, we store data and model generated during training in a binary format. Specifically, when you switch on `do_preprocess`, the converted data in the first stage will be stored in the directory `data_binary`, next time you can switch off this option to directly load data. As well, `do_preprocess` aims at the data in the second stage, and `first_stage` is for the retriever model. The model and metrics result can be found in  the directory `output/newsqa` after training.

## License

[Soochow University](https://www.suda.edu.cn/) © [Mengxing Dong](https://github.com/Walle1493)
