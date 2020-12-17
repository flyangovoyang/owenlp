import os


class BertConfig:
    bert_path = '/home/fuyang/Project/bert-base-chinese'
    bert_config_path = os.path.join(bert_path, 'config.json')
    bert_tokenizer_path = os.path.join(bert_path, 'vocab.txt')
    bert_model_path = os.path.join(bert_path, 'pytorch_model.bin')

    max_seq_len = 100
    tag2id = {
        "O": 0,
        "B": 1,
        "M": 2,
        "E": 3,
        "S": 4
    }
    id2tag = {v: k for k, v in tag2id.items()}