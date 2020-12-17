from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import torch
from config import BertConfig as config
import regex


class Sample:
    def __init__(self, sent, tag):
        assert len(sent) == len(tag)
        self.sent = sent
        self.tag = tag

    def __str__(self):
        return '({},{})'.format(self.sent, self.tag)

    __repr__ = __str__


def character_conversion(ustring):
    """
    全角转半角，多个连续控制符、空格替换成单个空格
    """

    if not ustring.strip():
        return ustring.strip()

    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return regex.sub(r'[\p{Z}\s]+', ' ', rstring.strip())


class DataProcessor:
    @staticmethod
    def _read_file(file_path):
        samples = []
        with open(file_path, encoding='utf8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                line_list = [character_conversion(s) for s in line.split('  ')]
                cur_seq_tag = []
                for item in line_list:
                    if len(item) == 1:
                        cur_seq_tag.append('S')
                    elif len(item) > 1:
                        cur_seq_tag.append('B')
                        for i in range(len(item) - 2):
                            cur_seq_tag.append('M')
                        cur_seq_tag.append('E')
                    else:
                        raise Exception('should not happend')
                samples.append(Sample(''.join(line_list), cur_seq_tag))
        return samples

    @staticmethod
    def _build_dataloader(samples, tokenizer, verbose):
        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []
        all_label_ids = []

        idx, encoding_batch_size = 0, 100
        while idx < len(samples):
            batch_samples = samples[idx: idx + encoding_batch_size]
            batch_labels = [sample.tag for sample in batch_samples]
            batch_samples = [sample.sent for sample in batch_samples]
            batch_encoding = tokenizer.batch_encode_plus(
                batch_samples, truncation=True, max_length=config.max_seq_len + 2, padding='max_length')
            input_ids = batch_encoding['input_ids']
            token_type_ids = batch_encoding['token_type_ids']
            attention_mask = batch_encoding['attention_mask']
            label_ids = [[config.tag2id[tag] for tag in tags] for tags in batch_labels]

            if idx == 0 and verbose:
                for i in range(5):
                    print('example-{}'.format(i))
                    print('input text:{}'.format(batch_samples[i]))
                    print('label:{}'.format(batch_labels[i]))

                    print('input ids:{}'.format(input_ids[i]))
                    print('attention_mask:{}'.format(attention_mask[i]))
                    print('token type ids:{}'.format(token_type_ids[i]))
                    print('label ids:{}'.format(label_ids[i]))
                    print()

            all_input_ids.extend(input_ids)
            all_token_type_ids.extend(token_type_ids)
            all_attention_mask.extend(attention_mask)
            all_label_ids.extend(label_ids)
            idx += encoding_batch_size

        dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
                                torch.tensor(all_attention_mask, dtype=torch.long),
                                torch.tensor(all_token_type_ids, dtype=torch.long),
                                torch.tensor(all_label_ids, dtype=torch.long))
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
        return dataloader

    @staticmethod
    def load_dataloader(tokenizer, file_path, verbose=False):
        samples = DataProcessor._read_file(file_path)
        dataloader = DataProcessor._build_dataloader(samples, tokenizer, verbose)
        return dataloader


if __name__ == '__main__':
    print(max([len(x.sent) for x in DataProcessor._read_file('pku_training.utf8')])) # 1019
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast(config.bert_tokenizer_path)
    dataloader = DataProcessor.load_dataloader(tokenizer, 'pku_training.utf8', verbose=True)
    for batch in dataloader:
        a, b, c, d = batch
        print(a, b, c, d)
        break