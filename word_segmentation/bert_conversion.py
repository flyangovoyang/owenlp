"""
data utility for target extraction

@Author: Young Fu
@Date: 2020-09-10
@Last Update: 2020-09-29
"""
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import logging
from target_extraction.crf import allowed_transitions

logging.basicConfig(format='[%(asctime)s]-%(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class InputFeature:
    def __init__(self, uid, input_ids, input_masks, tag_ids=None, token2char=None):
        self.uid = uid
        self.input_ids = input_ids
        self.tag_ids = tag_ids
        self.input_masks = input_masks
        self.token2char = token2char


class IOProcessor:
    """
    IOProcessor is responsible for
        (1)read corpus;
        (2)build vocabulary;
        (3)sentence vectorization;
        (4)build data loader;
        (5)build tag2id, id2tag

    Two most important function:
    (1) `_ch_seq_to_feature` convert character input representation into token input representation;
    (2) `token_tag_to_ch_tag`+`parse_entity_span_from_ch_tags` convert token prediction to character prediction

    notice: only support `BMES` tagging schema
    """

    def __init__(self, categories, max_seq_len, ptm_tokenizer):
        self.tag2id = self._build_tag2id(categories)
        self.ptm_tokenizer = ptm_tokenizer
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.max_seq_len = max_seq_len
        self.label_constraints = allowed_transitions("BMES", self.id2tag)

    @staticmethod
    def _build_tag2id(labels):
        tag_to_id = {'O': 0}
        position_symbols = ['B', 'M', 'E', 'S']
        for label in labels:
            for symbol in position_symbols:
                tag_to_id[symbol + '-' + label] = len(tag_to_id)
        return tag_to_id

    @staticmethod
    def _generate_position_symbol(tags):
        """ add position symbol to tag sequence (name, name, name, O, pos, pos) => (B-name, M-name, ....) """
        # forward scan
        mem = ''
        mark = [0] * len(tags)
        for i in range(len(tags)):
            if tags[i] != 'O':
                if tags[i] == mem:
                    mark[i] = mark[i - 1] + 1
                else:
                    mark[i] = 1
            mem = tags[i]
        # backward scan
        next_mark = 0
        symbol_res = ['O'] * len(tags)
        for i in range(len(tags) - 1, -1, -1):
            if mark[i] == 1:
                if next_mark <= 1:
                    symbol_res[i] = 'S'
                else:  # 2
                    symbol_res[i] = 'B'
            elif mark[i] >= 2:
                if next_mark > mark[i]:
                    symbol_res[i] = 'M'
                else:
                    symbol_res[i] = 'E'
            next_mark = mark[i]
        # merge
        merge_res = []
        for symbol, category in zip(symbol_res, tags):
            if symbol == 'O':
                merge_res.append('O')
            else:
                merge_res.append(symbol + '-' + category)
        return merge_res

    def _ch_seq_to_feature(self, uid, sent, tags=None):
        """
        convert (character + tag) sequence to (token + tag) id sequence

        Conversion including padding, truncating and vectorizing character sequence with pre-trained model tokenizer.
        For Chinese NER task, firstly convert character sequence to token sequence and modify the tag sequence according
        to `tokens_to_chars`. secondly, four quotations `“`, `”`, `’`, `‘` should be converted to their English
        counterparts to avoid `[unk]` noise.
        """
        sent_str = ''.join(sent).lower()

        en_style_sent_str = sent_str.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        batch_encoding = self.ptm_tokenizer.encode_plus(
            en_style_sent_str, truncation=True, max_length=self.max_seq_len, pad_to_max_length=True)
        sent_tokens = self.ptm_tokenizer.tokenize(en_style_sent_str)
        token_ids = batch_encoding['input_ids']
        input_masks = batch_encoding['attention_mask']
        tokens_len = len(token_ids)

        if tags is None:  # when predicting, tags is empty
            token2char = []
            for i in range(len(token_ids)):
                token2char.append(tuple(batch_encoding.token_to_chars(i)))
            if uid < 5:
                logging.debug('example-{}'.format(uid))
                logging.debug('sent_str({}):{}'.format(len(sent_str), sent_str))
                logging.debug('sent_tokens({}):{}'.format(len(sent_tokens), sent_tokens))
                logging.debug('token_ids({}):{}'.format(len(token_ids), token_ids))
                logging.debug('input_masks({}):{}'.format(len(input_masks), input_masks))
                logging.debug('token2char({}):{}'.format(len(token2char), token2char))
            return token_ids, input_masks, token2char

        # merge ch tags to token tag
        tokens_multi_tags = [[] for _ in range(len(token_ids))]
        token_merged_tags = ['O'] * len(token_ids)
        for i in range(tokens_len):
            span = batch_encoding.token_to_chars(i)  # when exceeds the character sequence, span=(0, 0)
            for j in range(span[0], span[1]):
                cur_category = tags[j] if tags[j] == 'O' else tags[j][2:]
                tokens_multi_tags[i].append(cur_category)  # make sure category symbol starts from index 2
        for i in range(tokens_len):
            tag_set = set(tokens_multi_tags[i]) - {'O'}  # remove the 'O' and count non 'O' categories
            if len(tag_set) == 1:
                token_merged_tags[i] = tag_set.pop()  # get the only one category
            elif len(tag_set) > 1:
                token_merged_tags[i] = 'mix'
                output_template = 'more than one ({}) categories on the single token {}, which is unexpected.'
                logging.warning(output_template.format(tag_set, sent_tokens[i + 1]))

        # add position symbols to token tags
        token_tags = IOProcessor._generate_position_symbol(token_merged_tags)
        token_tag_ids = [self.tag2id[tag] for tag in token_tags]

        if uid < 5:
            logging.debug('example-{}'.format(uid))
            logging.debug('sent_str({}):{}'.format(len(sent_str), sent_str))
            logging.debug('tags({}):{}'.format(len(tags), tags))
            logging.debug('sent_tokens({}):{}'.format(len(sent_tokens), sent_tokens))
            logging.debug('token_ids({}):{}'.format(len(token_ids), token_ids))
            logging.debug('token_tags({}):{}'.format(len(token_tags), token_tags))
            logging.debug('token_tag_ids({}):{}'.format(len(token_tag_ids), token_tag_ids))
            logging.debug('input_masks({}):{}'.format(len(input_masks), input_masks))
        return token_ids, input_masks, token_tag_ids

    @staticmethod
    def token_tag_to_ch_tag(token_tags, token2char):
        """ convert token tags sequence to character tags sequence """
        assert len(token_tags) == len(token2char)
        token_length = len(token_tags)  # n+2: [CLS] a1 ... an [SEP]

        # bug fixed:
        # token2char's length is n+2, ch's actual length can be obtained from the next to last(second from
        # bottom) index span (ch_s_index, ch_e_index), notice the last index span must be (0, 0)
        ch_tags = ['O'] * (token2char[-2][1])

        for i in range(1, token_length - 1):  # a1, ..., an
            if token_tags[i] != 'O':
                """
                |     token1    | token2 |     token3    |            token4
                |     B-xxx     | M-xxx  |     E-xxx     |            S-xxx

                | ch11  | ch12  |  ch2   |  ch31 | ch32  |        token41 token42
                | B-xxx | M-xxx | M-xxx  | M-xxx | E-xxx |         B-xxx   E-xxx

                for ch32, token3's end tag(E-xxx) should be preserved
                """
                pos_symbol = token_tags[i][0]
                category = token_tags[i][2:]
                s, e = token2char[i]
                if pos_symbol == 'B':
                    for j in range(s, e):
                        ch_tags[j] = token_tags[i] if j == s else 'M-' + category  # (B) -> (B, [M, M, ...])
                elif pos_symbol == 'M':
                    for j in range(s, e):
                        ch_tags[j] = token_tags[i]
                elif pos_symbol == 'E':
                    for j in range(s, e):
                        ch_tags[j] = token_tags[i] if j == e - 1 else 'M-' + category  # (E) -> ([M, ...,] E)
                else:  # S
                    if e == s + 1:
                        ch_tags[j] = token_tags[i]  # (S) -> (S)
                    else:
                        for j in range(s, e):  # (S) -> B[, [M, ] E]
                            if j == s:
                                ch_tags[j] = 'B-' + category
                            elif j == e - 1:
                                ch_tags[j] = 'E-' + category
                            else:
                                ch_tags[j] = 'M-' + category
        return ch_tags

    @staticmethod
    def parse_entity_indices_span_from_ch_tag(ch_tags):
        """ extract triples(start_index, end_index, target string) from tag sequence """
        entity_pairs = []
        index = 0
        while index < len(ch_tags):
            if ch_tags[index] != 'O':
                if ch_tags[index][0] == 'S':
                    entity_pairs.append((index, index + 1, ch_tags[index][2:]))
                elif ch_tags[index][0] == 'B':
                    s = index
                    while index < len(ch_tags) and (not ch_tags[index].startswith('E')) and ch_tags[index] != 'O':
                        index += 1
                    if index >= len(ch_tags) or ch_tags[index] == 'O':
                        entity_pairs.append((s, index, ch_tags[s][2:]))
                    else:
                        entity_pairs.append((s, index + 1, ch_tags[s][2:]))
            index += 1
        return entity_pairs

    def read_corpus_file(self, file_path, has_label=True):
        """
        load features from corpus file

        Notice: each line in labeled corpus file contains a single character and tag string separated by a single space
        """
        tokens = []
        tags = []
        uid = 0
        features = []
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.rstrip()  # character may be space symbol
                if not line and tokens:
                    if tags and len(tokens) != len(tags):
                        logging.warning('current sentence\'s length inconsistent: tokens({}):{}, tags({}):{}'.format(
                            len(tokens), tokens, len(tags), tags))
                        tokens, tags = [], []
                        continue
                    if self.ptm_tokenizer is not None:
                        if has_label:
                            input_ids, input_masks, tag_ids = self._ch_seq_to_feature(uid, tokens, tags)
                            features.append(
                                InputFeature(uid=uid, input_ids=input_ids, input_masks=input_masks, tag_ids=tag_ids))
                        else:
                            input_ids, input_masks, token2char = self._ch_seq_to_feature(uid, tokens)
                            features.append(
                                InputFeature(uid=uid, input_ids=input_ids, input_masks=input_masks,
                                             token2char=token2char))
                    else:
                        # TODO: feature implementation
                        raise Exception('TBD')

                    uid += 1
                    tokens, tags = [], []
                else:
                    if has_label:
                        ch, tag = line[0], line[2:]  # bug fixed: character can be empty some time
                        tokens.append(ch)
                        tags.append(tag)
                    else:
                        ch = line[0]
                        tokens.append(ch)

        logging.info('loading {} features in file {}'.format(len(features), file_path))
        return features

    def build_dataloader(self, corpus_file_path, batch_size, sample="random"):
        features = self.read_corpus_file(corpus_file_path)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.bool)
        if features[0].tag_ids is not None:
            all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_masks, all_tag_ids)
        else:
            all_token2char = torch.tensor([f.token2char for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_masks, all_token2char)
        if sample == 'random':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size, sampler=sampler)  # bug fixed: `sampler=sampler` not `sampler`
        return dataloader

# if __name__ == '__main__':
#     ptm_tokenizer = BertTokenizerFast('../pretrained_model/bert-base-chinese/vocab.txt')
#     io_processor = IOProcessor(['T'], 200, ptm_tokenizer)
#     eval_dataloader = io_processor.build_dataloader('../resource/absa-target/alarm.test.data', 1, sample='sequential')
#     for input_ids, input_masks, tag_ids in eval_dataloader:
#         print(ptm_tokenizer.decode(input_ids[0].tolist()))
#         assert 0
