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


def read_file(file_path):
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


print(read_file('pku_training.utf8'))
