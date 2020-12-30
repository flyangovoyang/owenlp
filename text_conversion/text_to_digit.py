def text_to_digit(text):
    digit2value = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    base2value = {'个': 0, '十': 1, '百': 2, '千': 3, '万': 4, '亿': 8}

    num = []
    for ch in text:
        if ch in digit2value:
            num.append(digit2value[ch])
        elif ch in base2value:
            if ch == '十' and not num:
                num.append(1)
            while len(num) >= 2 and len(str(num[-2])) - 1 < base2value[ch]:
                top = num.pop()
                num[-1] = num[-1] + top
            num[-1] = num[-1]*(10**base2value[ch])
    if len(num) == 1:
        return num[0]
    else:
        if len(str(num[0])) > len(str(num[1])):
            return sum(num)
        else:
            return ''.join(str(x) for x in num)


print(text_to_digit('十一'))