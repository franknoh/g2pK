import os
import re
import nltk
import mecab
from jamo import h2j, j2h
from nltk.corpus import cmudict

try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')

BOUND_NOUNS = "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"
eng2kor = {
    'A': '에이',
    'B': '비',
    'C': '씨',
    'D': '디',
    'E': '이',
    'F': '에프',
    'G': '지',
    'H': '에이치',
    'I': '아이',
    'J': '제이',
    'K': '케이',
    'L': '엘',
    'M': '엠',
    'N': '엔',
    'O': '오',
    'P': '피',
    'Q': '큐',
    'R': '알',
    'S': '에스',
    'T': '티',
    'U': '유',
    'V': '브이',
    'W': '더블유',
    'X': '엑스',
    'Y': '와이',
    'Z': '지',
}


def adjust(arpabets):
    string = " " + " ".join(arpabets) + " $"
    string = re.sub("\d", "", string)
    string = string.replace(" T S ", " TS ")
    string = string.replace(" D Z ", " DZ ")
    string = string.replace(" AW ER ", " AWER ")
    string = string.replace(" IH R $", " IH ER ")
    string = string.replace(" EH R $", " EH ER ")
    string = string.replace(" $", "")

    return string.strip("$ ").split()


def to_choseong(arpabet):
    d = {
        'B': 'ᄇ',
        'CH': 'ᄎ',
        'D': 'ᄃ',
        'DH': 'ᄃ',
        'DZ': 'ᄌ',
        'F': 'ᄑ',
        'G': 'ᄀ',
        'HH': 'ᄒ',
        'JH': 'ᄌ',
        'K': 'ᄏ',
        'L': 'ᄅ',
        'M': 'ᄆ',
        'N': 'ᄂ',
        'NG': 'ᄋ',
        'P': 'ᄑ',
        'R': 'ᄅ',
        'S': 'ᄉ',
        'SH': 'ᄉ',
        'T': 'ᄐ',
        'TH': 'ᄉ',
        'TS': 'ᄎ',
        'V': 'ᄇ',
        'W': 'W',
        'Y': 'Y',
        'Z': 'ᄌ',
        'ZH': 'ᄌ'
    }

    return d.get(arpabet, arpabet)


def to_jungseong(arpabet):
    d = {
        'AA': 'ᅡ',
        'AE': 'ᅢ',
        'AH': 'ᅥ',
        'AO': 'ᅩ',
        'AW': 'ᅡ우',
        'AWER': "ᅡ워",
        'AY': 'ᅡ이',
        'EH': 'ᅦ',
        'ER': 'ᅥ',
        'EY': 'ᅦ이',
        'IH': 'ᅵ',
        'IY': 'ᅵ',
        'OW': 'ᅩ',
        'OY': 'ᅩ이',
        'UH': 'ᅮ',
        'UW': 'ᅮ'
    }
    return d.get(arpabet, arpabet)


def to_jongseong(arpabet):
    d = {
        'B': 'ᆸ',
        'CH': 'ᆾ',
        'D': 'ᆮ',
        'DH': 'ᆮ',
        'F': 'ᇁ',
        'G': 'ᆨ',
        'HH': 'ᇂ',
        'JH': 'ᆽ',
        'K': 'ᆨ',
        'L': 'ᆯ',
        'M': 'ᆷ',
        'N': 'ᆫ',
        'NG': 'ᆼ',
        'P': 'ᆸ',
        'R': 'ᆯ',
        'S': 'ᆺ',
        'SH': 'ᆺ',
        'T': 'ᆺ',
        'TH': 'ᆺ',
        'V': 'ᆸ',
        'W': 'ᆼ',
        'Y': 'ᆼ',
        'Z': 'ᆽ',
        'ZH': 'ᆽ'
    }

    return d.get(arpabet, arpabet)


def reconstruct(string):
    pairs = [("그W", "ᄀW"),
             ("흐W", "ᄒW"),
             ("크W", "ᄏW"),
             ("ᄂYᅥ", "니어"),
             ("ᄃYᅥ", "디어"),
             ("ᄅYᅥ", "리어"),
             ("Yᅵ", "ᅵ"),
             ("Yᅡ", "ᅣ"),
             ("Yᅢ", "ᅤ"),
             ("Yᅥ", "ᅧ"),
             ("Yᅦ", "ᅨ"),
             ("Yᅩ", "ᅭ"),
             ("Yᅮ", "ᅲ"),
             ("Wᅡ", "ᅪ"),
             ("Wᅢ", "ᅫ"),
             ("Wᅥ", "ᅯ"),
             ("Wᅩ", "ᅯ"),
             ("Wᅮ", "ᅮ"),
             ("Wᅦ", "ᅰ"),
             ("Wᅵ", "ᅱ"),
             ("ᅳᅵ", "ᅴ"),
             ("Y", "ᅵ"),
             ("W", "ᅮ")
             ]
    for str1, str2 in pairs:
        string = string.replace(str1, str2)
    return string


def parse_table():
    lines = open(os.path.dirname(os.path.abspath(__file__)) + '/table.csv', 'r', encoding='utf8').read().splitlines()
    onsets = lines[0].split(",")
    table = []
    for line in lines[1:]:
        cols = line.split(",")
        coda = cols[0]
        for i, onset in enumerate(onsets):
            cell = cols[i]
            if len(cell) == 0: continue
            if i == 0:
                continue
            else:
                str1 = f"{coda}{onset}"
                if "(" in cell:
                    str2 = cell.split("(")[0]
                    rule_ids = cell.split("(")[1][:-1].split("/")
                else:
                    str2 = cell
                    rule_ids = []

                table.append((str1, str2, rule_ids))
    return table


def annotate(string, mecab):
    tokens = mecab.pos(string)
    if string.replace(" ", "") != "".join(token for token, _ in tokens):
        return string
    blanks = [i for i, char in enumerate(string) if char == " "]

    tag_seq = []
    for token, tag in tokens:
        tag = tag.split("+")[-1]
        if tag == "NNBC":
            tag = "B"
        else:
            tag = tag[0]
        tag_seq.append("_" * (len(token) - 1) + tag)
    tag_seq = "".join(tag_seq)

    for i in blanks:
        tag_seq = tag_seq[:i] + " " + tag_seq[i:]

    annotated = ""
    for char, tag in zip(string, tag_seq):
        annotated += char
        if char == "의" and tag == "J":
            annotated += "/J"
        elif tag == "E":
            if h2j(char)[-1] in "ᆯ":
                annotated += "/E"
        elif tag == "V":
            if h2j(char)[-1] in "ᆫᆬᆷᆱᆰᆲᆴ":
                annotated += "/P"
        elif tag == "B":
            annotated += "/B"

    return annotated


def compose(letters):
    letters = re.sub("(^|[^\u1100-\u1112])([\u1161-\u1175])", r"\1ᄋ\2", letters)
    string = letters
    syls = set(re.findall("[\u1100-\u1112][\u1161-\u1175][\u11A8-\u11C2]", string))
    for syl in syls:
        string = string.replace(syl, j2h(*syl))

    syls = set(re.findall("[\u1100-\u1112][\u1161-\u1175]", string))
    for syl in syls:
        string = string.replace(syl, j2h(*syl))

    return string


def group(inp):
    inp = inp.replace("ᅢ", "ᅦ")
    inp = inp.replace("ᅤ", "ᅨ")
    inp = inp.replace("ᅫ", "ᅬ")
    inp = inp.replace("ᅰ", "ᅬ")

    return inp


def _get_examples():
    text = open('rules.txt', 'r', encoding='utf8').read().splitlines()
    examples = []
    for line in text:
        if line.startswith("->"):
            examples.extend(re.findall("([ㄱ-힣][ ㄱ-힣]*)\[([ㄱ-힣][ ㄱ-힣]*)]", line))
    _examples = []
    for inp, gt in examples:
        for each in gt.split("/"):
            _examples.append((inp, each))

    return _examples


def get_rule_id2text():
    rules = open(os.path.dirname(os.path.abspath(__file__)) + '/rules.txt', 'r', encoding='utf8').read().strip().split(
        "\n\n")
    rule_id2text = dict()
    for rule in rules:
        rule_id, texts = rule.splitlines()[0], rule.splitlines()[1:]
        rule_id2text[rule_id.strip()] = "\n".join(texts)
    return rule_id2text


rule_id2text = get_rule_id2text()


def gloss(verbose, out, inp, rule):
    if verbose and out != inp and out != re.sub("/[EJPB]", "", inp):
        print(compose(inp), "->", compose(out))
        print("\033[1;31m", rule, "\033[0m")


def link1(inp, verbose=False):
    rule = rule_id2text["13"]
    out = inp

    pairs = [("ᆨᄋ", "ᄀ"),
             ("ᆩᄋ", "ᄁ"),
             ("ᆫᄋ", "ᄂ"),
             ("ᆮᄋ", "ᄃ"),
             ("ᆯᄋ", "ᄅ"),
             ("ᆷᄋ", "ᄆ"),
             ("ᆸᄋ", "ᄇ"),
             ("ᆺᄋ", "ᄉ"),
             ("ᆻᄋ", "ᄊ"),
             ("ᆽᄋ", "ᄌ"),
             ("ᆾᄋ", "ᄎ"),
             ("ᆿᄋ", "ᄏ"),
             ("ᇀᄋ", "ᄐ"),
             ("ᇁᄋ", "ᄑ")]
    for str1, str2 in pairs:
        out = out.replace(str1, str2)

    gloss(verbose, out, inp, rule)
    return out


def link2(inp, verbose=False):
    rule = rule_id2text["14"]
    out = inp

    pairs = [("ᆪᄋ", "ᆨᄊ"),
             ("ᆬᄋ", "ᆫᄌ"),
             ("ᆰᄋ", "ᆯᄀ"),
             ("ᆱᄋ", "ᆯᄆ"),
             ("ᆲᄋ", "ᆯᄇ"),
             ("ᆳᄋ", "ᆯᄊ"),
             ("ᆴᄋ", "ᆯᄐ"),
             ("ᆵᄋ", "ᆯᄑ"),
             ("ᆹᄋ", "ᆸᄊ")]
    for str1, str2 in pairs:
        out = out.replace(str1, str2)

    gloss(verbose, out, inp, rule)
    return out


def link3(inp, verbose=False):
    rule = rule_id2text["15"]
    out = inp

    pairs = [("ᆨ ᄋ", " ᄀ"),
             ("ᆩ ᄋ", " ᄁ"),
             ("ᆫ ᄋ", " ᄂ"),
             ("ᆮ ᄋ", " ᄃ"),
             ("ᆯ ᄋ", " ᄅ"),
             ("ᆷ ᄋ", " ᄆ"),
             ("ᆸ ᄋ", " ᄇ"),
             ("ᆺ ᄋ", " ᄉ"),
             ("ᆻ ᄋ", " ᄊ"),
             ("ᆽ ᄋ", " ᄌ"),
             ("ᆾ ᄋ", " ᄎ"),
             ("ᆿ ᄋ", " ᄏ"),
             ("ᇀ ᄋ", " ᄐ"),
             ("ᇁ ᄋ", " ᄑ"),
             ("ᆪ ᄋ", "ᆨ ᄊ"),
             ("ᆬ ᄋ", "ᆫ ᄌ"),
             ("ᆰ ᄋ", "ᆯ ᄀ"),
             ("ᆱ ᄋ", "ᆯ ᄆ"),
             ("ᆲ ᄋ", "ᆯ ᄇ"),
             ("ᆳ ᄋ", "ᆯ ᄊ"),
             ("ᆴ ᄋ", "ᆯ ᄐ"),
             ("ᆵ ᄋ", "ᆯ ᄑ"),
             ("ᆹ ᄋ", "ᆸ ᄊ")]

    for str1, str2 in pairs:
        for h in ['ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅟ']:
            out = out.replace(str1 + h, str2 + h)

    gloss(verbose, out, inp, rule)
    return out


def link4(inp, verbose=False):
    rule = rule_id2text["12.4"]

    out = inp

    pairs = [("ᇂᄋ", "ᄋ"),
             ("ᆭᄋ", "ᄂ"),
             ("ᆶᄋ", "ᄅ")]

    for str1, str2 in pairs:
        out = out.replace(str1, str2)

    gloss(verbose, out, inp, rule)
    return out


def jyeo(inp, verbose=False):
    rule = rule_id2text["5.1"]
    out = re.sub("([ᄌᄍᄎ])ᅧ", r"\1ᅥ", inp)
    gloss(verbose, out, inp, rule)
    return out


def ye(inp, descriptive=False, verbose=False):
    rule = rule_id2text["5.2"]
    if descriptive:
        out = re.sub("([ᄀᄁᄃᄄㄹᄆᄇᄈᄌᄍᄎᄏᄐᄑᄒ])ᅨ", r"\1ᅦ", inp)
    else:
        out = inp
    gloss(verbose, out, inp, rule)
    return out


def consonant_ui(inp, verbose=False):
    rule = rule_id2text["5.3"]
    out = re.sub("([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ", r"\1ᅵ", inp)
    gloss(verbose, out, inp, rule)
    return out


def josa_ui(inp, descriptive=False, verbose=False):
    rule = rule_id2text["5.4.2"]
    if descriptive:
        out = re.sub("의/J", "에", inp)
    else:
        out = inp.replace("/J", "")
    gloss(verbose, out, inp, rule)
    return out


def vowel_ui(inp, descriptive=False, verbose=False):
    rule = rule_id2text["5.4.1"]
    if descriptive:
        out = re.sub("(\Sᄋ)ᅴ", r"\1ᅵ", inp)
    else:
        out = inp
    gloss(verbose, out, inp, rule)
    return out


def jamo(inp, verbose=False):
    rule = rule_id2text["16"]
    out = inp

    out = re.sub("([그])ᆮᄋ", r"\1ᄉ", out)
    out = re.sub("([으])[ᆽᆾᇀᇂ]ᄋ", r"\1ᄉ", out)
    out = re.sub("([으])[ᆿ]ᄋ", r"\1ᄀ", out)
    out = re.sub("([으])[ᇁ]ᄋ", r"\1ᄇ", out)

    gloss(verbose, out, inp, rule)
    return out


def rieulgiyeok(inp, verbose=False):
    rule = rule_id2text["11.1"]

    out = inp
    out = re.sub("ᆰ/P([ᄀᄁ])", r"ᆯᄁ", out)

    gloss(verbose, out, inp, rule)
    return out


def rieulbieub(inp, verbose=False):
    rule = rule_id2text["25"]
    out = inp

    out = re.sub("([ᆲᆴ])/Pᄀ", r"\1ᄁ", out)
    out = re.sub("([ᆲᆴ])/Pᄃ", r"\1ᄄ", out)
    out = re.sub("([ᆲᆴ])/Pᄉ", r"\1ᄊ", out)
    out = re.sub("([ᆲᆴ])/Pᄌ", r"\1ᄍ", out)

    gloss(verbose, out, inp, rule)
    return out


def verb_nieun(inp, verbose=False):
    rule = rule_id2text["24"]
    out = inp

    pairs = [("([ᆫᆷ])/Pᄀ", r"\1ᄁ"),
             ("([ᆫᆷ])/Pᄃ", r"\1ᄄ"),
             ("([ᆫᆷ])/Pᄉ", r"\1ᄊ"),
             ("([ᆫᆷ])/Pᄌ", r"\1ᄍ"),

             ("ᆬ/Pᄀ", "ᆫᄁ"),
             ("ᆬ/Pᄃ", "ᆫᄄ"),
             ("ᆬ/Pᄉ", "ᆫᄊ"),
             ("ᆬ/Pᄌ", "ᆫᄍ"),

             ("ᆱ/Pᄀ", "ᆷᄁ"),
             ("ᆱ/Pᄃ", "ᆷᄄ"),
             ("ᆱ/Pᄉ", "ᆷᄊ"),
             ("ᆱ/Pᄌ", "ᆷᄍ")]

    for str1, str2 in pairs:
        out = re.sub(str1, str2, out)

    gloss(verbose, out, inp, rule)
    return out


def balb(inp, verbose=False):
    rule = rule_id2text["10.1"]
    out = inp
    syllable_final_or_consonants = "($|[^ᄋᄒ])"

    # exceptions
    out = re.sub(f"(바)ᆲ({syllable_final_or_consonants})", r"\1ᆸ\2", out)
    out = re.sub(f"(너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ)", r"\1ᆸ\2", out)
    gloss(verbose, out, inp, rule)
    return out


def palatalize(inp, verbose=False):
    rule = rule_id2text["17"]
    out = inp

    out = re.sub("ᆮᄋ([ᅵᅧ])", r"ᄌ\1", out)
    out = re.sub("ᇀᄋ([ᅵᅧ])", r"ᄎ\1", out)
    out = re.sub("ᆴᄋ([ᅵᅧ])", r"ᆯᄎ\1", out)

    out = re.sub("ᆮᄒ([ᅵ])", r"ᄎ\1", out)

    gloss(verbose, out, inp, rule)
    return out


def modifying_rieul(inp, verbose=False):
    rule = rule_id2text["27"]
    out = inp

    pairs = [("ᆯ/E ᄀ", r"ᆯ ᄁ"),
             ("ᆯ/E ᄃ", r"ᆯ ᄄ"),
             ("ᆯ/E ᄇ", r"ᆯ ᄈ"),
             ("ᆯ/E ᄉ", r"ᆯ ᄊ"),
             ("ᆯ/E ᄌ", r"ᆯ ᄍ"),

             ("ᆯ걸", "ᆯ껄"),
             ("ᆯ밖에", "ᆯ빠께"),
             ("ᆯ세라", "ᆯ쎄라"),
             ("ᆯ수록", "ᆯ쑤록"),
             ("ᆯ지라도", "ᆯ찌라도"),
             ("ᆯ지언정", "ᆯ찌언정"),
             ("ᆯ진대", "ᆯ찐대")]

    for str1, str2 in pairs:
        out = re.sub(str1, str2, out)

    gloss(verbose, out, inp, rule)
    return out


def word_to_hangul(word):
    ret = ''
    for alpha in word:
        ret += eng2kor[alpha]
    return ret


def convert_eng(string, cmu):
    words_set = set(re.findall(r"[a-z][a-z']*[a-z]|[a-z]", string, flags=re.I))
    eng_words = sorted(words_set, key=len, reverse=True)
    for eng_word in eng_words:
        if eng_word.isupper() or (eng_word.lower() not in cmu):
            ret = word_to_hangul(eng_word.upper())
            string = string.replace(eng_word, ret)
            continue
        word = eng_word.lower()
        arpabets = cmu[word][0]
        phonemes = adjust(arpabets)
        ret = ""
        for i in range(len(phonemes)):
            p = phonemes[i]
            p_prev = phonemes[i - 1] if i > 0 else "^"
            p_next = phonemes[i + 1] if i < len(phonemes) - 1 else "$"
            p_next2 = phonemes[i + 1] if i < len(phonemes) - 2 else "$"
            short_vowels = ("AE", "AH", "AX", "EH", "IH", "IX", "UH")
            vowels = "AEIOUY"
            consonants = "BCDFGHJKLMNPQRSTVWXZ"
            syllable_final_or_consonants = "$BCDFGHJKLMNPQRSTVWXZ"
            if p in "PTK":
                if p_prev[:2] in short_vowels and p_next == "$":  # 1
                    ret += to_jongseong(p)
                elif p_prev[:2] in short_vowels and p_next[0] not in "AEIOULRMN":  # 2
                    ret += to_jongseong(p)
                elif p_next[0] in "$BCDFGHJKLMNPQRSTVWXYZ":  # 3
                    ret += to_choseong(p)
                    ret += "ᅳ"
                else:
                    ret += to_choseong(p)
            elif p in "BDG":
                ret += to_choseong(p)
                if p_next[0] in syllable_final_or_consonants:
                    ret += "ᅳ"
            elif p in ("S", "Z", "F", "V", "TH", "DH", "SH", "ZH"):
                ret += to_choseong(p)
                if p in ("S", "Z", "F", "V", "TH", "DH"):  # 1
                    if p_next[0] in syllable_final_or_consonants:
                        ret += "ᅳ"
                elif p == "SH":  # 2
                    if p_next[0] in "$":
                        ret += "ᅵ"
                    elif p_next[0] in consonants:
                        ret += "ᅲ"
                    else:
                        ret += "Y"
                elif p == "ZH":  # 3
                    if p_next[0] in syllable_final_or_consonants:
                        ret += "ᅵ"
            elif p in ("TS", "DZ", "CH", "JH",):
                ret += to_choseong(p)  # 2

                if p_next[0] in syllable_final_or_consonants:  # 1
                    if p in ("TS", "DZ"):
                        ret += "ᅳ"
                    else:
                        ret += "ᅵ"
            elif p in ("M", "N", "NG"):
                if p in "MN" and p_next[0] in vowels:
                    ret += to_choseong(p)
                else:
                    ret += to_jongseong(p)
            elif p == "L":
                if p_prev == "^":
                    ret += to_choseong(p)
                elif p_next[0] in "$BCDFGHJKLPQRSTVWXZ":
                    ret += to_jongseong(p)
                elif p_prev in "MN":
                    ret += to_choseong(p)
                elif p_next[0] in vowels:
                    ret += "ᆯᄅ"
                elif p_next in "MN" and p_next2[0] not in vowels:
                    ret += "ᆯ르"
            elif p == "ER":
                if p_prev[0] in vowels:
                    ret += "ᄋ"
                ret += to_jungseong(p)
                if p_next[0] in vowels:
                    ret += "ᄅ"
            elif p == "R":
                if p_next[0] in vowels:
                    ret += to_choseong(p)
            elif p[0] in "AEIOU":
                ret += to_jungseong(p)

            else:
                ret += to_choseong(p)

        ret = reconstruct(ret)
        ret = compose(ret)
        ret = re.sub("[\u1100-\u11FF]", "", ret)
        string = string.replace(eng_word, ret)
    return string


def process_num(num, sino=True):
    num = re.sub(",", "", num)
    if num == "0":
        return "영"
    if not sino and num == "20":
        return "스무"
    digits = "123456789"
    names = "일이삼사오육칠팔구"
    digit2name = {d: n for d, n in zip(digits, names)}
    modifiers = "한 두 세 네 다섯 여섯 일곱 여덟 아홉"
    decimals = "열 스물 서른 마흔 쉰 예순 일흔 여든 아흔"
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}
    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        name = None
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = digit2name.get(digit, "") + "십"
                name = name.replace("일십", "십")
        else:
            if i == 0:
                name = digit2mod.get(digit, "")
            elif i == 1:
                name = digit2dec.get(digit, "")
        if digit == '0':
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)):]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        if i == 2:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 3:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 4:
            name = digit2name.get(digit, "") + "만"
            name = name.replace("일만", "만")
        elif i == 5:
            name = digit2name.get(digit, "") + "십"
            name = name.replace("일십", "십")
        elif i == 6:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 7:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 8:
            name = digit2name.get(digit, "") + "억"
        elif i == 9:
            name = digit2name.get(digit, "") + "십"
        elif i == 10:
            name = digit2name.get(digit, "") + "백"
        elif i == 11:
            name = digit2name.get(digit, "") + "천"
        elif i == 12:
            name = digit2name.get(digit, "") + "조"
        elif i == 13:
            name = digit2name.get(digit, "") + "십"
        elif i == 14:
            name = digit2name.get(digit, "") + "백"
        elif i == 15:
            name = digit2name.get(digit, "") + "천"
        elif i == 16:
            name = digit2name.get(digit, "") + "경"
        elif i == 17:
            name = digit2name.get(digit, "") + "십"
        elif i == 18:
            name = digit2name.get(digit, "") + "백"
        elif i == 19:
            name = digit2name.get(digit, "") + "천"
        if name is not None:
            spelledout.append(name)
        else:
            return num
    return "".join(elem for elem in spelledout)


def convert_num(string):
    global BOUND_NOUNS
    tokens = set(re.findall(r"(\d[\d,]*\d|\d)(\s*[ㄱ-힣]+(?=/B))", string))
    tokens = sorted(tokens, key=len, reverse=True)
    for token in tokens:
        num, bn = token
        bn_s = bn.lstrip()
        if bn_s in BOUND_NOUNS:
            spelledout = process_num(num, sino=False)
        else:
            spelledout = process_num(num, sino=True)
        string = string.replace(f"{num}{bn}/B", f"{spelledout}{bn}/B")
    remain = set(re.findall(r"(\d[\d,]*\d|\d)", string))
    remain = sorted(remain, key=len, reverse=True)
    for num in remain:
        string = string.replace(num, process_num(num, sino=True))
    digits = "0123456789"
    names = "영일이삼사오육칠팔구"
    for d, n in zip(digits, names):
        string = string.replace(d, n)
    pairs = [("십육", "심뉵"), ("백육", "뱅뉵")]
    for str1, str2 in pairs:
        string = string.replace(str1, str2)
    return string


class G2p(object):
    def __init__(self):
        self.mecab = mecab.MeCab()
        self.table = parse_table()
        self.cmu = cmudict.dict()
        self.rule2text = get_rule_id2text()
        self.idioms_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idioms.txt")

    def idioms(self, string, verbose=False):
        rule = "from idioms.txt"
        out = string

        for line in open(self.idioms_path, 'r', encoding="utf8"):
            line = line.split("#")[0].strip()
            if "===" in line:
                str1, str2 = line.split("===")
                out = re.sub(str1, str2, out)
        gloss(verbose, out, string, rule)

        return out

    def __call__(self, string, descriptive=False, verbose=False, group_vowels=False, to_syl=True):
        string = self.idioms(string, verbose)
        string = convert_eng(string, self.cmu)
        string = annotate(string, self.mecab)
        string = convert_num(string)
        inp = h2j(string)
        inp = jyeo(inp, verbose)
        inp = ye(inp, descriptive, verbose)
        inp = consonant_ui(inp, verbose)
        inp = josa_ui(inp, descriptive, verbose)
        inp = vowel_ui(inp, descriptive, verbose)
        inp = jamo(inp, verbose)
        inp = rieulgiyeok(inp, verbose)
        inp = rieulbieub(inp, verbose)
        inp = verb_nieun(inp, verbose)
        inp = balb(inp, verbose)
        inp = palatalize(inp, verbose)
        inp = modifying_rieul(inp, verbose)
        inp = re.sub("/[PJEB]", "", inp)
        for str1, str2, rule_ids in self.table:
            _inp = inp
            inp = re.sub(str1, str2, inp)
            if len(rule_ids) > 0:
                rule = "\n".join(self.rule2text.get(rule_id, "") for rule_id in rule_ids)
            else:
                rule = ""
            gloss(verbose, inp, _inp, rule)
        inp = link1(inp, verbose)
        inp = link2(inp, verbose)
        # inp = link3(inp, verbose)
        inp = link4(inp, verbose)
        if group_vowels:
            inp = group(inp)
        if to_syl:
            inp = compose(inp)
        return inp
