import re


def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    return answer


def remove_special_characters(text):
    pattern = (
        r"[-`\\【】\*\$、,，。.；;:：？\?！!\s\n\u4e00-\u9fff0-9①②③④⑤⑥⑦\[\]\<>a-z=\'\"\(\)\{\}]+"
    )
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def process_multiple_choice(answer):
    answer = strip_answer(answer)
    key_words = ["故选", "正确选项为", "答案选", "答案为", "答案是", "答案"]

    for key_word in key_words:
        if key_word in answer:
            answer = answer.split(key_word)[-1]
            break
    answer = remove_special_characters(answer)
    # keep the last line
    answer = answer.split("\n")[-1]
    pattern = r"[A-Z]"
    matches = re.findall(pattern, answer)
    return "".join(matches)


def remove_unit(value):
    units = ["cm", "m", "s", "h", "kg", "g", "l", "ml", "km", "mm"]
    unit_pattern = r"^(\d+)(?:" + "|".join(units) + ")$"
    match = re.match(unit_pattern, value)
    if match:
        return match.group(1)
    else:
        return value


def normalize_string(raw_answer):
    if "$" not in raw_answer:
        wrong_answer_words = ["\\times", "不对", "不正确", "×", "x", "X"]
        for word in wrong_answer_words:
            raw_answer = raw_answer.replace(word, "错误")
    raw_answer = re.sub(r"\\text\s*\{(.*?)\}", r"\1", raw_answer)
    replace_dict = {
        "√": "正确",
        "：": ":",
        "$": "",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "变小": "减小",
        "变大": "增大",
        "路程": "距离",
        "\\pi": "π",
    }
    for k, v in replace_dict.items():
        raw_answer = raw_answer.replace(k, v)
    key_words = ["答案为", "答案是", "答案", "为", "因此", "结果", " = "]
    # get text after key_word
    for key_word in key_words:
        if key_word in raw_answer:
            raw_answer = raw_answer.split(key_word)[-1]
            break
    raw_answer = raw_answer.strip()
    # remove leading :
    if raw_answer.startswith(":"):
        raw_answer = raw_answer[1:]
    if len(raw_answer) > 0 and raw_answer[-1] in [".", ",", ":", ";"]:
        raw_answer = raw_answer[:-1]
    raw_answer = remove_unit(raw_answer)
    return raw_answer.strip()
