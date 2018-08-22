#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import logging
import json


logging.basicConfig(filename="generate_word_char_map_log",
                    filemode="w",
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)


def str_reverse(s):

    eles = s.split()
    eles.reverse()
    return " ".join(eles)


def group_handle(pairs, word_start_char_map, word_end_char_map):
    if len(pairs) < 2:
        return None
    words_sens, char_sens = zip(*pairs)
    words_sens = [s.split() for s in words_sens]
    if len(set([s[0] for s in words_sens])) != 1:
        logging.info("group error")

    if len(set(["".join(s[:2]) for s in words_sens])) == 1:
        return None

    w1 = set([s[1] for s in words_sens])
    w1_start_chars = [word_start_char_map.get(w, None) for w in w1]
    # 存放第二个词的首个字符
    w1_start_chars = set([y for y in w1_start_chars if y])
    # 存放第一个词的最后一个字符
    w0_end_char = word_end_char_map.get(words_sens[0][0], None)

    char_sens = [s.split() for s in char_sens]
    len_min = min([len(char_sen) for char_sen in char_sens])
    for i in xrange(1, len_min + 1):
        pre_size = ["".join(s[:i]) for s in char_sens]
        if len(set(pre_size)) > 1 and i != 1:
            chars = char_sens[0][:i - 1]

            if len(w1_start_chars) > 1:
                return " ".join(chars)
            # if len(chars) == 1:
            #     return " ".join(chars)
            # if w0_end_char == chars[-1] and w0_end_char != chars[-2]:
            #     return " ".join(chars)

            # if len(w1_start_chars) == 1:
            #     w1_start_chars = list(w1_start_chars)[0]
            #     if w1_start_chars not in set(chars):
            #         return ' '.join(chars)
                # if chars[-1] == list(w1_start_chars)[0] and chars[-2] != list(w1_start_chars)[0] and \
                #         char_sens[0][i] != list(w1_start_chars)[0]:
                #     return " ".join(chars[0:-1])
            return None


def pair_remove_word_char(pair, word_char_map):
    # 去除已经确认的 word char 匹配对
    word_sen, char_sen = pair
    words = word_sen.split()
    for word in words:
        if word in word_char_map:
            char = word_char_map[word]
            if not str(word_sen).startswith(word):
                break
            if not str(char_sen).startswith(char):
                break
            word_sen = word_sen.replace(word, '', 1).strip()
            char_sen = char_sen.replace(char, '', 1).strip()
        else:
            break
    return word_sen, char_sen


def save_word_char_map(word_char_map):
    f = open('data/word_char_map.json', 'w')
    f.write(json.dumps(word_char_map))
    f.close()


def generate_word_char_map(questions_file):
    questions = pd.read_csv(questions_file)
    questions = zip(questions['words'], questions['chars'])
    # questions_reverse = [(str_reverse(x[0]), str_reverse(x[1])) for x in questions]
    # total_questions = questions + questions_reverse

    word_char_map = {}

    total_words = []

    word_sentences, char_sentences = zip(*questions)
    for s in word_sentences:
        total_words.extend(s.split())
    total_words = set(total_words)
    new_pairs = questions
    group = []
    word_start_char_map = {}
    word_end_char_map = {}
    for pair in questions:
        word_start_char_map[pair[0].split()[0]] = pair[1].split()[0]
        word_end_char_map[pair[0].split()[-1]] = pair[1].split()[-1]

    for i in xrange(len(total_words) * 2):
        for pair in new_pairs:
            word_start_char_map[pair[0].split()[0]] = pair[1].split()[0]
            word_end_char_map[pair[0].split()[-1]] = pair[1].split()[-1]

        total_ques = sorted(new_pairs, key=lambda x: x[0])
        if len(new_pairs) == 0:
            break
        last_start_word = total_ques[0][0].split()[0]
        new_word_char_map = {}
        count = 0
        for pair in total_ques:
            word_sen = pair[0]
            char_sen = pair[1]
            if len(word_sen.split()) == len(char_sen.split()):
                pair_word_char_map = dict(zip(word_sen.split(), char_sen.split()))

                new_word_char_map.update(pair_word_char_map)
                count += 1
                logging.info("same sent count %d" % count)

                # logging.info("word_sent: %s " % word_sen)
                # logging.info("char_sent: %s " % char_sen)

            start_word = word_sen.split()[0]
            if len(word_sen.split()) == 1:
                new_word_char_map[word_sen] = char_sen
            elif last_start_word == start_word:
                group.append(pair)
            else:
                res = group_handle(group, word_start_char_map, word_end_char_map)
                if res:
                    new_word_char_map[last_start_word] = res
                group = []
                group.append(pair)
                last_start_word = start_word
        res = group_handle(group, word_start_char_map, word_end_char_map)
        if res:
            new_word_char_map[last_start_word] = res
        group = []

        # for word in new_word_char_map:
        #     if word in word_char_map:
        #         logging.info('old, %s, new, %s' % (word_char_map[word], new_word_char_map[word]))
        for w in new_word_char_map:
            if w not in word_char_map:
                word_char_map[w] = new_word_char_map[w]
        error = []
        for item in word_char_map.items():
            word, chars = item
            if word in word_start_char_map:
                if word_start_char_map[word] != chars.split()[0]:
                    error.append((word, chars))
            if word in word_end_char_map:
                if word_end_char_map[word] != chars.split()[-1]:
                    error.append((word, chars))
            # if word not in word_start_char_map:
            #     word_start_char_map[word] = chars.split()[0]
            # if word not in word_end_char_map:
            #     word_end_char_map[word] = chars.split()[-1]

        new_pairs = []
        for pair in total_ques:
            word_sen, char_sen = pair_remove_word_char(pair, word_char_map)
            if i > 35:
                if len(pair[0].split()) == len(pair[1].split()):
                    logging.info('pair[0]: %s' % pair[0])
                    logging.info('pair[1]: %s' % pair[1])
                    logging.info('word_sen: %s' % word_sen)
                    logging.info('char_sen: %s' % char_sen)
            if len(word_sen.strip()) > 0 and len(char_sen.strip()) > 0:
                new_pairs.append((word_sen, char_sen))

        logging.info('iter %d done ' % i)
        logging.info('word char map size %d ' % len(word_char_map))
        logging.info('new word char map size %d ' % len(new_word_char_map))
        if i == 35:
            save_word_char_map(word_char_map)
            pd.DataFrame(new_pairs, columns=['word_sen', 'char_sen']).to_csv(
                'data/remain_word_char_sen_pair')
        if len(new_word_char_map) == 0:

            pd.DataFrame(new_pairs, columns=['word_sen', 'char_sen']).to_csv(
                'data/remain_word_char_sen_pair')
            break
        if len(word_char_map) == len(total_words):
            save_word_char_map(word_char_map)
            break


if __name__ == '__main__':
    generate_word_char_map('data/question.csv')
