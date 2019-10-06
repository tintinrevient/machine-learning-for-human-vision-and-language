import torch
import numpy

with open('../data/pride-and-prejudice.txt', encoding='utf8') as file:
    text = file.read()

lines = text.split('\n')
line = lines[200]

letter_one_hot = torch.zeros(len(line), 2**7)

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_one_hot[i][letter_index] = 1

def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

word_list = sorted(set(clean_words(text)))
word_dict = {word: i for (i, word) in enumerate(word_list)}

words_in_line = clean_words(line)
word_one_hot = torch.zeros(len(words_in_line), len(word_dict))
for i, word in enumerate(words_in_line):
    word_index = word_dict[word]
    word_one_hot[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))

print(word_one_hot.shape)
