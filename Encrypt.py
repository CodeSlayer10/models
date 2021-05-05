from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense
import numpy as np
import wandb
from wandb.keras import WandbCallback
import random
from keras.models import load_model
from sympy import *


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


# Parameters for the model and dataset.
class config:
    def __init__(self, training_size, digits, hidden_size, batch_size):
        self.training_size = training_size
        self.digits = digits
        self.hidden_size = hidden_size
        self.batch_size = batch_size





config.training_size = 50000
config.digits = 4

config.hidden_size = 256
config.batch_size = 256

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.

maxlen = config.digits + 4 + config.digits

print(maxlen)
# # All the numbers, plus sign and space for padding.
# list of not:
# chars maxlen ctable

chars = '0123456789: '
temp = []
temp1 = []
abc = "abcdefghijklmnopqrstuvwxyz"
for char in abc:
    index = abc.index(char)
    temp.append(str(index))
abcN = "".join(temp)
abcN2 = ":".join(temp)
a1 = list(abc)
random.shuffle(a1)
abc1 = "".join(a1)
for char in abc1:
    index = abc.index(char)
    temp1.append(str(index))
abcN1 = ":".join(temp1)
print(len(chars))

ctable = CharacterTable(chars)
query1 = ""
questions1 = []
expected1 = []
questions = []
expected = []

seen = set()


class NtL:
    def __init__(self, numList):
        self.numList = numList.split(":")
        self.result = []
        self.result1 = ""

    def NtL1(self):
        for num in self.numList:
            num = int(num)
            print(self.numList, "--")
            self.result.append(abc[num])
        self.result1 = "".join(self.result)
        return self.result1

    def NtL2(self):
        for num in self.numList:
            print(self.numList)
            num = int(num)
            print(abc1[0])
            self.result.append(abc1[num])
            # print(self.result)

        self.result1 = "".join(self.result)
        return self.result1




print(abcN2)
print(abcN1)
print('Generating data...')
while len(questions) < config.training_size:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                            for i in range(np.random.randint(1, config.digits + 1))))
    a = f()

    key = a
    if key in seen:
        continue
    seen.add(key)
    q = '{}'.format(a)
    query = q
    list4 = list(query)
    ansz = []
    for t, char in enumerate(list4):
        list4[t] = int(char)
    for t, char in enumerate(list4):
        if not t == len(query)-1:
            numT = int("{}{}".format(char, list4[t+1]))
            if not numT > 25:
                list4[t] = numT
    for t, char in enumerate(list4):
        list4[t] = str(char)
    query1 = ":".join(list4)





    for char in list4:
        index = abcN1.split(":").index(char)
        ansz.append(str(index))
    ans = ":".join(ansz)
    questions.append(query1)
    expected.append(ans)




print('Total questions:', len(questions))
print('Total answers:', len(expected))
# NtL1 = NtL(questions[0])
# NtL2 = NtL(expected[0])
# ans4 = NtL1.NtL2()
# ans5 = NtL2.NtL1()

# print(ans4, "---")
# print(ans5)
# print(questions[0])
# print("----------")
# print(expected[0])
# print(questions)
# print('-----------')
# print(expected)
# print('-----------')
print('Vectorization...')
x = np.zeros((len(questions), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), config.digits*3, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, maxlen)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, config.digits*3)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

model = Sequential()
model.add(LSTM(config.hidden_size, input_shape=(maxlen, len(chars))))
model.add(RepeatVector(config.digits*3))
model.add(LSTM(config.hidden_size, return_sequences=True))

model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 50):
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=config.batch_size,
              epochs=1,
              validation_data=(x_val, y_val))
    model.save('encryption_model_test')
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q, end=' ')
        print('T \n', correct, end=' ')
        # if correct == guess:
        #     print('☑', end=' ')
        # else:
        #     print('☒', end=' ')
        print('G', guess, end=' ')
