import nltk
import sklearn
import re

train = [
    "A casa azul sobre a colina está desmoronando.",
    "Estamos atrasados!",

]
test = [
    "João ama Maria que ama Francisco que ama Carlos que não ama ninguém."
]
classes = ["artigo",
           "substantivo",
           "adjetivo",
           "verbo",
           "preposicao",
           "pontuacao"]
y = [0, 1, 2, 4, 0, 1, 3, 3, 3, 2]

y_test = [1, 3, 1,
          4, 3, 1,
          4, 3, 1,
          4, 2, 3, 1]


def tokenize(x, y):
    tokens = []
    for i in x:
        tk = re.split(" ", i)
        words = [None, None]
        words += [a.lower() for a in tk]
        words += [None, None]
        my = 0
        for j in range(2, len(tk) + 2):
            obj = [words[j - 2], words[j - 1], words[j], words[j + 1], words[j + 2]]
            tokens.append(obj)
            my += 1
    return tokens


train_tk = tokenize(train, y)
test_tk = tokenize(test, y_test)

from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(train_tk, y)
print(clf.predict(test_tk))
