from sklearn.feature_extraction.text import CountVectorizer
corpus = []
with open('data/content.txt', 'r', encoding='utf-8') as file:
    for line in file:
        corpus.append(line.strip())
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
weight = X.toarray()

import numpy as np
import lda

model = lda.LDA(n_topics=3, n_iter=20)
model.fit(np.asarray(weight))     # model.fit_transform(X) is also available
topic_word = model.topic_word_    # model.components_ also works

#文档-主题（Document-Topic）分布
doc_topic = model.doc_topic_

print(doc_topic)
# print(topic_word)
print([line.argmax() for line in doc_topic].count(0))
print([line.argmax() for line in doc_topic].count(1))
print([line.argmax() for line in doc_topic].count(2))