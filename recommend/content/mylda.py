import numpy as np
import random
from math import log, exp


class LDA:
    def __init__(self, alpha, beta, k, steps, random_state):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.steps = steps

        if random_state:
            random.seed(random_state)

    def fit(self, word_mat):
        # m是文档数，n是所有word数
        # 将array转化为dict
        m, n = word_mat.shape
        # 初始化变量
        # nmk:文档主题分布,nmk[i][j]表示第i篇文档中第k个topic产生的词的个数
        # nm:每个文档中词的个数
        # nkw:主题词分布，nkw[i][j]表示第k个topic中第j个word的个数
        # nk:每个主题中词的个数
        # z:文档中词的主题
        self.nmk = np.zeros((m, self.k), dtype=int)
        self.nm = np.zeros(m, dtype=int)
        self.nkw = np.zeros((self.k, n), dtype=int)
        self.nk = np.zeros(self.k, dtype=int)
        self.z = np.zeros((m, n), dtype=int)

        # 随机分配topic
        for i in range(m):
            for j in range(n):
                if word_mat[i][j] != 0:
                    # 采样topic索引,服从多项分布multi(1/k)
                    topic = random.randint(0, self.k - 1)
                    self.z[i][j] = topic
                    self.nmk[i][topic] += word_mat[i][j]
                    self.nm[i] += word_mat[i][j]
                    self.nkw[topic][j] += word_mat[i][j]
                    self.nk[topic] += word_mat[i][j]

        # Gibbs sampling
        for step in range(self.steps):
            for i in range(m):
                for j in range(n):
                    if word_mat[i][j] != 0:
                        topic = self.z[i][j]
                        self.nmk[i][topic] -= word_mat[i][j]
                        self.nm[i] -= word_mat[i][j]
                        self.nkw[topic][j] -= word_mat[i][j]
                        self.nk[topic] -= word_mat[i][j]
                        # 采样topic index
                        # 计算采样概率
                        p = (self.nmk[i] + self.alpha) / (self.nm[i] + self.k * self.alpha) * \
                            (self.nkw[:, j] + self.beta) / (self.nk + n * self.beta)
                        # 累计概率
                        for k in range(1, self.k):
                            p[k] += p[k - 1]
                        # 随机一个数确定topic
                        u = random.uniform(0, p[self.k - 1])
                        for topic in range(self.k):
                            if p[topic] > u:
                                break
                        self.z[i][j] = topic
                        self.nmk[i][topic] += word_mat[i][j]
                        self.nm[i] += word_mat[i][j]
                        self.nkw[topic][j] += word_mat[i][j]
                        self.nk[topic] += word_mat[i][j]
            # 计算preplexity
            self.theta = [(self.nmk[i] + self.alpha) / (self.nm[i] + self.k * self.alpha) for i in range(m)]
            self.phi = [(self.nkw[j] + self.beta) / (self.nk[j] + n * self.beta) for j in range(self.k)]
            print('第%d次迭代 perplexity:%f' % (step,self.perplexity(word_mat)))
        # 迭代结束，计算theta,phi
        # theta 文档-主题分布
        # phi 主题-词分布
        self.theta = [(self.nmk[i] + self.alpha) / (self.nm[i] + self.k * self.alpha) for i in range(m)]
        self.phi = [(self.nkw[j] + self.beta) / (self.nk[j] + n * self.beta) for j in range(self.k)]

        # 输出每一类的数量
        argmax_list = [line.argmax() for line in self.theta]
        for i in range(self.k):
            print('topic %d 数量:%d' % (i,argmax_list.count(i)))

        # print(self.theta)
        # print(self.phi)

    # 预测新文档
    def predict(self, doc_new):
        m, n = doc_new.shape
        nmk = np.zeros((m, self.k), dtype=int)  # 该文档主题分布下词的数量
        nm = np.zeros(m, dtype=int)  # 词的总数
        nkw = np.zeros((self.k, n), dtype=int)
        nk = np.zeros(self.k, dtype=int)
        z = np.zeros((m, n), dtype=int)  # 该文档词对应的主题
        # 随机初始化
        for i in range(m):
            for j in range(n):
                if doc_new[i][j] != 0:
                    topic = random.randint(0, self.k - 1)
                    z[i][j] = topic
                    nmk[i][topic] += 1
                    nm[i] += 1
                    nkw[topic][j] += 1
                    nk[topic] += 1
        for step in range(self.steps):
            for i in range(m):
                for j in range(n):
                    if doc_new[i][j] != 0:
                        topic = z[i][j]
                        nmk[i][topic] -= 1
                        nm[i] -= 1
                        nkw[topic][j] -= 1
                        nk[topic] -= 1
                        p = (nmk[i] + self.alpha) / (nm[i] + self.k * self.alpha) * \
                            (self.nkw[:, j] + nkw[:, j] + self.beta) / (self.nk + nk + n * self.beta)
                        # 累计概率
                        for k in range(1, self.k):
                            p[k] += p[k - 1]
                        # 随机一个数确定topic
                        u = random.uniform(0, p[self.k - 1])
                        for topic in range(self.k):
                            if p[topic] > u:
                                break
                        z[i][j] = topic
                        nmk[i][topic] += 1
                        nm[i] += 1
                        nkw[topic][j] += 1
                        nk[topic] += 1
        theta = [(nmk[i] + self.alpha) / (nm[i] + self.k * self.alpha) for i in range(m)]
        print(theta)

    # 验证模型的好坏,该值越小，模型越好
    def perplexity(self, data_mat):
        m, n = data_mat.shape
        sum_1, count = 0., 0
        for i in range(m):
            sum_2 = 0.
            for j in range(n):
                if data_mat[i][j] != 0:
                    count += 1
                    sum_3 = 0.
                    for z in range(self.k):
                        sum_3 += self.theta[i][z] * self.phi[z][j]
                    sum_2 += log(sum_3)
            sum_1 += sum_2
        return exp(-sum_1 / count)


def main():
    lda = LDA(0.1, 0.1, 3, 20, random_state=1)
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = []
    with open('data/content.txt', 'r', encoding='utf-8') as file:
        for line in file:
            corpus.append(line.strip())
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus)
    train = vectorizer.transform(corpus).toarray()
    test = vectorizer.transform(['健客网 订购 两个 周期 陈老师泄油汤 胖大夫荷香茶 肚子痛 无法 正常 排便']).toarray()
    lda.fit(train)
    lda.predict(test)


if __name__ == '__main__':
    main()
