import csv
import numpy
from matplotlib import pylab
from operator import itemgetter
from sklearn.manifold import TSNE

reader = csv.reader(open("../../../datasets/centroids.csv", "rb"), delimiter=",")
x = list(reader)

result = sorted(numpy.array(x).astype("string"), key=itemgetter(0))

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=200)
dataset = map(lambda x: x.tolist(), result)

vectors = map(lambda x: map(lambda y: float(y), x[1:]), dataset)
labels = map(lambda x: x[0], dataset)
cat_2d_embeddings = tsne.fit_transform(vectors)


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()


print(result)

from sklearn.decomposition import PCA

X = vectors
y = labels
target_names = labels

pca = PCA(n_components=1)
X_r = pca.fit(X).transform(X)

print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))

pca_axis = zip(vectors[0], y)
print(pca_axis)

plot(cat_2d_embeddings, labels)
