import csv
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy
import numpy as np

column_labels = list(
    ['Mathematics', 'Logic', 'Animals', 'Theology', 'Philosophy', 'Mathematicians', 'Religion', 'Ancient_history',
     'Domesticated_animals', 'History', 'Philosophers', 'Ethics', 'War', 'Pets', 'Gods_and_goddesses'])
row_labels = column_labels

reader = csv.reader(open("../../../datasets/distances-final-embeddings.csv", "rb"), delimiter=",")
x = list(reader)

result = sorted(numpy.array(x).astype("string"), key=itemgetter(0))

print(result)

data = np.zeros([15, 15])
for i, val in enumerate(result):
    c = column_labels.index(val.item(0))
    r = row_labels.index(val.item(1))
    v = val.item(2)
    data[c, r] = v
    data[r, c] = v

fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.plasma)
ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
xticklabels = ax.set_xticklabels(row_labels, minor=False)
plt.setp(xticklabels, rotation=30)
ax.set_yticklabels(column_labels, minor=False)
plt.show()
