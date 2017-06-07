# Wiki Embeddings Research

### Preparing datasets
##### Download pretrained GloVe model
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip -d glove.6B
```
##### Download wikipedia dump
```
https://github.com/LGDoor/Dump-of-Simple-English-Wiki/blob/master/corpus.tgz
```
##### Place the downloaded datasets so that the dataset file looks like this:
```
$ cd datasets 
$ tree
    .
    ├── articles_dict
    ├── categories
    ├── cats_dict
    ├── corpus.txt
    └── glove.6B
        ├── glove.6B.100d.txt
        ├── glove.6B.200d.txt
        ├── glove.6B.300d.txt
        └── glove.6B.50d.txt
```

### Accuracy issues
article belongs to many categories
articleId -> Set(cat1Id, cat2Id, ...)

The following approaches have been tried but something better has to be set up.
##### Approach 1

articleId -> cat1Id,

articleId -> cat2Id

articleId -> ...

The issue is that the accuracy is low because the categories are somehow mixed

##### Approach 1

articleId -> cat1Id

The issue is that we loose many connections 

##### Creating embeddings:
1. each word is mapped onto one of its closest neighbours with equal probability 0.5
2. creating vocab dict of arbitrary size (10.000) most frequent words
2. initialize embeddings: creating matrix 10.000 x 100 (vocabulary size x embeddings size) with random values from U(-1,1)
3. in each batch (size: 128) we take the embeddings for words used in this batch
4. softmax weights are initialized with mean: 0 and sd=0.1, bias weights: 0
5. mean sample softmax loss for batch is calculated (sampled softmax: https://arxiv.org/pdf/1412.2007.pdf)
6. loss is optmized using Adagrad(1) optimizer
7. after optimization embeddings are normalized by dividing by L2 norm


#### TODO
- kilka grup kategorii np matematycy, filozxofowie, zwierzeta, historia
- matematycy powinni byc blisko filozofow, dalej zwierzat

- konwersja artykulx(vector + kategoria) -> odleglosc miedzy kategoriami
- albo wrzucic na NN te vectory
- albo zsumowac i policzyc odleglosc vectorow

- czy dodawanie nowych danych obniza jakosc?


### Manual verification

The following categories are a test set:
```
History	6602
Ancient_history	29636
War	38468


Animals	5861
Pets	14654
Domesticated_animals	33670


Mathematics	5195
Mathematicians	19894
Logic	41358


Philosophy	6536
Philosophers	5375
Ethics	25540
```

Expected results:
```
History -> Philosophers, Mathematicians, 
Phiolosophy -> Matemathicians
```