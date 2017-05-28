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