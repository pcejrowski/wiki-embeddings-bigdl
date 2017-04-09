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
