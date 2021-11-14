# Training a POS tagger model with spaCy


Instructions to retrain it:
```
$ python -m spacy debug data config-v3.2.cfg --paths.train ./data/train.spacy --paths.dev ./data/valid.spacy
$ python -m spacy debug config config-v3.2.cfg --paths.train ./data/train.spacy --paths.dev ./data/valid.spacy
$ python -m spacy train config-v3.2.cfg --paths.train ./data/train.spacy --paths.dev ./data/valid.spacy
```

The dataset was originally extracted from the Menota catalogue.

