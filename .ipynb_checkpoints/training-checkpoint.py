import codecs
import json
import os
import random


import spacy
from spacy.tokens import DocBin, Doc
from spacy.training import Example


def read_menota_annotations_for_spacy():
    l = []
    with codecs.open(os.path.join("data-menota-spacy.json"), "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            l.append(line)
    return l


def create_vocab(training_data):
    vocabulary_set = set()
    for text, _ in training_data:
        for lemma in text:
            vocabulary_set.add(lemma)
    return vocabulary_set


def create_training(training_data):
    vocab = spacy.Vocab(string=create_vocab(training_data))
    nlp = spacy.blank("is")
    db = DocBin()  # create a DocBin object
    for text, annotation in training_data:
        example = Example.from_dict(Doc(vocab, words=text, ), annotation)
        db.add(example.reference)
    return db


def save_training_data(training_data):
    random.shuffle(training_data)
    valid = training_data[:500]
    train = training_data[500:]

    train = create_training(train)
    train.to_disk("./data/train.spacy")

    valid = create_training(valid)
    valid.to_disk("./data/valid.spacy")


def reduce_tags(pos_tag: str) -> str:
    first_part = pos_tag.split(" ")[0]
    if first_part.startswith("x"):
        return first_part[1:]
    else:
        return pos_tag


def from_text_annotations_to_spacy_training_data(data):
    l = []
    for doc in data:
        ll = []
        for sentence, annotation in doc:
            try:
                ll.append(([token.lower() for token in sentence],
                           dict(pos=[reduce_tags(tag) for tag in annotation["pos"]])))
                # lemmas=annotation["lemmas"]
            except AttributeError:
                print(sentence)
                print(annotation)
        l.extend(ll)
    return l


if __name__ == "__main__":
    # data = read_menota_annotations()
    data = read_menota_annotations_for_spacy()
    # print(data[0].items())
    pos_training_data = from_text_annotations_to_spacy_training_data(data)
    # save_training_data(pos_training_data)
    # vocab = create_vocab(pos_training_data)
    # print(len(vocab))

    # print(pos_training_data[0:2])
    print(len(pos_training_data))
    print(pos_training_data[0])
    save_training_data(pos_training_data)
