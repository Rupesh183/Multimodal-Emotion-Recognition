from transformers import AutoModelForSequenceClassification, AutoTokenizer

BERT_MODEL_NAME = "google-bert/bert-base-uncased"


def Bert(num_classes: int):
    return AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_classes
    )


def BertTokenizer():
    return AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
