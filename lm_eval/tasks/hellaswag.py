"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf

Hellaswag is a commonsense inference challenge dataset. Though its questions are
trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). This is
achieved via Adversarial Filtering (AF), a data collection paradigm wherein a
series of discriminators iteratively select an adversarial set of machine-generated
wrong answers. AF proves to be surprisingly robust. The key insight is to scale up
the length and complexity of the dataset examples towards a critical 'Goldilocks'
zone wherein generated text is ridiculous to humans, yet often misclassified by
state-of-the-art models.

Homepage: https://rowanzellers.com/hellaswag/
"""
import re
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""


class HellaSwag(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "hellaswag"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        # ctx = doc["ctx_a"] + " " + doc["ctx_b"]
        out_doc = {
            "query": data_clean(self.preprocess(doc["activity_label"] + ": " + ctx)),
            "choices": [data_clean(self.preprocess(ending)) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }        
        # ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        # out_doc = {
        #     "query": self.preprocess(doc["activity_label"] + ": " + ctx),
        #     "choices": [self.preprocess(ending) for ending in doc["endings"]],
        #     "gold": int(doc["label"]),
        # }
        return out_doc

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def doc_to_text(self, doc):
        return data_clean(doc["query"])

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

def data_clean(text):
    import nltk
    text = text.replace("* ","").replace(" *", "")
    text = text.replace("`", "'").replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")
    nltk_tokenized = nltk.tokenize.sent_tokenize(text)

    res = []
    for x in nltk_tokenized:
        x = x.strip()
        if not x: # empty line
            continue
        # FIXIT: only for debugging!!
        res.append(x)
        # FIXIT: only for debugging!!
        #            
        # if x[0] in ["'", "\""] and len(x) > 1:
        #     res.append(x[0] + x[1].upper() + x[2:])
        # else:
        #     res.append(x[0].upper() + x[1:])
    
    text = " ".join(res)
    # text = "".join(res)
    # print(res)
    # data["text"] = "<|endoftext|>" + data["text"]
    return text

def add_period(text, punct):
    return text + ("" if text.strip()[-1] in ['.', '?', ',', '!', "'", "\""] else punct)