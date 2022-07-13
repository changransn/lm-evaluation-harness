"""
RACE: Large-scale ReAding Comprehension Dataset From Examinations
https://arxiv.org/pdf/1704.04683.pdf

RACE is a large-scale reading comprehension dataset with more than 28,000 passages
and nearly 100,000 questions. The dataset is collected from English examinations
in China, which are designed for middle school and high school students. The dataset
can be served as the training and test sets for machine comprehension.

Homepage: https://www.cs.cmu.edu/~glai1/data/race/
"""
import collections
import datasets
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_CITATION = """
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""


class each:
    def __init__(self, f):
        self.f = f

    def __rrshift__(self, other):
        return list(map(self.f, other))


class RACE(Task):
    VERSION = 1
    DATASET_PATH = "race"
    DATASET_NAME = "high"

    cache = {}
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _collate_data(self, set):
        if set in self.cache:
            return self.cache[set]
        # One big issue with HF's implementation of this dataset: it makes a
        # separate document for each question; meanwhile, in the GPT3 paper it
        # is shown that one document is made per passage.

        r = collections.defaultdict(list)
        for item in datasets.load_dataset(
            path=self.DATASET_PATH, name=self.DATASET_NAME
        )[set]:
            r[item["article"]].append(item)

        res = list(
            r.values()
            >> each(
                lambda x: {
                    "article": x[0]["article"],
                    "problems": x
                    >> each(
                        lambda y: {
                            "question": y["question"],
                            "answer": y["answer"],
                            "options": y["options"],
                        }
                    ),
                }
            )
        )

        self.cache[set] = res
        return res

    def training_docs(self):
        return self._collate_data("train")

    def validation_docs(self):
        return self._collate_data("validation")

    def test_docs(self):
        return self._collate_data("test")

    @classmethod
    def get_answer_option(cls, problem):
        answer = cls.letter_to_num[problem["answer"]]
        answer = data_clean(problem["options"][answer])
        if not answer.endswith("."):
            return answer + "."
        return answer

    @classmethod
    def last_problem(cls, doc):
        return doc["problems"][-1]

    def doc_to_text(self, doc):
        # text = "Article: " + doc["article"] + "\n\n"
        doc["article"] = data_clean(doc["article"])
        text = "Article: " + add_period(doc["article"], '.') + " "

        # Replace all "  _  ." with "?"
        for idx in range(len(doc["problems"])):
            doc["problems"][idx]["question"] = doc["problems"][idx]["question"].replace("  _  .", "?")

        for problem in doc["problems"][:-1]:
            
                          
            problem["question"] = data_clean(problem["question"])
            # question = "Question: " + problem["question"] + "\n"
            # answer = "Answer: " + self.get_answer_option(problem) + "\n"
            question = "Question: " + add_period(problem["question"], "?") + " "
            answer = "Answer: " + add_period(self.get_answer_option(problem), '.') + " "                
            text += question + answer
        # text += self.last_problem(doc)["question"]
        text += "Question: " + add_period(self.last_problem(doc)["question"], '?') + " "
        text += "Answer:"

        # return text
        # print(doc)
        # print(text)
        return text

    # def doc_to_text(self, doc):
    #     text = "Article: " + doc["article"] + "\n\n"
    #     for problem in doc["problems"][:-1]:
    #         if problem["question"][-6:] == "  _  .":
    #             text += (
    #                 problem["question"][-5:] + self.get_answer_option(problem) + "\n"
    #             )
    #         else:
    #             question = "Question: " + problem["question"] + "\n"
    #             answer = "Answer: " + self.get_answer_option(problem) + "\n"
    #             text += question + answer
    #     text += self.last_problem(doc)["question"]
    #     return text

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["article"]

    def doc_to_target(self, doc):
        # return " " + self.get_answer_option(self.last_problem(doc))
        # print("doc_to_target")
        # print(" " + self.get_answer_option(self.last_problem(doc)))
        return " " + self.get_answer_option(self.last_problem(doc))

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        problem = self.last_problem(doc)
        ll_choices = [
            rf.loglikelihood(ctx, " " + problem["options"][i])[0] for i in range(4)
        ]
        return ll_choices

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = self.letter_to_num[self.last_problem(doc)["answer"]]
        pred = np.argmax(results)
        return {"acc": int(pred == gold)}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}

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
        
        # if x[0] in ["'", "\""] and len(x) > 1:
        #     res.append(x[0] + x[1].upper() + x[2:])
        # else:
        #     res.append(x[0].upper() + x[1:])

    
    # text = " ".join(res)
    text = "".join(res)
    print(res)

    # data["text"] = "<|endoftext|>" + data["text"]
    return text

def add_period(text, punct):
    return text + ("" if text.strip()[-1] in ['.', '?', ',', '!', "'", "\""] else punct)