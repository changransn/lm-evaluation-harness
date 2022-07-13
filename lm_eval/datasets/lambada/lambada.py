# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""LAMBADA dataset."""


import json

import datasets


_CITATION = """\
@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}
"""

_DESCRIPTION = """\
LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
texts sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole text, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.
"""

_HOMEPAGE = "https://zenodo.org/record/2630551#.X4Xzn5NKjUI"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "original": "http://eaidata.bmk.sh/data/lambada_test.jsonl",
    "en": "http://eaidata.bmk.sh/data/lambada_test_en.jsonl",
    "fr": "http://eaidata.bmk.sh/data/lambada_test_fr.jsonl",
    "de": "http://eaidata.bmk.sh/data/lambada_test_de.jsonl",
    "it": "http://eaidata.bmk.sh/data/lambada_test_it.jsonl",
    "es": "http://eaidata.bmk.sh/data/lambada_test_es.jsonl",
}


class Lambada(datasets.GeneratorBasedBuilder):
    """LAMBADA is a dataset to evaluate the capabilities of computational models for text understanding by means of a word prediction task."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="original", version=VERSION, description="The LAMBADA dataset"
        ),
        datasets.BuilderConfig(
            name="en",
            version=VERSION,
            description="The English translated LAMBADA dataset",
        ),
        datasets.BuilderConfig(
            name="fr",
            version=VERSION,
            description="The French translated LAMBADA dataset",
        ),
        datasets.BuilderConfig(
            name="de",
            version=VERSION,
            description="The German translated LAMBADA dataset",
        ),
        datasets.BuilderConfig(
            name="it",
            version=VERSION,
            description="The Italian translated LAMBADA dataset",
        ),
        datasets.BuilderConfig(
            name="es",
            version=VERSION,
            description="The Spanish translated LAMBADA dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "original"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=f"{_DESCRIPTION}\n{self.config.description}",
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                # import nltk
                # data['text'] = data['text'].replace("* ","").replace(" *", "")
                # data['text'] = data['text'].replace("`", "'").replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")                    
                # nltk_tokenized = nltk.tokenize.sent_tokenize(data['text'])
                
                # # data['text'] = "".join([x[0].upper() + x[1:] for x in nltk_tokenized])
                # # data['text'] = " ".join([x[0].upper() + x[1:] for x in nltk_tokenized])
                # # data['text'] = " ".join([x for x in nltk_tokenized])
                # res = []
                # for x in nltk_tokenized:
                #     x = x.strip()
                #     if not x: # empty line
                #         continue
                #     if x[0] in ["'", "\""] and len(x) > 1:
                #         res.append(x[0] + x[1].upper() + x[2:])
                #     else:
                #         res.append(x[0].upper() + x[1:])
                
                # data["text"] = " ".join(res)
                # data["text"] = "<|endoftext|>" + data["text"]
                data['text'] = data_clean(data['text'])
                yield key, {"text": data["text"]}


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
        # # FIXIT: only for debugging!!
        res.append(x)
        # # FIXIT: only for debugging!!
        #            
        # if x[0] in ["'", "\""] and len(x) > 1:
        #     res.append(x[0] + x[1].upper() + x[2:])
        # else:
        #     res.append(x[0].upper() + x[1:])
    # text = " ".join(res)
    text = "".join(res)
    return text