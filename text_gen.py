import argparse
import numpy as np
import json
import os

import transformers
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
transformers.logging.set_verbosity_error()
# We download the nltk tokenizer to the ./cache folder instead of the default home dir

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# "/import/ml-sc-nlpcheckpoints-scratch/bol/sn_13b_150k_ckpt/"
model_name_or_path = "/import/ml-sc-nlpcheckpoints-scratch/bol/sn_13b_150k_ckpt/"
# model_name_or_path = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir="/import/ml-sc-nlpcheckpoints-scratch/changranh/cache/")
print("Model loaded! ", model_name_or_path)

while True:
    prompt = input("Please input a prompt:")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, do_sample=False, max_length=len(input_ids[0]) + 3)
    prediction = tokenizer.batch_decode(outputs[:, len(input_ids[0]):], skip_special_tokens=True)[0]
    print("Prediction: \n")
    print(prediction)
