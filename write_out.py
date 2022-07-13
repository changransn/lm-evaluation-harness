import argparse
import numpy as np
import json
import os
import random
from lm_eval import tasks
from lm_eval.utils import join_iters
from tqdm import tqdm
import logging
# Example usage
# python3 write_out.py     --tasks hellaswag,winogrande,lambada,piqa,triviaqa,arc_easy,arc_challenge,logiqa,openbookqa,race,anli_r1,anli_r2,anli_r3,cola,mnli,rte,boolq,wsc,arithmetic_2da --num_fewshot 0     --num_examples 100     --output_base_path ./results/zero_shot_elutherai_example_inputs_with_target/

import transformers
EXAMPLE_DIVIDER = "\n!!@@##@@!! -- Example {i}\n"
transformers.logging.set_verbosity_error()

import nltk
# We download the nltk tokenizer to the ./cache folder instead of the default home dir
nltk_cache_dir = './cache'
nltk.download('punkt', download_dir=nltk_cache_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--sets", type=str, default="val")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--description_dict_path", default=None)
    # parser.add_argument("--model_name_or_path", default=None)
    return parser.parse_args()


def main():

    args = parse_args()
    np.random.seed(args.seed)

    # For lambada model.generate()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # "/import/ml-sc-nlpcheckpoints-scratch/bol/sn_13b_150k_ckpt/"
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="/import/ml-sc-nlpcheckpoints-scratch/changranh/cache/")
    


    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    os.makedirs(args.output_base_path, exist_ok=True)
    for task_name, task in task_dict.items():
        print("Task Name")
        print(task_name)
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []

        for set in args.sets.split(","):
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
            iters.append(docs)

        docs = join_iters(iters)

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )

        result_acc = []
        with open(os.path.join(args.output_base_path, task_name), "w") as f:
            for i, doc in tqdm(
                zip(range(args.num_examples), docs)
                if args.num_examples > 0
                else enumerate(docs)
            ):
                f.write(EXAMPLE_DIVIDER.format(i=i))
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                    rnd=rnd,
                    description=description,
                )
                f.write("**Context**\n")
                f.write(ctx + "\n")
                tgt = task.doc_to_target(
                    doc=doc
                )
                f.write("**Target**\n")
                f.write(tgt + "\n")
                

                # if task_name == "lambada":
                #     # prompt = ctx
                #     # prompt = ''.join(nltk.tokenize.sent_tokenize(ctx))
                #     ctx = ctx.replace("* ","").replace(" *", "")
                #     ctx = ctx.replace("`", "'").replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")                    
                #     nltk_tokenized = nltk.tokenize.sent_tokenize(ctx)
                #     prompt = "".join([x[0].upper() + x[1:] for x in nltk_tokenized])
                #     # normalize target quotes as well
                #     tgt = tgt.replace("`", "'").replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")
                #     # prompt = "Predict the next word: " + prompt
                #     f.write('**Prompt with nltk segmentation**\n')
                #     f.write(prompt + '\n')

                #     f.write("**Target**\n")
                #     f.write(tgt + "\n")
                #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                #     outputs = model.generate(input_ids, do_sample=False, max_length=len(input_ids[0]) + 3)
                #     f.write("**Prediction**\n")
                #     prediction = tokenizer.batch_decode(outputs[:, len(input_ids[0]):], skip_special_tokens=True)[0]
                #     f.write(prediction + "\n")
                #     if len(prediction) >= len(tgt):
                #         if tgt == prediction[:len(tgt)]:
                #             f.write("Correct\n")
                #             result_acc.append(True)
                #         else:
                #             f.write("Incorrect\n")
                #             result_acc.append(False)
                #     else:
                #         if prediction == tgt[:len(prediction)]:
                #             f.write("**Target too long**\n")
                #         else:
                #             f.write("Incorrect\n")
                #             result_acc.append(False)                            


    
            # print(sum(result_acc) / len(result_acc), len(result_acc))
            # f.write(str(sum(result_acc) / len(result_acc)) + " " + str(len(result_acc)))
                    
                    # f.write(tokenizer.tokenize(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]))
                    

if __name__ == "__main__":

    main()
