import torch
from src.language_models import GPT3, GPT2, ALICE
from src.pretrained_classifiers import HateBERT, ToxDectRoBERTa
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prompt_file", type=str)
    parser.add_argument("--language_model", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_generations_per_prompt", type=int)
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--classifier", type=str)
    parser.add_argument("--ALICE", default=False, type=bool)

    args = parser.parse_args()

    # --- load prompts ---
    prompts = [l.strip() for l in open(args.input_prompt_file).readlines()]

    # --- initialize language model ---
    if args.language_model == "GPT3":
        language_model = GPT3(endpoint_url=args.endpoint_url, apikey=args.api_key)
    elif args.language_model == "GPT2":
        language_model = GPT2()
    else:
        raise ValueError

    if args.ALICE:
        # --- initialize pretrained toxicity classifier ---
        if args.classifier == "HateBERT":
            classifier = HateBERT("<path_to_hatebert_files>")
        elif args.classifier == "ToxDectRoBERTa":
            classifier = ToxDectRoBERTa()
        else:
            raise NotImplementedError

        # --- wrap language model and toxicity detector in ALICE ---
        language_model = ALICE(language_model, classifier)

    # --- loop through prompts and generate responses ---
    for prompt in prompts:
        for i in range(args.num_generations_per_prompt):
            response = language_model(prompt)
            print(response)
            with open(args.output_file, "a") as f:
                f.write(response + "\n")
            assert 2 == 3

if __name__ == "__main__":
    main()
