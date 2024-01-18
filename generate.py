# import os
# import argparse
# from dotenv import load_dotenv
# from toxigen.language_models import GPT3, ALICE
# from toxigen.pretrained_classifiers import HateBERT, ToxDectRoBERTa

# load_dotenv()

# def main():

#     # # Set up default values from environment variables
#     # default_language_model = os.getenv('LANGUAGE_MODEL')
#     # default_num_generations_per_prompt = int(os.getenv('NUM_GENERATIONS_PER_PROMPT', 1))
#     # default_endpoint_url = os.getenv('ENDPOINT_URL')
#     # default_api_key = os.getenv('API_KEY')
#     # default_classifier = os.getenv('CLASSIFIER')
#     # default_generation_mode = os.getenv('GENERATION_MODE')
#     # default_alice = os.getenv('ALICE', 'False').lower() in ('true', '1', 't')
#     # default_input_prompt_file = os.getenv('INPUT_PROMPT_FILE')
#     # default_output_file = os.getenv('OUTPUT_FILE')

#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--input_prompt_file", type=str, default=default_input_prompt_file)
#     # parser.add_argument("--output_file", type=str, default=default_output_file)
#     # parser.add_argument("--language_model", type=str, default=default_language_model)
#     # parser.add_argument("--num_generations_per_prompt", type=int, default=default_num_generations_per_prompt)
#     # parser.add_argument("--endpoint_url", type=str, default=default_endpoint_url)
#     # parser.add_argument("--api_key", type=str, default=default_api_key)
#     # parser.add_argument("--classifier", type=str, default=default_classifier)
#     # parser.add_argument("--generation_mode", type=str, default=default_generation_mode)
#     # parser.add_argument("--ALICE", type=bool, default=default_alice)

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_prompt_file", type=str)
#     parser.add_argument("--language_model", type=str)
#     parser.add_argument("--output_file", type=str)
#     parser.add_argument("--num_generations_per_prompt", type=int)
#     parser.add_argument("--endpoint_url", type=str)
#     parser.add_argument("--api_key", type=str)
#     parser.add_argument("--classifier", type=str)
#     parser.add_argument("--generation_mode", type=str)  # Can either be "neutral" or "hate", and indicates what you want to generate
#     parser.add_argument("--ALICE", default=False, type=bool)

#     args = parser.parse_args()

#      # --- load prompts ---
#     prompts = [l.strip() for l in open(args.input_prompt_file).readlines()]

#     # --- initialize language model ---
#     if args.language_model == "GPT3":
#         language_model = GPT3(endpoint_url=args.endpoint_url, apikey=args.api_key)
#     elif args.language_model == "GPT2":
#         language_model = GPT2()
#     else:
#         raise ValueError

#     if args.ALICE:
#         # --- initialize pretrained toxicity classifier ---
#         if args.classifier == "HateBERT":
#             classifier = HateBERT("<path_to_hatebert_files>")
#         elif args.classifier == "RoBERTa":
#             classifier = ToxDectRoBERTa()
#         else:
#             raise NotImplementedError

#         # --- wrap language model and toxicity detector in ALICE ---
#         language_model = ALICE(language_model, classifier, args.generation_mode)

#     # --- loop through prompts and generate responses ---
#     for prompt in prompts:
#         for i in range(args.num_generations_per_prompt):
#             response = language_model(prompt)
#             print(response)
#             with open(args.output_file, "a") as f:
#                 f.write(response + "\n")

    

#             # try:
#             #     response = language_model(prompt)
#             #     # Only write to file if output file is a string
#             #     if isinstance(response, str):
#             #         with open(args.output_file, "a") as f:
#             #             f.write(response + "\n")
#             #     else:
#             #         print(f"Error occurred: {response}")
#             # except Exception as e:
#             #     print(f"An exception occurred: {e}")


# if __name__ == "__main__":
#     main()

import os
from toxigen.language_models import GPT3, ALICE
from toxigen.pretrained_classifiers import HateBERT, ToxDectRoBERTa
import argparse

def main():


    # Set up default values from environment variables
    default_language_model = os.getenv('LANGUAGE_MODEL')
    default_num_generations_per_prompt = int(os.getenv('NUM_GENERATIONS_PER_PROMPT', 1))
    default_endpoint_url = os.getenv('ENDPOINT_URL')
    default_api_key = os.getenv('API_KEY')
    default_classifier = os.getenv('CLASSIFIER')
    default_generation_mode = os.getenv('GENERATION_MODE')
    default_alice = os.getenv('ALICE', 'False').lower() in ('true', '1', 't')
    default_input_prompt_file = os.getenv('INPUT_PROMPT_FILE')
    default_output_file = os.getenv('OUTPUT_FILE')
    default_max_tokens = int(os.getenv('MAX_TOKENS', 20))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prompt_file", type=str, default=default_input_prompt_file)
    parser.add_argument("--language_model", type=str, default=default_language_model)
    parser.add_argument("--output_file", type=str, default=default_output_file)
    parser.add_argument("--num_generations_per_prompt", type=int, default=default_num_generations_per_prompt)
    parser.add_argument("--endpoint_url", type=str, default=default_endpoint_url)
    parser.add_argument("--api_key", type=str, default=default_api_key)
    parser.add_argument("--classifier", type=str, default=default_classifier)
    parser.add_argument("--generation_mode", type=str, default=default_generation_mode) # Can either be "neutral" or "hate", and indicates what you want to generate
    parser.add_argument("--ALICE", default=default_alice, type=bool)
    parser.add_argument("--max_tokens", default=default_max_tokens, type=int, help="Number of tokens to generate per prompt")

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
        elif args.classifier == "RoBERTa":
            classifier = ToxDectRoBERTa()
        else:
            raise NotImplementedError

        # --- wrap language model and toxicity detector in ALICE ---
        language_model = ALICE(language_model, classifier, args.generation_mode)

    # --- loop through prompts and generate responses ---
    for prompt in prompts:
        for i in range(args.num_generations_per_prompt):
            response = language_model.generate(prompt, max_tokens=args.max_tokens)
            print("START" + response + " END")
            with open(args.output_file, "a") as f:
                f.write(response + "\n")

if __name__ == "__main__":
    main()