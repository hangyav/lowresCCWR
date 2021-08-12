import torch
import argparse
from transformers import AutoModelForPreTraining, AutoTokenizer


def convert_state_dict(state_dict, prefix='bert'):
    state_dict_2 = {}

    for k, v in state_dict.items():
        if prefix in k:
            k = k.split(".")[1:]
            k = ".".join(k)

        state_dict_2[k] = v

    return state_dict_2


def get_tokenizer_model(base_model_type, state_dict):
    tokenizer = AutoTokenizer.from_pretrained(base_model_type)
    model = AutoModelForPreTraining.from_pretrained(base_model_type)

    if state_dict is not None:
        state_dict = torch.load(state_dict, map_location=model.device)['state_dict']

        #  prefix = base_model_type.split('-')[0]
        #  assert prefix == 'bert', 'Only BERT models were tested yet. Please verify other model types manually first!'
        #  state_dict = convert_state_dict(state_dict, prefix)

        model.load_state_dict(state_dict, strict=False)

    return tokenizer, model


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict', help='Model weights to load')
    parser.add_argument('--base_model_type', default='bert-base-multilingual-cased', help='Type of base model')
    parser.add_argument('--output', help='Folder to save to')
    args = parser.parse_args()

    tokenizer, model = get_tokenizer_model(args.base_model_type, args.state_dict)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
