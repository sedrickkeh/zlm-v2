import argparse 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from langvec_model import MyGPT2LMHeadModel
import logging
from tqdm.auto import tqdm
logging.basicConfig(level=logging.INFO)

class MyDecoder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def do_decoding(self, args, c):
        tokenized = self.tokenizer(c, truncation=True, max_length=args.max_len, return_tensors='pt')['input_ids']
        if args.beam_search:
            generated_outs = self.model.generate(tokenized, max_length=args.max_len, pad_token_id=self.tokenizer.eos_token_id, no_repeat_ngram_size=args.no_repeat_ngram_size, 
                                num_beams=args.num_beams)[0]
        elif args.sampling_simple:
            generated_outs = self.model.generate(tokenized, max_length=args.max_len, pad_token_id=self.tokenizer.eos_token_id, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                temperature=args.temperature)[0]
        elif args.sampling_top_k:
            generated_outs = self.model.generate(tokenized, max_length=args.max_len, pad_token_id=self.tokenizer.eos_token_id, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                top_k=args.k_val)[0]
        elif args.sampling_top_p:
            generated_outs = self.model.generate(tokenized, max_length=args.max_len, pad_token_id=self.tokenizer.eos_token_id, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                top_p=args.p_val)[0]
        out = self.tokenizer.decode(generated_outs, truncation=True, max_length=args.max_len)
        print(out)
        return out    

def main(args):
    if args.use_langvecs:
        from utils import load_lang2vec, load_lang_ids
        lang2vec = load_lang2vec(args.lang2vec_dir)
        args.langvec_initial_dim = len(lang2vec['eng'])
        lang_ids = load_lang_ids(args.lang2vec_dir)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.max_len,
            pad_to_max_length=True,
            return_overflowing_tokens=True,
            return_length=True,
        )
        if args.use_langvecs:
            lang_vec_arr, lang_id_arr = [], []
            for l in element["lang"]:
                lang_vec_arr.append(lang2vec[l])
                lang_id_arr.append(lang_ids[l])
            for i in range(len(lang_vec_arr)):
                outputs["input_ids"][i].extend(lang_vec_arr[i])
                outputs["input_ids"][i].extend([lang_id_arr[i]])
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            input_batch.append(input_ids)
        return {"input_ids": input_batch}

    if args.use_langvecs:
        model = MyGPT2LMHeadModel.from_pretrained(args.model_path, args)
        model.transformer.set_knn_vec()
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)

    if args.start_chars=="alphabet":
        start_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    elif args.start_chars=="wordlist":
        import pandas as pd
        filename = f"{args.wordlist_path}/{args.lang}.csv"
        texts = pd.read_csv(filename).text.tolist()
        start_chars = []
        for t in texts:
            start_chars.extend(t.split('_'))
    else:
        assert(args.start_chars in ["alphabet", "wordlist"])

    d = MyDecoder(model, tokenizer)
    outputs = []
    for c in tqdm(start_chars):
        curr_output = d.do_decoding(args, c)
        outputs.append(curr_output)
    with open(args.output_path, 'w') as f:
        for o in outputs:
            f.write(o+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang2vec_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/src_new/uriel_embeddings_new.txt")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--wordlist_path", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/lexicons/csv_random/")

    # Generation starting characters
    parser.add_argument("--start_chars", type=str, default="alphabet", choices=["alphabet", "wordlist"])

    # Wordlists
    parser.add_argument("--use_wordlists", action="store_true")

    # Langvec related
    parser.add_argument("--use_langvecs", action='store_true')
    parser.add_argument("--langvec_dim", type=int, default=30)
    parser.add_argument("--projection_method", action='store_true',
                        help="If true, learns a projection. If false, learns embedding directly.")
    parser.add_argument("--random_langvecs", action='store_true')
    parser.add_argument("--average_langvecs", action='store_true')
    parser.add_argument("--average_knn", type=int, default=5)

    # Decoding related
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--beam_search", action='store_true')
    parser.add_argument("--sampling_simple", action='store_true')
    parser.add_argument("--sampling_top_p", action='store_true')
    parser.add_argument("--sampling_top_k", action='store_true')
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--p_val", type=float, default=0.9)
    parser.add_argument("--k_val", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    args = parser.parse_args()

    # Need exactly 1 way to specify langvec initialization
    if args.use_langvecs:
        assert((args.projection_method is not False) + (args.random_langvecs is not False) + (args.average_langvecs is not False) >= 1)
        if args.random_langvecs:
            assert((args.projection_method is not False) + (args.average_langvecs is not False) == 0)

    # Need exactly 1 way to do decoding
    assert(args.beam_search + args.sampling_simple + args.sampling_top_p + args.sampling_top_k == 1)

    main(args)