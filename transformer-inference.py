import argparse 
from transformers import GPT2Tokenizer, AutoConfig, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from load_data import load_bible_data
from langvec_model import MyGPT2LMHeadModel
import logging
logging.basicConfig(level=logging.INFO)

blue = ["acu","alb","ceb","cjp","dik","eng","ewe","gla","hun","jak","kek","mam","nor","por","quw","slk","spa","tgl","vie","zul"]
green = ["ake","cak","chq","deu","dop","eus","gbi","hrv","ita","kbh","lit","nld","pol","quc","shi","som","swe","usp","wol","xho"]
red = ["afr","agr","amu","ces","cni","dje","epo","fin","glv","ind","jiv","lat","mri","pck","pot","rom","slv","srp","tmh","wal"]
yellow = ["bsn","cha","dan","djk","est","fra","hat","isl","kab","lav","nhg","plt","ppk","ron","sna","ssw","tur"]
langs = blue+green+red+yellow
splits_dict = {'blue':blue, 'green':green, 'red':red, 'yellow':yellow}

def main(args):
    if args.use_langvecs:
        from utils import load_lang2vec, load_lang_ids
        lang2vec = load_lang2vec(args.lang2vec_dir)
        args.langvec_initial_dim = len(lang2vec['eng'])
        lang_ids = load_lang_ids(args.lang2vec_dir)
    raw_datasets = load_bible_data(args)

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

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["valid"].column_names
    )

    config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=64,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if args.use_langvecs:
        model = MyGPT2LMHeadModel(config, args)
        model = model.from_pretrained(args.model_path, args)
    else:
        model = GPT2LMHeadModel(config)
        model = model.from_pretrained(args.model_path)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="transformer-zerolm",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size//2,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        fp16=True,
        logging_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["valid"],
        eval_dataset=tokenized_datasets["valid"],
    )

    for lang in args.test_languages:
        print(lang)
        print(trainer.evaluate(
            eval_dataset=tokenized_datasets[lang]
        ))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/bibles_latin_csv/")
    parser.add_argument("--lang2vec_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/src_new/uriel_embeddings_new.txt")
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--test_split", type=str, default=None,
                        help="red/blue/green/yellow group of languages")
    parser.add_argument("--test_language", type=str, default=None,
                        help="individual language")
    parser.add_argument("--test_file", type=str, default=None,
                        help="txt file with one language on each line")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)

    # Wordlists
    parser.add_argument("--use_wordlists", action="store_true")

    # Langvec related
    parser.add_argument("--use_langvecs", action='store_true')
    parser.add_argument("--langvec_dim", type=int, default=30)
    parser.add_argument("--projection_method", action='store_true',
                        help="If true, learns a projection. If false, learns embedding directly.")

    args = parser.parse_args()

    # Need exactly 1 way to specify val language
    assert((args.test_split is not None) + (args.test_language is not None) + (args.test_file is not None) == 1)
    test_languages = []
    if args.test_split is not None:
        test_languages = splits_dict[args.test_split]
    elif args.test_language is not None:
        test_languages.append(args.test_language)
    elif args.test_file is not None:
        with open(args.test_file, 'r') as f:
            for line in f:
                test_languages.append(line.strip())
    args.test_languages = test_languages
    args.train_languages = []

    main(args)