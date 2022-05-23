import argparse 
from transformers import GPT2Tokenizer, AutoConfig, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from langvec_model import MyGPT2LMHeadModel
from utils import load_lang2vec
from load_data import load_bible_data
from data_statistics import DataStatistics
from lang_distances import NN_Extractor
from ideal_vec_trainer import IdealVecTrainer
import logging
logging.basicConfig(level=logging.INFO)

blue = ["acu","alb","ceb","cjp","dik","eng","ewe","gla","hun","jak","kek","mam","nor","por","quw","slk","spa","tgl","vie","zul"]
green = ["ake","cak","chq","deu","dop","eus","gbi","hrv","ita","kbh","lit","nld","pol","quc","shi","som","swe","usp","wol","xho"]
red = ["afr","agr","amu","ces","cni","dje","epo","fin","glv","ind","jiv","lat","mri","pck","pot","rom","slv","srp","tmh","wal"]
yellow = ["bsn","cha","dan","djk","est","fra","hat","isl","kab","lav","nhg","plt","ppk","ron","sna","ssw","tur"]
langs = blue+green+red+yellow
langs_dict = {'blue':blue, 'green':green, 'red':red, 'yellow':yellow}

def main(args):
    lang2vec = load_lang2vec(args.lang2vec_dir)
    raw_datasets = load_bible_data(args.data_dir, args.train_languages, args.val_languages)

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
            langarr = []
            for l in element["lang"]:
                langarr.append(lang2vec[l])
            for i in range(len(langarr)):
                outputs["input_ids"][i].extend(langarr[i])
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=args.max_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if args.use_langvecs:
        model = MyGPT2LMHeadModel(config, args)
    else:
        model = GPT2LMHeadModel(config)
    if args.model_dir is not None:
        if args.use_langvecs:
            model = model.from_pretrained(args.model_dir, args)
        else:
            model = model.from_pretrained(args.model_dir)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size//2,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        fp16=True,
        logging_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = IdealVecTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train(args.freeze_except_langvec)

    for lang in args.val_languages:
        print(lang)
        print(trainer.evaluate(
            eval_dataset=tokenized_datasets[lang]
        ))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/bibles_latin_csv/")
    parser.add_argument("--lang2vec_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/src_new/uriel_embeddings_new.txt")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="transformer-zerolm")
    parser.add_argument("--train_split", type=str, default=None,
                        help="red/blue/green/yellow group of languages")
    parser.add_argument("--train_language", type=str, default=None,
                        help="individual language")
    parser.add_argument("--train_file", type=str, default=None,
                        help="txt file with one language on each line")
    parser.add_argument("--train_by_langvec", type=str, choices=["geo", "fam"], default=None,
                        help="method of selecting which languages to train on")
    parser.add_argument("--train_knn", type=int, default=5)
    parser.add_argument("--val_split", type=str, default=None,
                        help="red/blue/green/yellow group of languages")
    parser.add_argument("--val_language", type=str, default=None,
                        help="individual language")
    parser.add_argument("--val_file", type=str, default=None,
                        help="txt file with one language on each line")
    parser.add_argument("--include_val_in_train", action='store_true')

    parser.add_argument("--use_langvecs", action='store_true')
    parser.add_argument("--langvec_dim", type=int, default=30)
    parser.add_argument("--freeze_except_langvec", action='store_true')

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    # Probably don't need to change these
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=64)

    args = parser.parse_args()

    # Check asserts
    if args.freeze_except_langvec:
        assert(args.model_dir is not None)

    # Need exactly 1 way to specify val language
    assert((args.val_split is not None) + (args.val_language is not None) + (args.val_file is not None) == 1)
    val_languages = []
    if args.val_split is not None:
        val_languages = langs_dict[args.val_split]
    elif args.val_language is not None:
        val_languages.append(args.val_language)
    elif args.val_file is not None:
        with open(args.val_file, 'r') as f:
            for line in f:
                val_languages.append(line.strip())
    args.val_languages = val_languages

    # Need at most 1 way to specify train language
    assert((args.train_split is not None) + (args.train_language is not None) + (args.train_file is not None) + (args.train_by_langvec is not None) <= 1)
    train_languages = []
    if args.train_split is not None:
        train_languages = langs_dict[args.train_split]
    elif args.train_language is not None:
        train_languages.append(args.train_language)
    elif args.train_file is not None:
        with open(args.train_file, 'r') as f:
            for line in f:
                train_languages.append(line.strip())
    elif args.train_by_langvec is not None:
        assert(args.val_language is not None)
        langvec_extractor = NN_Extractor()
        if args.train_by_langvec=="geo":
            train_languages = langvec_extractor.by_geography(args.val_language, k=args.train_knn)
        elif args.train_by_langvec=="fam":
            ds = DataStatistics(args.data_dir)
            current_family = ds.langdict[args.val_language][1]
            for lang in ds.langdict:
                if lang==args.val_language: continue
                if ds.langdict[lang][1]==current_family:
                    train_languages.append(lang)
        else:
            assert(args.train_by_langvec in ["geo", "fam"])
    else:
        for l in langs:
            if l not in val_languages:
                train_languages.append(l)
    if args.include_val_in_train:
        train_languages.extend(val_languages)
    args.train_languages = train_languages

    ds = DataStatistics(args.data_dir)
    print("Train languages: ", ds.get_counts(args.train_languages), args.train_languages)
    print("Test languages: ", ds.get_counts(args.val_languages), args.val_languages)
    main(args)