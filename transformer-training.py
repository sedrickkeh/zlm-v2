import argparse 
from transformers import GPT2Tokenizer, AutoConfig, GPT2LMHeadModel
from load_data import load_bible_data

blue = ["acu","alb","ceb","cjp","dik","eng","ewe","gla","hun","jak","kek","mam","nor","por","quw","slk","spa","tgl","vie","zul"]
green = ["ake","cak","chq","deu","dop","eus","gbi","hrv","ita","kbh","lit","nld","pol","quc","shi","som","swe","usp","wol","xho"]
red = ["afr","agr","amu","ces","cni","dje","epo","fin","glv","ind","jiv","lat","mri","pck","pot","rom","slv","srp","tmh","wal"]
yellow = ["bsn","cha","dan","djk","est","fra","hat","isl","kab","lav","nhg","plt","ppk","ron","sna","ssw","tur"]
langs = blue+green+red+yellow
langs_dict = {'blue':blue, 'green':green, 'red':red, 'yellow':yellow}

def main(args):
    raw_datasets = load_bible_data(args.data_dir, args.train_languages, args.val_languages)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=64,
            pad_to_max_length=True,
            return_overflowing_tokens=True,
            return_length=True,
        )
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
            n_ctx=64,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model = GPT2LMHeadModel(config)

    from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
    import logging
    logging.basicConfig(level=logging.INFO)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="transformer-zerolm",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=8,
        num_train_epochs=5,
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
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()

    for l in blue:
        print(l)
        print(trainer.evaluate(
            eval_dataset=tokenized_datasets[l]
        ))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/bibles_latin_csv/")
    parser.add_argument("--train_split", type=str, default=None,
                        help="red/blue/green/yellow group of languages")
    parser.add_argument("--train_language", type=str, default=None,
                        help="individual language")
    parser.add_argument("--train_file", type=str, default=None,
                        help="txt file with one language on each line")
    parser.add_argument("--val_split", type=str, default=None,
                        help="red/blue/green/yellow group of languages")
    parser.add_argument("--val_language", type=str, default=None,
                        help="individual language")
    parser.add_argument("--val_file", type=str, default=None,
                        help="txt file with one language on each line")
    args = parser.parse_args()

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
    assert((args.train_split is not None) + (args.train_language is not None) + (args.train_file is not None) <= 1)
    train_languages = []
    if args.train_split is not None:
        train_languages = langs_dict[args.train_split]
    elif args.train_language is not None:
        train_languages.append(args.train_language)
    elif args.train_file is not None:
        with open(args.train_file, 'r') as f:
            for line in f:
                train_languages.append(line.strip())
    else:
        for l in langs:
            if l not in val_languages:
                train_languages.append(l)
    args.train_languages = train_languages

    main(args)