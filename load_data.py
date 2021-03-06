from datasets import load_dataset, DatasetDict

def load_bible_data(args):
    data_dir = args.data_dir
    train_languages = args.train_languages
    if "val_languages" in args: 
        val_languages = args.val_languages
    else: 
        val_languages = args.test_languages

    train_files = []
    for lang in train_languages:
        train_files.append(f'{data_dir}/{lang}/train.csv')
    valid_files = []
    for lang in val_languages:
        valid_files.append(f'{data_dir}/{lang}/valid.csv')
    if "use_wordlists" in args:
        if args.use_wordlists:
            for lang in val_languages:
                train_files.append(f'{args.wordlist_dir}/{lang}.csv')

    langs_dict = {}
    if len(train_files)>0:      # 0 during inference
        ds_train = load_dataset('csv', data_files=train_files)
        langs_dict["train"] = ds_train['train']
    ds_valid = load_dataset('csv', data_files=valid_files)
    langs_dict["valid"] = ds_valid['train']

    for lang in val_languages:
        test_d = load_dataset('csv', data_files=[f'{data_dir}/{lang}/test.csv'])
        langs_dict[lang] = test_d['train']

    raw_datasets = DatasetDict(langs_dict)
    return raw_datasets