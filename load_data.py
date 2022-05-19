from datasets import load_dataset, DatasetDict

def load_bible_data(data_dir, train_languages, val_languages):
    train_files = []
    for lang in train_languages:
        train_files.append(f'{data_dir}/{lang}/train.csv')
    valid_files = []
    for lang in val_languages:
        valid_files.append(f'{data_dir}/{lang}/valid.csv')

    ds_train = load_dataset('csv', data_files=train_files)
    ds_valid = load_dataset('csv', data_files=valid_files)

    langs_dict = {}
    for lang in val_languages:
        test_d = load_dataset('csv', data_files=[f'{data_dir}/{lang}/test.csv'])
        langs_dict[lang] = test_d['train']
    langs_dict["train"] = ds_train['train']
    langs_dict["valid"] = ds_valid['train']

    raw_datasets = DatasetDict(langs_dict)
    return raw_datasets