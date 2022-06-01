def load_lang2vec(langvec_dir="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/src_new/uriel_embeddings_new.txt"):
    lang2vec = {}
    with open(langvec_dir) as f:
        for line in f:
            langsplit = line.strip().split('\t')
            lang, vec = langsplit[0], langsplit[1:]
            vec = [int(float(i)) for i in vec]
            lang2vec[lang] = vec
    return lang2vec
    
def load_lang_ids(langvec_dir="/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/src_new/uriel_embeddings_new.txt"):
    lang_ids = {}
    with open(langvec_dir) as f:
        idx = 0
        for line in f:
            lang = line.split('\t')[0]
            lang_ids[lang] = idx
            idx+=1
    return lang_ids