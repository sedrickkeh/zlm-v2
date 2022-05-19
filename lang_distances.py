import numpy as np
import lang2vec.lang2vec as l2v

class NN_Extractor:
    def __init__(self):
        blue = ["acu","alb","ceb","cjp","dik","eng","ewe","gla","hun","jak","kek","mam","nor","por","quw","slk","spa","tgl","vie","zul"]
        green = ["ake","cak","chq","deu","dop","eus","gbi","hrv","ita","kbh","lit","nld","pol","quc","shi","som","swe","usp","wol","xho"]
        red = ["afr","agr","amu","ces","cni","dje","epo","fin","glv","ind","jiv","lat","mri","pck","pot","rom","slv","srp","tmh","wal"]
        yellow = ["bsn","cha","dan","djk","est","fra","hat","isl","kab","lav","nhg","plt","ppk","ron","sna","ssw","tur"]
        self.langs = blue+green+red+yellow
        self.geo = l2v.get_features(self.langs, "geo")

    def by_geography(self, lang, k=76):
        dists = {}
        for l in self.langs:
            if l==lang: continue
            dists[l] = np.linalg.norm(np.array(self.geo[l]) - np.array(self.geo[lang]))
        dists_arr = []
        for l in dists:
            dists_arr.append((dists[l], l))
        dists_arr.sort()
        dists_arr = [i[1] for i in dists_arr]
        return dists_arr[:k]