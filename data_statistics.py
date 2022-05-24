import pandas as pd

class DataStatistics:
    def __init__(self, data_dir='/content/drive/MyDrive/Colab Notebooks/zero-shot-lm/bibles_latin_csv/'):
        self.data_dir = data_dir
        blue = ["acu","alb","ceb","cjp","dik","eng","ewe","gla","hun","jak","kek","mam","nor","por","quw","slk","spa","tgl","vie","zul"]
        green = ["ake","cak","chq","deu","dop","eus","gbi","hrv","ita","kbh","lit","nld","pol","quc","shi","som","swe","usp","wol","xho"]
        red = ["afr","agr","amu","ces","cni","dje","epo","fin","glv","ind","jiv","lat","mri","pck","pot","rom","slv","srp","tmh","wal"]
        yellow = ["bsn","cha","dan","djk","est","fra","hat","isl","kab","lav","nhg","plt","ppk","ron","sna","ssw","tur"]
        self.langs = blue+green+red+yellow
        self.init_dict()
        self.init_counts(data_dir)

    def init_counts(self, data_dir):
        self.counts = {}
        for lang in self.langs:
            df = pd.read_csv(f"{data_dir}/{lang}/train.csv")
            self.counts[lang] = len(df)

    def get_counts(self, arr):
        cnt = 0
        arr = list(arr)
        for lang in arr:
            cnt += self.counts[lang]
        return cnt

    def init_dict(self):
        self.langdict = {}
        self.langdict["acu"] = ("Achuar-Shiwiar", "Jivaroan")
        self.langdict["alb"] = ("Albanian", "Indo-European")
        self.langdict["ceb"] = ("Cebuano", "Austronesian")
        self.langdict["cjp"] = ("Cabécar", "Chibchan")
        self.langdict["dik"] = ("Dinka", "Eastern Sudanic")
        self.langdict["eng"] = ("English", "Indo-European")
        self.langdict["ewe"] = ("Ewe", "Niger-Congo")
        self.langdict["gla"] = ("Gaelic", "Indo-European")
        self.langdict["hun"] = ("Hungarian", "Uralic")
        self.langdict["jak"] = ("Jakaltek", "Mayan")
        self.langdict["kek"] = ("K'ekchí", "Mayan")
        self.langdict["mam"] = ("Mam", "Mayan")
        self.langdict["nor"] = ("Norwegian", "Indo-European")
        self.langdict["por"] = ("Portuguese", "Indo-European")
        self.langdict["quw"] = ("Quechua", "Quechuan")
        self.langdict["slk"] = ("Slovak", "Indo-European")
        self.langdict["spa"] = ("Spanish", "Indo-European")
        self.langdict["tgl"] = ("Tagalog", "Austronesian")
        self.langdict["vie"] = ("Vietnamese", "Austro-Asiatic")
        self.langdict["zul"] = ("Zulu", "Niger-Congo")

        self.langdict["ake"] = ("Akawaio", "Cariban")
        self.langdict["cak"] = ("Cakchiquel", "Mayan")
        self.langdict["chq"] = ("Chinantec", "Oto-Manguean")
        self.langdict["deu"] = ("German", "Indo-European")
        self.langdict["dop"] = ("Lukpa", "Niger-Congo")
        self.langdict["eus"] = ("Basque", "Basque")
        self.langdict["gbi"] = ("Galela", "West Papuan")
        self.langdict["hrv"] = ("Serbian-Croatian", "Indo-European")
        self.langdict["ita"] = ("Italian", "Indo-European")
        self.langdict["kbh"] = ("Camsá", "Camsá")
        self.langdict["lit"] = ("Lithuanian", "Indo-European")
        self.langdict["nld"] = ("Dutch", "Indo-European")
        self.langdict["pol"] = ("Polish", "Indo-European")
        self.langdict["quc"] = ("Quiché", "Mayan")
        self.langdict["shi"] = ("Tashlhiyt", "Afro-Asiatic")
        self.langdict["som"] = ("Somali", "Afro-Asiatic")
        self.langdict["swe"] = ("Swedish", "Indo-European")
        self.langdict["usp"] = ("Uspanteco", "Mayan")
        self.langdict["wol"] = ("Wolof", "Niger-Congo")
        self.langdict["xho"] = ("Xhosa", "Niger-Congo")

        self.langdict["afr"] = ("Afrikaans", "Indo-European")
        self.langdict["agr"] = ("Aguaruna", "Jivaroan")
        self.langdict["amu"] = ("Amuzgo", "Oto-Manguean")
        self.langdict["ces"] = ("Czech", "Indo-European")
        self.langdict["cni"] = ("Asháninka", "Arawakan")
        self.langdict["dje"] = ("Zarma", "Songhay")
        self.langdict["epo"] = ("Esperanto", "Indo-European")
        self.langdict["fin"] = ("Finnish", "Uralic")
        self.langdict["glv"] = ("Manx", "Indo-European")
        self.langdict["ind"] = ("Indonesian", "Austronesian")
        self.langdict["jiv"] = ("Jivaro", "Jivaroan")
        self.langdict["lat"] = ("Latin", "Indo-European")
        self.langdict["mri"] = ("Maori", "Austronesian")
        self.langdict["pck"] = ("Paite", "Sino-Tibetan")
        self.langdict["pot"] = ("Potawatomi", "Algic")
        self.langdict["rom"] = ("Romani", "Indo-European")
        self.langdict["slv"] = ("Slovene", "Indo-European")
        self.langdict["srp"] = ("Serbian", "Indo-European")
        self.langdict["tmh"] = ("Tamashek", "Afro-Asiatic")
        self.langdict["wal"] = ("Wolaytta", "Afro-Asiatic")

        self.langdict["bsn"] = ("Barasano", "Tucanoan")
        self.langdict["cha"] = ("Chamorro", "Austronesian")
        self.langdict["dan"] = ("Danish", "Indo-European")
        self.langdict["djk"] = ("Ndyuka", "other")
        self.langdict["est"] = ("Estonian", "Uralic")
        self.langdict["fra"] = ("French", "Indo-European")
        self.langdict["hat"] = ("Haitian Creole", "other")
        self.langdict["isl"] = ("Icelandic", "Indo-European")
        self.langdict["kab"] = ("Kabyle", "Afro-Asiatic")
        self.langdict["lav"] = ("Latvian", "Indo-European")
        self.langdict["nhg"] = ("Nahuatl", "Uto-Aztecan")
        self.langdict["plt"] = ("Malagasy", "Austronesian")
        self.langdict["ppk"] = ("Uma", "Austronesian")
        self.langdict["ron"] = ("Romanian", "Indo-European")
        self.langdict["sna"] = ("Shona", "Niger-Congo")
        self.langdict["ssw"] = ("Swati", "Niger-Congo")
        self.langdict["tur"] = ("Turkish", "Altaic")