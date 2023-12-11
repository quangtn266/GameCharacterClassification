from .base import ClassificationDataset
from utils.registry import Datasets

@Datasets.register_class()
class Legenddata(ClassificationDataset):
    mapper = {"Draven":0,
            "Evelynn":1,
            "Ezreal":2,
            "Fiora":3,
            "Fizz":4,
            "Garen":5,
            "Galio":6,
            "Gragas":7,
            "Graves":8,
            "Janna":9,
            "Jarvan_IV":10,
            "Jax":11,
            "Jhin":12,
            "Jinx":13,
            "Katarina":14,
            "Kennen":15,
            "Leona":16,
            "Lee_Sin":17,
            "Lulu":18,
            "Lux":19,
            "Malphite":20,
            "Master_Yi":21,
            "Miss_Fortune": 22,
            "Nami": 23,
            "Nasus": 24,
            "Olaf": 25,
            "Orianna": 26,
            "Pantheon": 27,
            "Rakan": 28,
            "Rammus": 29,
            "Rengar": 30,
            "Seraphine": 31,
            "Shyvana": 32,
            "Singed": 33,
            "Sona": 34,
            "Soraka": 35,
            "Teemo": 36,
            "Tristana": 37,
            "Tryndamere": 38,
            "Twisted_Fate": 39,
            "Varus": 40,
            "Vayne": 41,
            "Vi": 42,
            "Xin_Zhao": 43,
            "Yasuo": 44,
            "Wukong": 45,
            "Zed": 46,
            "Ziggs": 47,
            "Dr._Mundo": 48,
            "Ahri": 49,
            "Akali": 50,
            "Alistar": 51,
            "Amumu": 52,
            "Annie": 53,
            "Ashe": 54,
            "Aurelion_Sol": 55,
            "Blitzcrank": 56,
            "Braum": 57,
            "Camille": 58,
            "Corki": 59,
            "Darius": 60,
            "Diana": 61,
            "KaiSa": 62,
            "KhaZix": 63,
              }

    labels = {0:"Draven",1:"Evelynn",2:"Ezreal",3:"Fiora",4:"Fizz",5:"Garen",6:"Galio",
              7:"Gragas",8:"Graves",9:"Janna",10:"Jarvan_IV",11:"Jax",12:"Jhin",13:"Jinx",
              14:"Katarina",15:"Kennen",16:"Leona",17:"Lee_Sin",18:"Lulu",19:"Lux",20:"Malphite",
              21:"Master_Yi",22:"Miss_Fortune",23:"Nami",24:"Nasus",25:"Olaf",26:"Orianna",27:"Pantheon",
              28:"Rakan",29:"Rammus",30:"Rengar",31:"Seraphine",32:"Shyvana",33:"Singed",34:"Sona",35:"Soraka",
              36:"Teemo",37:"Tristana",38:"Tryndamere",39:"Twisted_Fate",40:"Varus",41:"Vayne",42:"Vi",43:"Xin_Zhao",
              44:"Yasuo",45:"Wukong",46:"Zed",47:"Ziggs",48:"Dr._Mundo",49:"Ahri",50:"Akali",51:"Alistar",52:"Amumu",
              53:"Annie",54:"Ashe",55:"Aurelion_Sol",56:"Blitzcrank",57:"Braum",58:"Camille",59:"Corki",60:"Darius",
              61:"Diana",62:"KaiSa",63:"KhaZix"
              }