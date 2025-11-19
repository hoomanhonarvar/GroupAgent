from langchain.tools import tool
from happytransformer import HappyTextToText, TTSettings
from nltk.corpus import wordnet as wn



save_path = "../../models/saved_t5_grammar"
happy_tt = HappyTextToText("T5", save_path)
args = TTSettings(num_beams=5, min_length=1)


@tool
def correct_grammar(text: str) -> str:
    """this tool for correcting grammar of sentence"""
    print("hello")
    result=happy_tt.generate_text(text, args=args).text
    print("result  ",result)
    return result

@tool 
def hooman_sentence() -> str:
    """ 
    this tool is a sentence creator that returns hooman sentences
    """
    return "baba chera nemifahmi????"

@tool
def print_hello() -> str:
    """
    this tool is just a test and does nothing
    """
    return "Hello"

@tool
def syn_ant(word: str):
    """
    this tool provides gives a word and returns a set of synonyms and a set of antonyms
    """
    synonyms=set()
    antonyms=set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name().replace('_', ' '))
    return list(synonyms), list(antonyms)
    