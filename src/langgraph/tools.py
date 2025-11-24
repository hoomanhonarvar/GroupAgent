from langchain.tools import tool
from happytransformer import HappyTextToText, TTSettings
from nltk.corpus import wordnet as wn
from pydantic import BaseModel,Field


save_path = "../../models/saved_t5_grammar"
happy_tt = HappyTextToText("T5", save_path)
args = TTSettings(num_beams=5, min_length=1)


@tool("correct_grammar", description="this tool corrects the grammar of an english sentence it gets a sentence and returns grammar corrected form")
def correct_grammar(wrong_sentence: str) -> str:
    """this tool corrects the grammar of an english sentence
        it gets a sentence and returns grammar corrected form"""
    print("hello")
    correct_grammar=happy_tt.generate_text(wrong_sentence, args=args).text
    print("result  ",correct_grammar)
    return correct_grammar




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
    

