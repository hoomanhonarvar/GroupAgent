# from happytransformer import HappyTextToText, TTSettings
from nltk.corpus import wordnet as wn


# # happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

# # args = TTSettings(num_beams=5, min_length=1)
# save_path = "../../models/saved_t5_grammar"

# # happy_tt.tokenizer.save_pretrained(save_path)
# # happy_tt.model.save_pretrained(save_path)


# happy_tt = HappyTextToText("T5", save_path)
# args = TTSettings(num_beams=5, min_length=1)

# result = happy_tt.generate_text("grammar: you is a students", args=args)

# print(result.text)

def get_synonyms_antonyms(word):
    synonyms = set()
    antonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name().replace('_', ' '))

    return list(synonyms), list(antonyms)


syns, ants = get_synonyms_antonyms("happy")
print("Synonyms:", syns)
print("Antonyms:", ants)
