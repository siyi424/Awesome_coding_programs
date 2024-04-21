import spacy

def tokenize_src(text):
    nlp = spacy.load("en_core_web_sm")
    return [tok.text for tok in nlp.tokenizer(text)]

doc = tokenize_src("I am a student.")

print(doc) 
# ['I', 'am', 'a', 'student', '.']