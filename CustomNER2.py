import spacy
import srsly
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin
nlp = spacy.load("en_core_web_sm")
text = "I saw a beautiful orchid and a playful dog in the park today."
corpus = []

doc = nlp(text)
for sent in doc.sents:
    corpus.append(sent.text)

nlp = spacy.blank("en")

ruler = nlp.add_pipe("entity_ruler")

patterns = [
    {"label": "PLANT_NAME", "pattern": "orchid"},
    {"label": "ANIMAL_NAME", "pattern": "dog"}
]

ruler.add_patterns(patterns)
print('ruler patter is :',ruler.patterns)
def convert(lang: str, TRAIN_DATA, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for text, annot in TRAIN_DATA:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)

TRAIN_DATA = []
for sentence in corpus:
    doc = nlp(sentence)
    entities = []

    for ent in doc.ents:
        entities.append([ent.start_char, ent.end_char, ent.label_])
    TRAIN_DATA.append([sentence, {"entities": entities}])

print('check tokens : ',[(token.text, token.pos_, token.tag_) for token in doc])
print ('train data outout : ',TRAIN_DATA)

convert("en", TRAIN_DATA, "trains.spacy")
convert("en", TRAIN_DATA, "valid.spacy")

nlp1 = spacy.load("output/model-best") #load the best model
print('pipe names nlp1 : ',nlp1.pipe_names)
doc = nlp1("orchid is the best flower ever and i have a dog ") # input sample text
print('sample test')
print(doc.ents)