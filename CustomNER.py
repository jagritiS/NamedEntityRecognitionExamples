
import pandas as pd
import os
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

# Load the plant dataset
TRAIN_DATA = [("I saw a beautiful orchid in the park today", {"entities": [(18, 24, "PLANT")]} ),
              ("The roses in my garden are blooming",{"entities": [(4, 9, "PLANT")]}),
              ("I love to drink peppermint tea",  {"entities": [(19, 28, "PLANT")]} )]


#nlp = spacy.blank("en") # load a new spacy model
nlp = spacy.load("en_core_web_sm") # load other spacy model

db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)


db.to_disk("train.spacy") # save the docbin object

