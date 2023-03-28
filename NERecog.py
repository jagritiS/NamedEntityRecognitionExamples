import spacy
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
import spacy
from spacy.training.example import Example
from pdfminer.high_level import extract_text
from random import shuffle

nlp = spacy.load("en_core_web_lg")

# Example training data with misaligned entities
TRAIN_DATA = [
    ("Amazon launches new streaming service", {"entities": [(0, 6, "ORG"), (14, 21, "PRODUCT")]}),
    ("Tesla announces new electric car model", {"entities": [(0, 5, "ORG"), (20, 26, "PRODUCT")]}),
    ("Google acquires startup for $1 billion", {"entities": [(0, 6, "ORG"), (16, 23, "PRODUCT"), (28, 36, "MONEY")]}),
    ("Microsoft acquires LinkedIn for $26 billion", {"entities": [(0, 9, "ORG"), (17, 25, "ORG"), (30, 38, "MONEY")]}),
    ("Apple is looking to buy a UK startup for $1 billion", {"entities": [(0, 5, "ORG"), (26, 29, "GPE"), (44, 51, "PRODUCT"), (56, 64, "MONEY")]}),
]

# Check alignment of entities in the training data
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    tags = offsets_to_biluo_tags(doc, annotations.get("entities"))
    print(text)
    print(tags)




# Load the plant dataset
plant_data = [("I saw a beautiful orchid in the park today", {"entities": [(18, 24, "PLANT")]} ),
              ("The roses in my garden are blooming",{"entities": [(4, 9, "PLANT")]}),
              ("I love to drink peppermint tea",  {"entities": [(19, 28, "PLANT")]} )]

# Create a new NLP object and add a blank "ner" pipeline
nlp = spacy.load("en_core_web_sm")
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add the plant labels to the NER model
ner.add_label("PLANT")

# Convert the dataset to spaCy's Example format
examples = []
for text, entities in plant_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, {"entities": entities})
    examples.append(example)

# Train the NER model
optimizer = nlp.initialize()
for i in range(10):
    shuffle(examples)
    for example in examples:
        nlp.update([example], sgd=optimizer)

# Extract text from the PDF file
text = extract_text('miappe.pdf')

doc = nlp(text)
plants = []
for ent in doc.ents:
    if ent.label_ == "PLANT":
        plants.append(ent.text)
print(plants)
