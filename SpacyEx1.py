# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("We implemented this setup to capture dynamic biological processes in challenging environments using the opti-cal arrangement and resolution obtained above. To this end, we first optimized the time-lapse imaging of plant samples using the camera’s built-in intervalometer to establish an adequate frequency for visualizing dynamic processes. This setup allowed us to operate the camera autonomously and use a computer to control the system for more specific and intricate tasks, such as automated focus stacking operations.")
doc = nlp(text)
doc = " ".join([token.text for token in doc if token.pos_ != 'DET'])
# Analyze syntax
doc = nlp(doc)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

text = ("We implemented this setup to capture dynamic biological processes in challenging environments using the opti-cal arrangement and resolution obtained above. To this end, we first optimized the time-lapse imaging of plant samples using the camera’s built-in intervalometer to establish an adequate frequency for visualizing dynamic processes. This setup allowed us to operate the camera autonomously and use a computer to control the system for more specific and intricate tasks, such as automated focus stacking operations.")
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

