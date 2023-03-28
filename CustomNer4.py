import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

text = "IN THE MATTER OF a proposed contract between the Department of Citywide Administrative Services" \
       " of the City of New York and Tesla, Inc., located at 3500 Deer Creek Rd., Palo Alto, CA 94304," \
       " for procuring Tesla Model 3 All-Electric Sedans. The contract is in the amount of $12,360,000.00." \
       " The term of the contract shall be five years from date of Notice of Award. The proposed contractor " \
       "has been selected by Sole Source Procurement Method, pursuant to Section 3-05 of the Procurement Policy " \
       "Board Rules. If the plan does go through, the $12.36 million could effectively purchase about 274 units of the" \
       " base Model 3 Rear-Wheel-Drive, which cost $44,990 under Tesla's current pricing structure."

doc = nlp(text)
displacy.render(doc, style="ent")
def built_spacy_ner(text, target, type):
    start = str.find(text, target)
    end = start + len(target)

    return (text, {"entities": [(start, end, type)]})
TRAIN_DATA = []
TRAIN_DATA.append(
  built_spacy_ner("I work for Autodesk.", "Autodesk", "ORG")
  )