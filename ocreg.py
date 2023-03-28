import spacy
import scispacy
from pdfreader import SimplePDFViewer, PDFDocument
import spacy
import spacy
import scispacy
from spacy import displacy
import en_core_web_sm
import en_core_sci_sm
import en_core_sci_md
# Load spaCy's pre-trained model for English
nlp = spacy.load('en_core_sci_md')# Read in the PDF file and extract text
with open('miappe.pdf', 'rb') as f:
    # Create a PDF viewer object and set its render callbacks
    viewer = SimplePDFViewer(f)
    viewer.render()
    # Extract the text from the viewer object
    article_text = ''.join(viewer.canvas.strings)

# Process the text with spaCy's NLP pipeline
doc = nlp(article_text)
print(list(doc.sents))
print(doc.ents)
# Extract relevant entities
# Extract relevant entities
dates = []
locations = []
researchers = []
funding_sources = []
plant_material = []
environmental_conditions = []
data_acquisition = []
experimental_design = []

for ent in doc.ents:
    print('ent level is ',ent.label_)
    print('ent text is ',ent.text)

    if ent.label_ == 'DATE':
        dates.append(ent.text)
    elif ent.label_ == 'GPE':
        locations.append(ent.text)
    elif ent.label_ == 'PERSON':
        researchers.append(ent.text)
    elif ent.label_ == 'ORG':
        funding_sources.append(ent.text)
    elif 'plant' in ent.text.lower() or 'material' in ent.text.lower():
        plant_material.append(ent.text)
    elif 'environment' in ent.text.lower() or 'condition' in ent.text.lower():
        environmental_conditions.append(ent.text)
    elif 'data' in ent.text.lower() or 'instrument' in ent.text.lower() or 'protocol' in ent.text.lower() or 'processing' in ent.text.lower():
        data_acquisition.append(ent.text)
    elif 'design' in ent.text.lower() or 'replicate' in ent.text.lower() or 'layout' in ent.text.lower() or 'randomization' in ent.text.lower() or 'blocking' in ent.text.lower():
        experimental_design.append(ent.text)

# Print the results

print('Experiment dates:', dates)
print('Experiment locations:', locations)
print('Researchers involved:', researchers)
print('Funding sources:', funding_sources)
print('Plant material description:', plant_material)
print('Environmental conditions:', environmental_conditions)
print('Data acquisition and analysis methods:', data_acquisition)
print('Experimental design:', experimental_design)
