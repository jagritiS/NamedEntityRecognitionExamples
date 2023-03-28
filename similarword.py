import spacy
from pdfminer.high_level import extract_text
from pdfreader import SimplePDFViewer, PDFDocument
# Load the pre-trained model for NER from spaCy
nlp = spacy.load('en_core_web_sm')

# Define a function to extract named entities from text
def extract_entities(text):
    with open('miappe.pdf', 'rb') as f:
        # Create a PDF viewer object and set its render callbacks
        viewer = SimplePDFViewer(f)
        viewer.render()
        # Extract the text from the viewer object
        article_text = ''.join(viewer.canvas.strings)
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['PLANT', 'DISEASE', 'GENE']:
            entities.append((ent.text, ent.label_))
    return entities

# Extract text from the PDF file
text = extract_text('miappe.pdf')

# Extract named entities from the text
entities = extract_entities(text)

# Extract the plant name, disease name, and related gene from the named entities
plant_name = None
disease_name = None
gene_name = None

for entity in entities:
    if entity[1] == 'PLANT':
        plant_name = entity[0]
    elif entity[1] == 'DISEASE':
        disease_name = entity[0]
    elif entity[1] == 'GENE':
        gene_name = entity[0]

# Print the results
print(f"Plant Name: {plant_name}")
print(f"Disease Name: {disease_name}")
print(f"Related Gene: {gene_name}")
