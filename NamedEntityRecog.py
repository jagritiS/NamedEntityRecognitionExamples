import spacy
import spacy
import scispacy
from spacy import displacy
import en_core_web_sm
import en_core_sci_sm
import en_core_sci_md

from pdfreader import SimplePDFViewer, PDFDocument
class NER:
    def usingCustomEntities(self,custom_named_entities):
        nlp = spacy.load('en_core_sci_md')
 #doc = nlp(article_text) # Process the text with the NER model
        doc = nlp(self)
        # Extract named entities and their labels
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        # Filter for named entities of interest
        named_entities = [ne for ne in named_entities if
                          ne[0] in custom_named_entities]
        # Print the named entities
        return(named_entities)

    def usingDefinedEntities(self):
        # Load pre-trained English NER model
        nlp = spacy.load('en_core_web_sm')

        # Define the text
        text = 'The genotypic variation of time to anthesis was from 63.9 to 75.9 d 20°C in the panel, with narrow sense heritabilities from 0.19 to 0.83 (median = 0.68). The correlation between time to anthesis and yield tended to be positive in WW fields (r from 0.10 ns to 0.56, P value , 0.001; data not shown), indicating that latest hybrids had slightly higher yield and grain number than earlier hybrids, most likely due to a longer cumulated photosynthesis.'

        # Process the text with the NER model
        doc = nlp(self)

        # Print the named entities and their labels
        for ent in doc.ents:
            return(ent.text, ent.label_)

    def usingCustomAndDefinedEntities(self,custom_named_entities):
        # Load pre-trained English NER model
        nlp = spacy.load('en_core_web_sm')

        # Add your custom named entities to the NER model
        for entity in custom_named_entities:
            nlp.vocab.strings.add(entity)

        # Process the text with the modified NER model
        doc = nlp(self)

        # Print the named entities and their labels
        for ent in doc.ents:
            return(ent.text, ent.label_)


def main():
    text = '''The second step consisted in performing ﬁeld
                experiments with a panel of genotypes over a range of
                conditions. This was done in 29 ﬁeld experiments (de-
                ﬁned as combinations of site 3 year 3 watering re-
                gime), in which a panel of 244 maize hybrids was
                analyzed along a climatic transect from west to east
                Europe, plus one experiment in Chile. This panel,
                genotyped with 515 000 single nucleotide polymor-
                phism (SNP) markers, maximized the genetic variabil-
                ity in the dent maize group while restricting the range
                of ﬂowering time to 10 d in order to avoid confounding
                the effects of phenology with intrinsic responses to
                drought and heat. It included ﬁrst-cycle lines derived
                from historical landraces and more recent lin experimental design were carefully controlled during the study.'''
    custom_named_entities = ['Plant', 'Growing conditions', 'Experimental design']
    # Load pre-trained English NER model
    nlp = spacy.load('en_core_web_sm')
    # Define the text
    # Load spaCy's pre-trained model for English
    nlp = spacy.load('en_core_web_sm')

    # Read in the PDF file and extract text
    with open('miappe.pdf', 'rb') as f:
        # Create a PDF viewer object and set its render callbacks
        viewer = SimplePDFViewer(f)
        viewer.render()

        # Extract the text from the viewer object
        article_text = ''.join(viewer.canvas.strings)

    print('Using predefined entities :',NER.usingDefinedEntities(article_text))
    print('Using custom entities :',NER.usingCustomEntities(article_text,custom_named_entities))
    print('Using predefined and custom entities :',NER.usingCustomAndDefinedEntities(article_text,custom_named_entities))


if __name__ == "__main__":
    main()