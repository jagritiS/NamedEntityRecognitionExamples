import pandas as pd
import random

import spacy
from spacy.training.example import Example
from spacy import displacy
from pathlib import Path
nlp = spacy.load("en_core_web_sm")

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.tree import Tree

text = '''
The migration of maize from tropical to temperate climates was accompanied by a dramatic evolution in flowering time. To
gain insight into the genetic architecture of this adaptive trait, we conducted a 50K SNP-based genome-wide association
and diversity investigation on a panel of tropical and temperate American and European representatives. Eighteen genomic
regions were associated with flowering time. The number of early alleles cumulated along these regions was highly
correlated with flowering time. Polymorphism in the vicinity of the ZCN8 gene, which is the closest maize homologue to
Arabidopsis major flowering time (FT) gene, had the strongest effect. This polymorphism is in the vicinity of the causal factor
of Vgt2 QTL. Diversity was lower, whereas differentiation and LD were higher for associated loci compared to the rest of the
genome, which is consistent with selection acting on flowering time during maize migration. Selection tests also revealed
supplementary loci that were highly differentiated among groups and not associated with flowering time in our panel,
whereas they were in other linkage-based studies. This suggests that allele fixation led to a lack of statistical power when
structure and relatedness were taken into account in a linear mixed model. Complementary designs and analysis methods
are necessary to unravel the architecture of complex traits. Based on linkage disequilibrium (LD) estimates corrected for
population structure, we concluded that the number of SNPs genotyped should be at least doubled to capture all QTLs
contributing to the genetic architecture of polygenic traits in this panel. These results show that maize flowering time is
controlled by numerous QTLs of small additive effect and that strong polygenic selection occurred under cool climatic
conditions. They should contribute to more efficient genomic predictions of flowering time and facilitate the dissemination
of diverse maize genetic resources under a wide range of environments.'''

doc = nlp(text)
html = displacy.render(doc, style="ent", jupyter=False, page=True, minify=True)
output_path = Path("ner_nyt.html")
output_path.open("w", encoding="utf-8").write(html)

text = '''High density genotyping tools available today are expected to
help in the discovery, fine mapping and allele diversity character-
ization of regions involved in flowering time. However, the choice
of panel is very important as the level of polymorphism in each
genetic group will determine the power of the analysis. In
domesticated species like maize, loci that are critical to both local
adaptation and yield performance, such as flowering time loci, are
often targets of both natural and artificial selection, leading to
complex forms of allele sharing and admixture among diverse
genetic groups. Differentiation of flowering time between maize
genetic groups is actually clear at the QTL level [29,30]. Genome-
wide association mapping and selection scans can provide
complementary information to help decipher the architecture of
such adaptive traits. For example, in the case of extreme
differentiation leading to fixation of different alleles in different
groups, the loci will be undetectable when association genetics
approaches are used that include structure in the model, but will
show significant tests of selection.
This study was thus designed to a'''
doc = nlp(text)
html = displacy.render(doc, style="ent", jupyter=False, page=True, minify=True)
output_path = Path("ner_teslarati.html")
output_path.open("w", encoding="utf-8").write(html)


def built_spacy_ner(text, target, type_ent):
    start = str.find(text, target)
    end = start + len(target)

    if start != -1:
        return (text, {"entities": [(start, end, type_ent)]})
    else:
        return -1


def built_spacy_ner(text, target, type):
    start = str.find(text, target)
    end = start + len(target)

    return (text, {"entities": [(start, end, type)]})

TRAIN_DATA = []

TRAIN_DATA.append(built_spacy_ner("Genome-Wide Analysis of Yield in Europe: Allelic Effects Vary with Drought and "
                                  "Heat Scenarios", "Genome-Wide Analysis of Yield in Europe: Allelic Effects Vary "
                                                    "with Drought and Heat Scenarios", "TOPIC"))
TRAIN_DATA.append(built_spacy_ner("Assessing the genetic variability of plant performance under heat and drought "
                                  "scenarios can contribute to reduce the negative effects of climate change",
                                  "heat and drought", "ENVIRONMENT_CONDITION"))
TRAIN_DATA.append(built_spacy_ner("six scenarios of temperature and water deﬁcit as experienced by maize (Zea mays "
                                  "L.) plants", "maize", "PLANT"))
TRAIN_DATA.append(built_spacy_ner("Forty-eight quantitative trait loci(QTLs) of yield were identiﬁed by association "
                                  "genetics using a multi-environment multi-locus model.", "multi-locus model",
                                  "MODEL_NAME"))
TRAIN_DATA.append(built_spacy_ner("Genome-wide association study (GWAS) allows associations of phenotypic traits",
                                  "Genome-wide association study (GWAS)", "STUDY_NAME"))
TRAIN_DATA.append(built_spacy_ner("this way, estimate the frequencies of positive, negative, or null effects for "
                                  "each QTL in each climatic scenario, depending on measured environmental conditions in each ﬁeld.", "positive, negative, or null effects", "FORMAT"))
TRAIN_DATA.append(built_spacy_ner("Future conditions have been simulated by using the model LARS-WG", "LARS-WG",
                                  "MODEL_NAME"))
TRAIN_DATA.append(built_spacy_ner("I work for Autodesk.", "Autodesk", "ORG"))
TRAIN_DATA.append(built_spacy_ner("Model Derivative API provides translation", "Model Derivative API", "API"))
TRAIN_DATA.append(built_spacy_ner("The Model Derivative API used in conjunction with the Viewer", "Model Derivative API", "API"))
TRAIN_DATA.append(built_spacy_ner("I would like to automate Revit with the Design Automation API", "Design Automation API", "API"))

text = '''
dern uniform varieties [5].
Compared to some other species like Arabidopsis [6] sorghum
[7] and rice [8,9], for which natural variations at a limited number
of genes have been shown to have a large effect, flowering time
architecture in maize is more complex. Several tens of small effect
QTLs have been detected [4,10]. This suggests that maize
flowering involves a network of genes interacting in many
signaling pathways. Among the loci that have been highlighted,
the maize INDETERMINATE1 (ID1) gene is an important
regulator of maize autonomous flowering that acts in leaves to
mediate the expression of mobile signals that are hypothetical
flowering hormones called florigens [11,12,13] which promote
flowering at the shoot apical meristem. ZCN8 was found to be
controlled by ID1 and to express a florigen in leaves [14]. It is
homologous to the Arabidopsis FLOWERING LOCUS T (FT), a
kinase regulator [15]. FT is a key integrator because almost all
flowering pathways (autonomous, gibberellins, photoperiod and
vernalization) converge on it, and FT transmits the floral inductive
signal to downstream floral identity genes [16]. In maize, a family
of 25 FT homologues including ZCN8 have been published [17].
They are named Zea CENTRORADIALIS (ZCN) genes. Expression
analysis demonstrated that some of them are involved in
developmental processes. A second gene that has been shown to
have a major downstream effect is Dfl1 (Delayed flowering1), a
transcription factor that expresses in the shoot apical region [18].
Mutants,'''

doc = nlp(text)
html = displacy.render(doc, style="ent", jupyter=False, page=True, minify=True)
output_path = Path("ner_forge_before.html")
output_path.open("w", encoding="utf-8").write(html)

# adding a named entity label
ner = nlp.get_pipe('ner')

# Iterate through training data and add new entitle labels.
for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# creating an optimizer and selecting a list of pipes NOT to train
optimizer = nlp.create_optimizer()
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):
    for itn in range(10):
        random.shuffle(TRAIN_DATA)
        losses = {}

        # batch the examples and iterate over them
        for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)

print("Final loss: ", losses)

text = '''Grain yield averaged over the whole panel ranged
from 1.5 to 11.2 t ha 21 in our experiments (from 1.2 to
12.9 t ha 21 for best linear unbiased estimators (BLUEs)
of the reference hybrid; Fig. 2A; Supplemental Table
S1). The respective differences in yield between sites
and within each site is illustrated in Figure 3, A, C, and
E, for three experiments in either favorable conditions
or with water deﬁcit or high temperature. Large dif-
ferences in yield were observed between experiments in
each scenario but with an overlap between the distri-
butions of yields in the panel (Supplemental Fig. S2).
Figure 4 presents the genotypic variability of yield for
hybrids belonging to the ﬁrst (Fig. 4, A–C) and third
(Fig. 4, D–E) quartile of yield values in WW-Cool
scenarios, ordered in each quartile by yield values in
the scenario WW-Hot (all hybrids are presented in
Supplemental Fig. S3 and Supplemental Table S3).
Highest yields and grain numbers were observed in
experiments classiﬁed as Cool-WW (10.8 t ha 21 average
for the reference hybrid, 9.4 t ha 21 for panel mean; Fig.
4; Fig. 2A, blue ellipse)'''

doc = nlp(text)
html = displacy.render(doc, style="ent", jupyter=False, page=True, minify=True)
output_path = Path("ner_forge_after.html")
output_path.open("w", encoding="utf-8").write(html)


text = '''The single environment (SE) and multi-environment
(ME) GWAS together identiﬁed 467 SNPs signiﬁcantly
associated with grain yield, as illustrated in Figure 3
for three experiments, 296 with grain number and
215 with grain size. Signiﬁcant SNPs were then grouped
according to genetic distances, with a threshold at 0.1 cM,
leading to the identiﬁcation of 115 QTLs for grain yield,For example, a QTL on bin 5.01 (5.4 Mb; Fig. 6K)
showed a larger allelic effect on yield in WW com-
pared to WD scenarios. Seven QTLs were signiﬁcant
for grain yield in cool temperatures but disappeared at
high temperature (Table II; Fig. 5) as shown in Figure
6L and Figure 7D for a QTL of grain yield on bin 1.01
(2.4 Mb) with a high correlation between its allelic ef-
fect and T night (r = 20.42) and with T max (r = 20.34).
Eleven QTLs appeared only in WW and cool experi-
ments and tended to disappear in hot and/or drought
scenarios.
'''

for sent in sent_tokenize(text):
   for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
      if hasattr(chunk, 'label'):
         print(chunk.label(), ' '.join(c[0] for c in chunk))
