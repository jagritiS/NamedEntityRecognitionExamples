from pdfreader import SimplePDFViewer
import io
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
class Topics:
    def extract_text_from_pdf(pdf_path):
        print("Extracting text from PDF...")
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)

        converter.close()
        text = fake_file_handle.getvalue()
        fake_file_handle.close()

        # Tokenize the text and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stop_words]

        # Tokenize the text and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Create a TF-IDF vectorizer and fit it to the tokens
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
        tfidf = vectorizer.fit_transform(tokens)

        # Apply Non-negative Matrix Factorization (NMF) to the TF-IDF matrix to extract topics
        nmf = NMF(n_components=10, random_state=1)
        nmf.fit(tfidf)
        feature_names = vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[:-10 - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
def main():
    pdf_path = 'miappe.pdf'
    Topics.extract_text_from_pdf(pdf_path)
    print('completed')


if __name__ == "__main__":
    main()