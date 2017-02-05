from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

LANGUAGE = "english"
SENTENCES_COUNT = 5

def summarize(terms, http_url="http://www.apple.com/legal/internet-services/itunes/us/terms.html"):
    url = http_url #"http://www.wwe.com/page/terms-and-conditions"# "https://www.google.com/adsense/localized-terms"# "https://www.spotify.com/us/legal/end-user-agreement/"# "http://www.apple.com/legal/internet-services/itunes/us/terms.html"
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    sentence_list = []
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        sentence_list.append(sentence)
        print ('----')
        print (sentence)
    return sentence_list

# summarize("n")
