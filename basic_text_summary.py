import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import heapq

text = """Crowds had been flocking there all night and all morning, their vehicles jamming up the typically sleepy two-lane road that gets you in and out of these parts. Infrastructure-wise, Waimea Bay isn’t exactly meant to host one of the world’s most-storied sporting events. But every once in a while, when the waves are just right, it does. The famed Eddie Aikau Big Wave Invitational is the rarest of rare surf contests, one that defies human scheduling and relies instead on the whims of nature. It requires exceptionally specific conditions: Waves in Waimea Bay must reliably reach an awesome, gut-churning height of 40 feet minimum. Even though the contest—named in honor of the legendary Native Hawaiian lifeguard and big-wave surfer Eddie Aikau—has been going since 1985, this was only its 10th run.
The chosen few who are invited to compete will drop everything and fly in from Australia and Tahiti, Brazil and Portugal. So when word got out in January that the contest was on, the world’s best surfers began racing to Waimea."""    

def basic_summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    # print(stopwords)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)

    tokens = [token.text for token in doc]
    # print(tokens)

    #calculate frequencies

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1


    # print(word_freq)

    max_freq = max(word_freq.values())

    # print(max_freq)

    sent_tokens = [sent for sent in doc.sents]

    #creates sentence scores where the score is how many SPECIAL words
    # there are in a sentence

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] +=  word_freq[word.text]

    # print(sent_scores)


    #n of senstences you want
    select_len = int( len(sent_tokens) * 0.3)

    # Get the n largest (select_len) from the dict sent_scores.
    #Order using the sentence scores

    summary = heapq.nlargest(select_len , sent_scores, key=sent_scores.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    print(summary)

    return summary, doc, len(rawdocs.split(' ')) , len(summary)





