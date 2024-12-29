#Here, we are attempting to create a NLP project that can take our provided text and generate a good summary of that text.

#For this we can follow following steps:
# 1. Tokenize the text
# 2. Find the importance of each word
# 3. Sum the importance of words to know the importance of sentence
# 4. Select the sentence with highest importance

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter



        

with open('ValidatorWikiText.txt') as f:
    text = f.read()
nlp = spacy.load('en_core_web_md')   

doc=nlp(text)


#Its always a good idea to remove the stop words from the text as they are not important

# for token in doc:
#     if (not token.is_stop& (not token.is_punct) & (token !='\n')):
#         print(token.text.lower())

tokens=[token.text.lower() for token in doc 
        if not token.is_stop 
        and not token.is_punct
        and token.text !='\n']


#We can also do it like this using the Spacy library
tokens1=[]
stopwords= list(STOP_WORDS)
allowed_pos=['NOUN','VERB','PROPN','ADJ'] #These are the part of the sentences that are the most important.
for token in doc:
    if token.text.lower() in stopwords or token.text.lower() in punctuation:
        continue
    if token.pos_ in allowed_pos:
        tokens1.append(token.text.lower())


#We use the counter class in Collections library to count the frequency of each word in the text.
#This will return a disctionary with the frequency of each word in the whole text.
#Now, we divide the frequency of each word by the maximum frequency of any word in the text to get the frequency of each word from 0 to 1
freq_counter=Counter(tokens1)
max_freq=max(freq_counter.values())

for freq in freq_counter.keys():
    freq_counter[freq]=freq_counter[freq]/max_freq #This works because each key corresponds to a certain value of frequency which is being reassigned.


#Now lets tokenize each sentences
sent_token=[sent.text for sent in doc.sents]


#Now, lets determine the importance of each sentences by the help of how many non_Stop tokens they have.
sent_score={} #A disctionary
for sent in sent_token:
    for word in sent.split(): #THis splits the sentence into each words by spliting then on new word character.
        if word.lower() in freq_counter.keys(): #This checks if the word is a non-stop word 
            if sent not in sent_score.keys():#THis now assigns the score of the sentence by finding the first non-stop word in the sentence.
                sent_score[sent]=freq_counter[word.lower()]
            else:#This will add the socre by finding additional non-stop words in the sentences.
                sent_score[sent]+=freq_counter[word.lower()]


#Now, we can do either of extractive text summarization, which involves us selecting a number of sentences with highest scores amd using them as it is to show summary.
#Or we can do abstractive text summarization, which involves us generating our own sentences based on the scores of the sentences.

#Extractive Text Summarization
from heapq import nlargest

print("How many sentences do you want in the summary?")
n_sentences=int(input())
summary_sentence= nlargest(n_sentences,sent_score,key=sent_score.get) #This selectes the no. of keys with the highest values from the disctionary
print(" ".join(summary_sentence),"\n")




#We can also show summary without the no of sentences by simply using the no of words in original sentence and using 1/4 of that
QuaterSummary= nlargest(int(len(sent_score)/4),sent_score,key=sent_score.get)
print("Here is a quater summary of the text \n".join(QuaterSummary),"\n")




#Now lets try abstracting text summarization
#We will use a transformer model for thsi purpose.
#A transformer is a deep learning model. They work by encoding the sequence of words and transform it into format that captures the essence of the sentence and various relationships in that sentence
#We can then use these outputs from transformer to our purpose.
#These transformers are the backbone of NLP in current world.


#This project only deals with extractive text summarization.






#Here is this summarized in class so, they can be exported easily
class TextSummarizer:
    def __init__(self,text):
        self.text=text
        self.tokens=[]
        

    def tokenizer(self):
        nlp = spacy.load('en_core_web_md')   
        self.doc=nlp(self.text)

        stopwords= list(STOP_WORDS)
        allowed_pos=['NOUN','VERB','PROPN','ADJ'] #These are the part of the sentences that are the most important.
        for token in self.doc:
            if token.text.lower() in stopwords or token.text.lower() in punctuation:
                continue
            if token.pos_ in allowed_pos:
                self.tokens.append(token.text.lower())

    def sent_score(self):
        freq_counter=Counter(self.tokens)
        max_freq=max(freq_counter.values())

        for freq in freq_counter.keys():
            freq_counter[freq]=freq_counter[freq]/max_freq

        sent_token=[sent.text for sent in self.doc.sents]

        self.sent_score={} 
        for sent in sent_token:
            for word in sent.split(): 
                if word.lower() in freq_counter.keys():  
                    if sent not in self.sent_score.keys():
                        self.sent_score[sent]=freq_counter[word.lower()]
                    else:
                        self.sent_score[sent]+=freq_counter[word.lower()]

    def extractive_summarizer(self):
        self.tokenizer()
        self.sent_score()
        summary_sentence= nlargest(int(len(self.sent_score)/7),self.sent_score,key=self.sent_score.get) 
        return " ".join(summary_sentence)
    


summarizer=TextSummarizer(text)
print("The  Summary from Class is: \n")
print(summarizer.extractive_summarizer())


        

        
