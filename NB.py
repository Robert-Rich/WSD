## naive-bayes classification model

## I was not able to upload the folders DICT, TRAIN, TEST, and GOLD, but I assume you have these so I have only turned in the code.
import re
from collections import defaultdict
import nltk 
from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import sys 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# this is a global list with punctuation symbols
punctuation=['.','[',']','@','(',')','!','"','--',"'",'/',':','#','?',';',',','-']
#global list with stopwords 
StopWords = stopwords.words('english')

# the lemmatizer ...
lem = WordNetLemmatizer()

def scrubText(line):
    ## This function receives in a list of words: line
    ## It should return a list of cleaned words, removing undesired tokens and lowercasing.
    cline = []
    for word in line:
        word=word.lower()
        word=word.strip()
        word=re.sub('^<.*>$','',word)
        word=re.sub('^@@\d+','',word)
        for p in punctuation : 
            if word == p : word=''
        word = lem.lemmatize(word)
        if not(word in StopWords):
            if len(word) != 0 : cline.append(word)
    return(cline)

def  readDict(f):
    tagUID = defaultdict(lambda: '')
    for line in f.readlines():
        uid_search = re.search('<sen uid=(\d+).*tag=(\w+)',line)
        if(uid_search):
            uid = uid_search.group(1).strip()
            tag = uid_search.group(2).strip()
            #print(line)
            #print("uid=",uid, "tag=",tag)
            tagUID[uid] = tag
    return(tagUID)

def readTrain(f):
    countInst= defaultdict(lambda: 0)
    # In order to implement smoothing I initialize the dictionary to 1 for all words
    countCond= defaultdict(lambda: defaultdict(lambda: 1))
    Ndoc = 0
    line = f.readline()
    while line :
        entry_number = re.match('^[0-9]{6}$',line)
        if(entry_number):
            text=[]
            id = line.strip()
            Ndoc +=1 
            #print(id)
            line=f.readline()
            while(not(re.match('^\n+',line))):
                #print(line)
                tag_search=re.search('<tag .([0-9]{6})',line)
                if(tag_search):
                    tag=tag_search.group(1)
                    #print(tag)
                    countInst[tag]+=1 
                    #remove the tag
                    line=re.sub('<tag.*</>','',line)
                words = line.split()
                text.extend(scrubText(words))
                line=f.readline()
            for w in text:
                countCond[tag][w]+=1

            #line=f.readline()
            #print(text)
            
        line=f.readline()
    return countInst,countCond,Ndoc

def readTest(f):
    testDict= defaultdict(lambda: '')
 
    line = f.readline()
    while line :
        entry_number = re.match('^[0-9]{6}$',line)
        if(entry_number):
            
            id = line.strip()
            #print(id)
            testDict[id]=[]
            line=f.readline()
            while(not(re.match('^\n+',line))):
                #remove the tag
                line=re.sub('<tag.*</>','',line)
                words = line.split()
                testDict[id].extend(scrubText(words))
                line=f.readline()
            #line=f.readline()
            #print(testDict[id])
            
        line=f.readline()
    return testDict

def readGold(f):
    gold = defaultdict(lambda: [])
    for line in f.readlines():
        line = line.strip()
        fld = re.split(':',line)
        sen = re.split('or',fld[1])
        for w in range(len(sen)):
            sen[w]=sen[w].strip()
        gold[fld[0]]=sen
        
    return gold

# calculate the prior and conditional probabilities
def calcProbs(countInst, countCond,Ndoc):
    logPrior = defaultdict(lambda: 0.0)
    logCond = defaultdict(lambda: defaultdict(lambda: 0.0))
    for s in countInst.keys():
        logPrior[s] = math.log(countInst[s]/Ndoc)

    # get the vocabulary
    Vocab = []
    for t in countCond.keys():
        Vocab.extend(countCond[t].keys())
    #print(Vocab)

    for t in countCond.keys():
        denum = 0
        for w in Vocab:
            denum += countCond[t][w]            
        for w in Vocab:
            logCond[t][w] = math.log(countCond[t][w]/denum)
            
    return logPrior,logCond,Vocab

def testNaiveBayes(testdoc,logPrior,logCond,Vocab):
    prob = defaultdict(lambda: 0)
    for s in logPrior.keys():
        prob[s] = logPrior[s]
        for w in testdoc:
            if w in Vocab:
                prob[s] += logCond[s][w]
    #gets the key with the max prob
    sens = max(prob,key=prob.get)
    #print(prob)
    #print(sens)
    
    return sens

### the main function
def main():
    # give the option to specify an other word
    tt='-p'
    if(len(sys.argv)>1):
        word = sys.argv[1]
        if(len(sys.argv)>2):
            tt = sys.argv[2]
        else:
            tt = '-p'
    else:
        word = 'sanction'

    print('Working with: ', word, tt)

    # read the dictionary
    print("Reading dictionary...")
    fdict = open('DICT/'+word+'.dic')
    tagUID=readDict(fdict)
    fdict.close()
    print("... Done!")

    #print(tagUID)

    #read the training data
    print("Reading training data...")
    ftrain = open('TRAIN/'+word+'.cor')
    countInst,countCond,Ndoc = readTrain(ftrain)
    ftrain.close()
    print("...Done!")
    #print(countInst)

    print("Reading Test data... ")
    ftest = open('TEST/'+word+tt+'.eval')
    testDict = readTest(ftest)
    ftest.close()
    print("...Done!")
    #print(testDict)

    print("Reading GOLD...")
    fgold = open('GOLD/'+word+tt)
    gold = readGold(fgold)
    fgold.close()
    print("...Done!")
    #print(gold)
    
    print("Training...")
    # get all the senses of this word
    sensIDs = list(countInst.keys())
    logPrior,logCond,Vocab=calcProbs(countInst,countCond,Ndoc)
    print("...Done!")

    print("Testing...")
    NtestDoc =0
    Nsuccess = 0 
    for d in testDict.keys():
        NtestDoc += 1
        sens = testNaiveBayes(testDict[d],logPrior,logCond,Vocab)
        #print(d,tagUID[sens],gold[d])
        if tagUID[sens] in gold[d] :
            #print("   @@@Success!!!!")
            Nsuccess +=1
        #else:
        #    print("   @@@Failure....")

    print("\t***********************************")
    print("\tSuccess rate: ", Nsuccess/ NtestDoc)
    print("\t***********************************")
    print("Well done!")
if __name__ == "__main__":
    main()
    
