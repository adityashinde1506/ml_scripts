#!/usr/bin/env python3

import os
import sys
import re

from collections import Counter

DATAFOLDER=sys.argv[1]

def collect_data(doc_tuples):
    doc_list=list()
    label_list=list()
    word_list=list()
    data_list=list()
    for doc,label,words in doc_tuples:
        _id=get_id(doc_list,doc)
        _label=get_id(label_list,label)
        _counter=get_wordcount(word_list,list(words))
        print("Finished {} \t word search space is {} \t data is {},{}".format(doc,len(word_list),_id,_label))
        data_list.append((_id,_label,_counter))
    return data_list

def format_data(data):
    data_file=open("datafile.txt","w")
    label_file=open("labels.txt","w")
    for _id,_label,_counter in data:
        for word in _counter.items():
            print("Writing {} {} {} to data file.".format(_id,word[0],word[1]))
            data_file.write("{} {} {}\r\n".format(_id,word[0],word[1]))
        label_file.write("{}\r\n".format(_label))
    data_file.close()
    label_file.close()

def get_id(_list,element):
    if element not in _list:
        _list.append(element)
    return _list.index(element)

def get_wordcount(wordlist,words):
    counter=Counter()
    for word in words:
        if word not in wordlist:
            wordlist.append(word)
        counter[str(wordlist.index(word))]+=1
    return counter

def get_words(doc_tuple):
    pattern=r"[a-zA-Z]+"
    return doc_tuple+(map(lambda x:x.lower(),re.findall(pattern,open(doc_tuple[0]).read())),)

def get_documents(folder):
    return list(map(lambda x:(folder+"/"+x,folder),os.listdir(folder)))

if __name__=="__main__":

# Get folders from dataset and choose last 2 for classification.
    folders=os.listdir(DATAFOLDER)[-2:]
    print(folders)
    sys.exit()
# Convert to absolute path.
    folders=list(map(lambda x:DATAFOLDER+"/"+x,folders))
# Get documents from folders.
    raw_data=list(map(get_documents,folders))
    raw_data=list(map(get_words,raw_data[0]+raw_data[1]))
    data=collect_data(raw_data)
    format_data(data)
