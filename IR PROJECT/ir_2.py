import os
import nltk
import collections
import math
import numpy as np
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from natsort import natsorted,ns
from nltk.stem import PorterStemmer

files = natsorted(os.listdir('files'))
def preprocessing(doc):
    stemmed = PorterStemmer()
    token_docs = word_tokenize(doc)
    prepared_doc = [stemmed.stem(term) for term in token_docs]
    return natsorted(prepared_doc)

tokenized_documents = []

for file in files:
    with open(f"files/{file}") as f:
        document = f.read()
        print(document)
#tokenization
        tokenized_document = word_tokenize(document)
        list_of_terms = []
        for term in tokenized_document:
                list_of_terms.append(term)
        tokenized_documents.append(list_of_terms)

print("------------------------------tokenized_document-------------------------------")
print(tokenized_documents)


# stemming
def stemming(tokenized_documents):
    stemmer = PorterStemmer()
    stem_document = []

    for terms in tokenized_documents:
        stemmd_term = [] 
        for word in terms:
            stemmed_word = stemmer.stem(word)
            stemmd_term.append(stemmed_word)
        stem_document.append(stemmd_term)

    return stem_document

stem=stemming(tokenized_documents)
print("--------------------------stemming-----------------------------------")
print(stem)

   
fileno=1
pos_index={}
 #open files
file_name = natsorted(os.listdir('files'))
documentOfTerms = []
for files in file_name:
    with open (f'files/{files}','r') as f :
        doc = f.read()
        # print(document)  
    token_list=preprocessing(doc)
    
#positional index:    
    for pos,term in enumerate(token_list):
        if term in pos_index:
            pos_index[term][0]=pos_index[term][0]+1
            if fileno in pos_index[term][1]:
                pos_index[term][1][fileno].append(pos)
            else:
                pos_index[term][1][fileno]=[pos] 
        else:
            pos_index[term]=[]
            pos_index[term].append(1)
            pos_index[term].append({})
            pos_index[term][1][fileno]=[pos]
    fileno+=1
print("--------------------------positinal_index-----------------------------------\n",pos_index)                      
       
       
#phrase query: 
def put_query(q, display=1):
    lis = [[] for i in range(10)]
    q = preprocessing(q)
    for term in q:

        if term in pos_index.keys():
            for key in pos_index[term][1].keys():
            
                if lis[key-1] != []:
                    
                    if lis[key-1][-1] == pos_index[term][1][key][0]-1:
                        lis[key-1].append(pos_index[term][1][key][0])
                else:
                    lis[key-1].append(pos_index[term][1][key][0])
    positions = []
    if display==1:
        for pos, list in enumerate(lis, start=1):
            if len(list) == len(q):
                positions.append('document '+str(pos))
        return positions
    else:
        for pos, list in enumerate(lis, start=1):
            if len(list) == len(q):
                positions.append('doc'+str(pos))
        return positions
   

all_terms = []
for document in stem:
    for token in document:
        all_terms.append(token)
        
#computing TF        
def compute_tf(document):

    terms_freq = dict.fromkeys(all_terms, 0)

    for term in document:
        terms_freq[term] += 1

    return terms_freq


def get_term_freq():
    terms_freq = pd.DataFrame()
    for i in range(len(stem)):
        terms_freq[i] = compute_tf(stem[i])
    terms_freq.columns=['doc'+str(i) for i in range(1, 11)]    
    return terms_freq

terms_freq = get_term_freq()
print("------------------------------TF--------------------------------------")
print(terms_freq)

# compute weighted_tf
def compute_wtf(term_freq):
    if term_freq > 0:
        return math.log(term_freq)+1
    return 0

def get_wtf():
    wtf = pd.DataFrame()
    for i in range(len(stem)):
        wtf['doc'+str(i+1)] = terms_freq['doc'+str(i+1)].apply(compute_wtf)
    return wtf

wtf= get_wtf()
print("------------------------------WTF-----------------------------------")
print(wtf)


#compute idf
get_idf = pd.DataFrame(columns=['df', 'idf'])

for i in range(len(terms_freq)):
    frequency = terms_freq.iloc[i].values.sum()
    get_idf.loc[i, 'df'] = frequency
    get_idf.loc[i, 'idf'] = math.log10(10 / float(frequency))


get_idf.index = terms_freq.index
print("------------------------------df&idf----------------------------------")
print(get_idf)

# compute tf*idf
tf_idf = terms_freq.multiply(get_idf['idf'], axis=0)

print("------------------------------tf-idf----------------------------------")
print(tf_idf)

#  Documents_lenghts
def get_length(document):
    return np.sqrt(tf_idf[document].apply(lambda x: x**2).sum())

doc_len = pd.DataFrame()
for document in tf_idf.columns:
    doc_len.loc[0, document] = get_length(document)

print("------------------------------length----------------------------------")
print(doc_len)


# normlized tf*idf
norm= pd.DataFrame()

def normlize(document, tf_idf):
    try:
        return tf_idf / doc_len[document].values[0]
    except ZeroDivisionError:
        return 0
    
for document in tf_idf.columns:
    norm[document] = tf_idf[document].apply(lambda tf_idf: normlize(document, tf_idf))

print("------------------------------Normlized tf*idf-------------------------------")
print(norm)


print("------------------------------insert query-------------------------------")

def insert_query(q):
    is_found = put_query(q, 2)
    if is_found == []:
        return "Not Found"
    new_q = preprocessing(q)  
    query = pd.DataFrame(index=norm.index)
    query['tf'] = [1 if x in new_q else 0 for x in list(norm.index)]
    query['wtf'] = query['tf'].apply(lambda x: compute_wtf(x))
    product = norm.multiply(query['wtf'], axis=0)
    query['idf'] = get_idf['idf'] * query['wtf']
    query['tf_idf'] = query['wtf'] * query['idf']
    query['normalized'] = 0
    for i in range(len(query)):
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
    print('Query')
    print(query.loc[new_q])
    product2 = product.multiply(query['normalized'], axis=0)
    scores = {}
    for col in put_query(q, 2):
        scores[col] = product2[col].sum()
    product_result = product2[list(scores.keys())].loc[new_q]
    print()
    print('Product (query*matched doc):')
    print(product_result)
    print()
    print('sum:')
    print(product_result.sum())
    print()
    print(' Length:')
    q_len = math.sqrt(sum([x**2 for x in query['idf'].loc[new_q]]))
    print(q_len)
    print()
    print('Simliarity:')
    print(product_result.sum())
    print()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print('Returned docs:')
    for typle in sorted_scores:
        print(typle[0], end=" ")
x=input("Enter Phrase :")
print(insert_query(x))

def preprocess(document):
    return set(document.lower().split())

def read_documents_from_path(path):
    documents = []
    for filename in os.listdir('files'):
        with open(os.path.join('files', filename), 'r') as file:
            document_text = file.read()
            documents.append(document_text)
    return documents

#  boolean retrieval
def boolean_retrieval(query):
    query = query.lower()
    if 'and' in query or 'or' in query or 'not' in query:
        query_parts = query.split()
        operator_stack = []
        operand_stack = []
        for part in query_parts:
            if part == 'not':
                operator_stack.append('not')
            elif part in {'and', 'or'}:
                operator_stack.append(part)
            else:
                matched_docs = []
                for doc_idx, doc_set in preprocessed_docs.items():
                    if all(word in doc_set for word in part.split()):
                        matched_docs.append(doc_idx)
                operand_stack.append(set(matched_docs))

        result = operand_stack.pop()
        while operator_stack:
            operator = operator_stack.pop()
            if operator == 'not':
                result = set(preprocessed_docs.keys()).difference(result)
            else:
                operand = operand_stack.pop()
                if operator == 'and':
                    result = result.intersection(operand)
                elif operator == 'or':
                    result = result.union(operand)
        return [(idx, documents[idx - 1]) for idx in sorted(result)]
    else:
        matched_docs = []
        for doc_idx, doc_set in preprocessed_docs.items():
            if all(word in doc_set for word in query.split()):
                matched_docs.append(doc_idx)
        if matched_docs:
            return [(idx, documents[idx - 1]) for idx in matched_docs]
        else:
            print(f"Error: '{query}' not found in documents.")
            return []

# Get the path where the documents are stored
documents_path = "path_to_your_documents_directory"  # Replace this with your actual path

# Read documents from the specified path
documents = read_documents_from_path(documents_path)

# Create a dictionary of preprocessed documents
preprocessed_docs = {idx + 1: preprocess(doc) for idx, doc in enumerate(documents)}

# Get user input for query
user_query = input("Enter your boolean query: ")
retrieved_documents = boolean_retrieval(user_query)
if retrieved_documents:
    print("documents:")
    for doc_id, doc_text in retrieved_documents:
        print(f"Document num: {doc_id} - {doc_text}")