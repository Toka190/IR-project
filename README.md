IR Project (Python-based Information Retrieval System)

This project is a Python-based Information Retrieval (IR) system that implements essential IR techniques like tokenization, stemming, indexing, and phrase query processing. The system allows you to read text files, process the text, and search for terms or phrases with detailed frequency counts.

Technologies: Python, NLTK (Natural Language Toolkit) , Panadas

Main Features:

Tokenization: Splits the text into tokens (words or phrases).

Stemming: Reduces words to their root form using stemming algorithms.

Indexing: Builds an index of terms to allow efficient search and retrieval.

Phrase Query: Supports search queries for exact phrases within the text.

Text Preprocessing: Includes stop word removal and normalization of terms.

Installation:

Clone the repository: git clone https://github.com/Toka190/ir-project

Install dependencies: pip install -r requirements.txt

Run the script: python ir_project.py

How It Works:

Tokenization: The text is broken down into individual tokens (words/phrases) for easier processing.

Stemming: Words are reduced to their base or root form (e.g., "running" becomes "run").

Indexing: An index is created for all the unique terms in the document, making search operations faster.

Phrase Query: Users can input exact phrases, and the system retrieves results based on phrase matching and frequency counts.

This project provides a simple way to search through text data using common IR techniques.

