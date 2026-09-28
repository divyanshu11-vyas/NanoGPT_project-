ftfy library
-> full form - Fixes Text For You
-> to rair broken unicode text and mojibake
uses - Database Migration, data cleaning & nlp

chardet library
-> Character Detector
-> used to automatically detect the type of character encoding of unknown byte strings
-> calculates confidence
-> detects byte order marks (BOM) at the beginning of files to identify UTF variants immediately.


bs4 (Beyond basic web scraping)
-> it is module used to import beautifulsoup for parsing raw HTML and XML documents into a navigable "parse tree".
-> A parse tree is a tree-shaped data structure that represents the grammatical 
structure of a string or source code based on a specific set of rules (a context-free grammar).

-> Root Node: 
   Represents the start symbol of the language's grammar (e.g., Program, Expression, or <html>).  
-> Interior Nodes: 
   Represent non-terminal grammar rules (e.g., Statement, Operator, Tag).
-> Leaf Nodes: 
   Represent the actual raw terminal tokens from the source text (e.g., numbers, keywords, operators, string literals, or closing tags).
-> Completeness: 
   Captures 100% of the original syntax, including explicit punctuation, spaces, grouping parentheses, and semicolons.

bs4 uses
-> navigating the DOM ( document object model) tree. 
-> searching and filtering the tags by names
-> html cleanup aka document repair ( beautifiation) and fixing corrupt html files

-> DOM 
   The DOM (Document Object Model) is a cross-platform, language-independent programming interface that converts 
   an HTML or XML document into an object-oriented structure. It bridges static web pages and dynamic programming, 
   allowing languages like JavaScript or libraries like bs4 to read, modify, add, or delete document elements programmatically.

-> DOM tree 
                    [ Document ]
                            |
                         <html>
                       /        \
                 <head>          <body>
                   |             /    \
                <title>       <h1>     <a>
                   |           |        |
               "My Page"    "Hello"  "Click"

-> how code interact with DOM
   1. browsers - js modifies the dom in real time updates
   2. python parsers

-> COMMON PARSERS IN python
   html.parser - for .html files and has descent speed, is built in and its features are standard battery included parser.

   lxml - for html and xml files and is extremely fast, not built in and High-performance C-based parser (libxml2/libxslt).
          Best for heavy web scraping and large XML files.
   
   html5lib - for html files but it slower, not built in

   xml.etree.ElementTree - for XML, FastNone, (Built-in), Python's native lightweight C-optimized XML tree parser and API.




re Library in python
-> provides support for Regular Expressions (Regex), which are specialized text strings used to 
   search, match, extract, and manipulate patterns within textual data.
-> re.sub(PATTERN, REPLACEMENT, TEXT)
-> common functions are

+--------------+-------------------------------------------------------------+------------------------------------+
| Function     | Description                                                 | Common Use Case                    |
+--------------+-------------------------------------------------------------+------------------------------------+
| re.search()  | Scans string to find the FIRST location where pattern matches | Checking if a keyword exists     |
| re.match()   | Matches pattern ONLY at the beginning of the string         | Validating header or start format  |
| re.findall() | Returns a list of ALL non-overlapping pattern matches       | Extracting all emails or numbers   |
| re.finditer()| Returns an ITERATOR yielding Match objects for all matches  | Processing large text efficiently  |
| re.sub()     | REPLACES matched patterns with a replacement string         | Masking sensitive data or cleaning |
| re.subn()    | Same as sub(), but returns a tuple: (new_string, count)     | Replacing while tracking count     |
| re.split()   | SPLITS string by the occurrences of the pattern             | Tokenizing on multiple delimiters  |
| re.compile() | COMPILES a regex pattern into a reusable RegexObject        | Optimizing repeated search loops   |
| re.escape()  | ESCAPES special characters in a pattern automatically       | Escaping raw user input safety     |
+--------------+-------------------------------------------------------------+------------------------------------+




unicode library in python
-> unicode identification
-> removal of accented words
-> unicode normalization

+-----------------------+-------------------------------------------------------------+----------------------------------------------+
| Function              | Description                                                 | Common Use Case                              |
+-----------------------+-------------------------------------------------------------+----------------------------------------------+
| unicodedata.name()    | Returns the official Unicode name of a character            | Identifying unknown symbols, emojis, marks   |
| unicodedata.lookup()  | Returns the character matching a given Unicode name         | Looking up characters by name ("GREEK SIGMA")|
| unicodedata.category()| Returns the general category code (e.g., "Lu", "Nd", "P")   | Filtering out punctuation or symbol types    |
| unicodedata.normalize()| Normalizes text to forms like NFC, NFD, NFKC, or NFKD       | Comparing strings or removing accents        |
| unicodedata.numeric() | Returns the numeric value assigned to a character           | Extracting numbers from Roman numerals/fractions|
+-----------------------+-------------------------------------------------------------+----------------------------------------------+


types of normalization

+------------------------+------------------------------------+-----------------------------------------------------+
| Normalization Form     | Domain                             | Main Goal in Data Cleaning                          |
+------------------------+------------------------------------+-----------------------------------------------------+
| NFKC / Unicode Norm    | Text / Token Preprocessing         | Unify visually identical fonts, symbols & accents   |
| Case Folding           | Text Retrieval / Search            | Eliminate case sensitivity discrepancies            |
| Min-Max Scaling        | Numerical Tabular Features         | Bounds range between [0, 1]                         |
| Z-Score Standardization| ML Feature Engineering             | Centers mean to 0 and variance to 1                 |
| L2 Vector Normalization| Embeddings & Vector Search (RAG)   | Makes dot product equal to cosine similarity        |
+------------------------+------------------------------------+-----------------------------------------------------+





