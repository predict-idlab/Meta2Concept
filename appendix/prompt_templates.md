# Prompt templates
## Enriching of vocabulary
```  
Can you clean up the following? Fix mistakes and translate everything 
in English if not. Write abbreviations or acronyms in full and add the 
abbreviation in brackets after the word. Convert names to their type.

### Example
Q: 
0. last win
1. charles
2. Harelbeke

A: 
0. Last win
1. King (Charles)
2. City (Harelbeke)

### Data
```
## Enriching of table metadata
```
Can you clean up the following? Fix mistakes and translate everything 
in English if not. Write abbreviations or acronyms in full and add the 
abbreviation in brackets after the word. Convert names to their type. 
Add a relation e.g. is a, of a or other. You must understand the meaning 
of the word first to write the relation.

### Example
Q: 
0. year, film
1. developer, videogame
2. harelbeke, country

A: 
0. Year of a film
1. Developer of a videogame
3. City (Harelbeke) of a country

### Data
```
## Top-k matching
```
Given a table '{table_name}' with columns {table_columns} what is the best 
description for the column '{column}'? I will provide you with {num} triples
in the format [domain, property, range] with a description of the property, 
each indicated by number identifier [].

{descriptions}

You must rank the top five descriptions best matching the column and provide
the 5 unique identifiers using the output format [].

### Example
Analysis: <reasoning>

1. [id1]
2. [id2]
3. [id3]
4. [id4]
5. [id5]

### Answer
```
## Table matching
```
Given a table '{table_name}' with columns:

{table_columns_descriptions}

Your task is to determine which of the above is the best description for the 
column '{column}' given the context of the other columns. I will provide you 
with {num} descriptions, each indicated by number identifier [].

{descriptions}

You must give only the best description and provide the unique identifier 
using the output format [].

### Example
Analysis: <your step-by-step reasoning here>

[identifier of the best fitting description]
### Answer
```
## Non-contextual Method
```
You are given a table in JSON-LD standard and a prompt. Take into account 
the table_name and table_columns to rank the top five official fine-grained 
property from dbpedia.org/ontology/ vocabulary for the prompt column. Don't 
respond in json.

### Example
table: 
{{"table_name": "Museum", "table_columns": ["RANG", "Museum", "Stadt", 
"Facebook-Fans"]}}

prompt: Stadt

fine-grained property:
1: location
2: locationCity
3: locationName
4: city
5: livingPlace

### Task
table: 
{{"table_name": "{table_name}", "table_columns": {table_columns}}}

prompt: {column}

fine-grained property:
```
