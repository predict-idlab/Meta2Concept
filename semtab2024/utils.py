from openai import OpenAI
from constants import *
import re
import json


def message_gpt(messages, temperature=0.7, seed=42):
    client = OpenAI(
        api_key=API_KEY_OPENAI,
    )

    response = client.chat.completions.create(
        #model="gpt-4-0125-preview",
        model="gpt-4o",
        response_format={ "type": "text" },
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
        #top_p=1.0,
        seed=seed,
    )

    # print input and output tokens
    print('Tokens:', response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
    
    return response.choices[0].message.content


def message_llama(messages, temperature=0.7, seed=42):
    client = OpenAI(
        api_key=API_KEY_DEEPINFRA,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    response = client.chat.completions.create(
        #model="meta-llama/Meta-Llama-3-8B-Instruct",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        response_format={ "type": "text" },
        messages=messages,
        temperature=temperature,
        max_tokens=8000,
        #top_p=1.0,
        seed=seed,
    )

    # print input and output tokens
    print('Tokens:', response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)

    return response.choices[0].message.content


def print_glossary(glossary):
    txt = ""
    # print each row
    for i in range(len(glossary)):
        # id_no_prefix, label, description
        desc = glossary.iloc[i]['desc']
        # end with . if not already
        if desc[-1] != '.':
            desc += '.'

        domain = glossary.iloc[i]['domain']
        domain = '' if domain == None else '' + domain 

        r = glossary.iloc[i]['range']
        r = '' if r == None else '' + r
        
        txt += f"[{domain},{glossary.iloc[i]['id_no_prefix']},{r}] {desc}\n"
    
    # remove escape characters for links
    txt = txt.replace('\\', '')
    return txt


def extract_content(text):
    # Define the regex pattern to match numbered lines and their content
    pattern = re.compile(r'\d+\.\s*(.*?)(?=(?:\d+\.\s)|$)', re.DOTALL)
    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    # Extract the content between the numbers
    extracted_content = [match.strip() for match in matches]
    
    return extracted_content


def extract_identifiers(text):
    # find all numbered ids
    ids = re.findall(r'\[\d+\]', text)

    # remove brackets
    ids = [re.sub(r'\[|\]', '', i) for i in ids]

    # conver to integers
    ids = [int(i)-1 for i in ids]

    return ids


def get_numbered_ids(text):
    # find all numbered ids
    ids = re.findall(r'\d+\.\s.*', text)

    # remove numbers
    ids = [re.sub(r'\d+\.\s', '', i) for i in ids]

    # split by ":"
    ids = [i.split(';')[0] for i in ids]

    # remove leading and trailing whitespaces
    ids = [i.strip() for i in ids]

    # remove special characters from markdown, *, etc
    ids = [re.sub(r'[\*]', '', i) for i in ids]

    # get the first word
    ids = [i.split(' ')[0] for i in ids]

    #print(ids)

    return ids


def print_descriptions(descriptions):
    txt = ""
    for i in range(len(descriptions)):
        txt += f"[{i+1}] {descriptions[i]}\n"
    # remove last newline
    txt = txt[:-1]
    return txt


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_column_descriptions(row, glossary, rerank):
    # get metadata
    id = row['id']
    label = row['label']
    table_cols = row['table_columns']
    table = row['table_id']

    descriptions = []
    # get the top match
    for col in table_cols:
        if col == label:
            continue
        # make id from table
        col_id = table+'##'+col
        # get the top match
        m = [m['mappings'] for m in rerank if m['id'] == col_id][0]
        m = int(m[0]['id'])
        descriptions.append(col + ': ' + glossary.iloc[m]['desc'])

    return descriptions