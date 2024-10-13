import requests
import pandas as pd
import html2text
import json
from sentence_transformers import SentenceTransformer
import textwrap
import faiss
import numpy as np
import google.generativeai as genai
from openai import OpenAI
import semchunk

from app import text_chunks


def company_search (ticker,email):
    base_url="https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": f"{email}"
    }
    response= requests.get(base_url,headers=headers)

    parsed_data = json.loads(response.text)

    df = pd.DataFrame.from_dict(parsed_data, orient='index')

    col_filtered_data = df[df['ticker'] == ticker]


    ren_data = col_filtered_data['cik_str'].iloc[0]

    padded_ren = str(int(ren_data)).zfill(10)

    return padded_ren


def reports_find_docs_recent_10k(cik,email):
    base_url= f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": f"{email}"
    }
    response = requests.get(base_url, headers=headers)

    data = json.loads(response.text)

    recent_filings = data["filings"]["recent"]

    df = pd.DataFrame(recent_filings)

    docs =['10-K']

    docs_sorted=df[df['form'].isin(docs)]

    columns_to_keep = ['form', 'filingDate', 'accessionNumber', 'primaryDocument']

    sorted_df = docs_sorted[columns_to_keep]

    sorted_df['accessionNumber'] = sorted_df['accessionNumber'].str.replace('-', '')

    new_df = new_df = sorted_df.reset_index(drop=True)

    accession = new_df ['accessionNumber'].iloc[0]

    document_id =new_df ['primaryDocument'].iloc[0]

    return accession, document_id

def get_reports (cik,accession,document_id,email):
    cik_without_zeros = cik.lstrip('0')

    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document_id}"

    headers = {
        "User-Agent": f"{email}"
    }
    response = requests.get(base_url, headers=headers)

    text = html2text.html2text(response.text)

    return text






def chunking_normal(text,chunk_size):
    text_chunks = textwrap.wrap(text, chunk_size)
    return text_chunks


def semantic_chunking(text,chunk_size):
    chunker = semchunk.chunkerify('umarbutler/emubert', chunk_size)
    text_chunks = chunker(text)
    return text_chunks

def  embeddings_gen(text_chunks):
    hf_model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = hf_model.encode(text_chunks)

    return embeddings

def creation_and_adding(embeddings):


    dimension = 384

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def quarying(query,index,text_chunks):
    hf_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = hf_model.encode([query])
    k = 8
    distances, indices = index.search(np.array(query_vector), k)
    rel_chunks = ""
    for idx in indices[0]:
        rel_chunks += (f"{text_chunks[idx]}")

    return rel_chunks

def google_gen_ai(query,rel_chunks):
    import os

    genai.configure(api_key = "")

    prompt = f"""
    QUERY: {query}
    RELEVENT DOCS: {rel_chunks}
    Instructions:
    1. Answer the user's query by breaking down your response into clear, detailed points.
    2. Use headings for each point to enhance readability.
    3. Provide thorough explanations and insights based on the relevant lines with directly citing the source text.
    4. Do not use phrases like "based on the information" or "things like that"; instead, provide concrete details.
    5. Provide a detailed response as possible

    Ensure the response is comprehensive and informative, maintaining clarity throughout.

    """

    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text


def openai_gen_ai(query,rel_chunks):
    token = ""
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "gpt-4o"

    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )
    prompt =f"""
   You are a helpful assistant tasked with answering user queries based on specific report lines for a company, which may include quarterly or annual data.

   USER QUERY: {query}

   RELEVANT LINES: {rel_chunks}

   Instructions:
   1. Answer the user's query by breaking down your response into clear, detailed points.
   2. Use headings for each point to enhance readability.
   3. Provide thorough explanations and insights based on the relevant lines without directly citing the source text.
   4. Do not use phrases like "based on the information" or "things like that"; instead, provide concrete details.

   Ensure the response is comprehensive and informative, maintaining clarity throughout.
   """
    answer = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return answer.choices[0].message.content






email = "arunavdaniel@gmail.com"

ticker = "SPGI"

results1 = company_search(ticker,email)

results2 = reports_find_docs_recent_10k(results1,email)

results3 = get_reports(results1,results2[0],results2[1],email)










