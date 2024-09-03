import os
import json
from spider import Spider
from pydantic import BaseModel, Field
from textwrap import dedent
from openai import OpenAI
from collections.abc import MutableMapping
import streamlit as st
import pandas as pd
import dotenv

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL = "gpt-4o-2024-08-06"

def spider_cloud_scrape(url):
    # Initialize the Spider with your API key
    app = Spider(api_key=os.getenv('SPIDER_CLOUD_API_KEY'))

    # Crawl a entity
    crawler_params = {
        "limit": 1,
        "proxy_enabled": True,
        "store_data": False,
        "metadata": False,
        "request": "http",
        "return_format": "markdown",
    }

    try:
        scraped_data = app.crawl_url(url, params=crawler_params)
        print("scraped data found")
        markdown = scraped_data[0]["content"]
    except Exception as e:
        print(e)
        markdown = "Error: " + str(e)

    return markdown

def extract_data(raw_data, response_format):
    
    prompt = '''
        You are a world class web scraper.
        You will extract data based on raw_content we crawl from an url'''

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": prompt
            },
            {
                "role": "user", 
                "content": f"RAW_CONTENT: {raw_data}"
            }
        ],
        response_format=response_format
        )

    return json.loads(response.choices[0].message.content)

def url_to_data(url, response_format):
    raw_data = spider_cloud_scrape(url)

    return extract_data(raw_data, response_format)

def flatten_json(nested_json):
    out = []

    def flatten(x, name='', parent_key=''):
        if isinstance(x, MutableMapping):
            for a in x:
                flatten(x[a], name + a + '_', a)
        elif isinstance(x, list):
            for i, item in enumerate(x):
                flatten(item, name, parent_key)
        else:
            out.append((parent_key if parent_key else name[:-1], x))

    flatten(nested_json)
    return dict(out)

def flatten_json_array(json_data):
    flattened_data = []
    
    for item in json_data:
        arrays_found = False
        
        # Split the key-value pairs into those with lists and those without
        single_entries = {}
        array_entries = {}
        
        for key, value in item.items():
            if isinstance(value, list):
                array_entries[key] = value
                arrays_found = True
            else:
                single_entries[key] = value
        
        if arrays_found:
            for key, value_list in array_entries.items():
                for entry in value_list:
                    flattened_entry = flatten_json(entry)
                    combined_entry = {**single_entries, **flattened_entry}
                    flattened_data.append(combined_entry)
        else:
            flattened_data.append(single_entries)
    
    return flattened_data

class Price(BaseModel):
    size: str
    price: str

class MenuItem(BaseModel):
    name: str = Field(..., description='name of the menu item')
    ingredients: str = Field(..., description='what ingredients does this menu item has?')
    prices: list[Price]

class Menu(BaseModel):
    menus: list[MenuItem]

st.title("Menu scraper")
url = st.text_input("Restaurant url", "https://bravotrattoria.com.au/lunch-and-dinner-menu/")
if st.button("Scrape"):
    menu = url_to_data(url, Menu)
    df = pd.DataFrame(flatten_json_array(menu['menus']))
    print(df)

    st.data_editor(df)