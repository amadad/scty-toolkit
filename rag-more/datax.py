from openai import OpenAI
import fitz  # PyMuPDF
import io
import os
from PIL import Image
import base64
import json

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


@staticmethod
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pdf_to_base64_images(pdf_path):
    #Handles PDFs with multiple pages
    pdf_document = fitz.open(pdf_path)
    base64_images = []
    temp_image_paths = []

    total_pages = len(pdf_document)

    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        temp_image_path = f"temp_page_{page_num}.png"
        img.save(temp_image_path, format="PNG")
        temp_image_paths.append(temp_image_path)
        base64_image = encode_image(temp_image_path)
        base64_images.append(base64_image)

    for temp_image_path in temp_image_paths:
        os.remove(temp_image_path)

    return base64_images

def extract_invoice_data(base64_image):
    system_prompt = f"""
    You are an OCR-like data extraction tool that extracts hotel invoice data from PDFs.
   
    1. Please extract the data in this hotel invoice, grouping data according to theme/sub groups, and then output into JSON.

    2. Please keep the keys and values of the JSON in the original language. 

    3. The type of data you might encounter in the invoice includes but is not limited to: hotel information, guest information, invoice information,
    room charges, taxes, and total charges etc. 

    4. If the page contains no charge data, please output an empty JSON object and don't make up any data.

    5. If there are blank data fields in the invoice, please include them as "null" values in the JSON object.
    
    6. If there are tables in the invoice, capture all of the rows and columns in the JSON object. 
    Even if a column is blank, include it as a key in the JSON object with a null value.
    
    7. If a row is blank denote missing fields with "null" values. 
    
    8. Don't interpolate or make up data.

    9. Please maintain the table structure of the charges, i.e. capture all of the rows and columns in the JSON object.

    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "extract the data in this hotel invoice and output into JSON "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def extract_from_multiple_pages(base64_images, original_filename, output_directory):
    entire_invoice = []

    for base64_image in base64_images:
        invoice_json = extract_invoice_data(base64_image)
        invoice_data = json.loads(invoice_json)
        entire_invoice.append(invoice_data)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the output file path
    output_filename = os.path.join(output_directory, original_filename.replace('.pdf', '_extracted.json'))
    
    # Save the entire_invoice list as a JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(entire_invoice, f, ensure_ascii=False, indent=4)
    return output_filename


def main_extract(read_path, write_path):
    for filename in os.listdir(read_path):
        file_path = os.path.join(read_path, filename)
        if os.path.isfile(file_path):
            base64_images = pdf_to_base64_images(file_path)
            extract_from_multiple_pages(base64_images, filename, write_path)


read_path= "./data/hotel_invoices/receipts_2019_de_hotel"
write_path= "./data/hotel_invoices/extracted_invoice_json"

main_extract(read_path, write_path)

[
    {
        "Hotel Information": {
            "Name": "Hamburg City (Zentrum)",
            "Address": "Willy-Brandt-Straße 21, 20457 Hamburg, Deutschland",
            "Phone": "+49 (0) 40 3039 379 0"
        },
        "Guest Information": {
            "Name": "APIMEISTER CONSULTING GmbH",
            "Guest": "Herr Jens Walter",
            "Address": "Friedrichstr. 123, 10117 Berlin"
        },
        "Invoice Information": {
            "Rechnungsnummer": "GABC19014325",
            "Rechnungsdatum": "23.09.19",
            "Referenznummer": "GABC015452127",
            "Buchungsnummer": "GABR15867",
            "Ankunft": "23.09.19",
            "Abreise": "27.09.19",
            "Nächte": 4,
            "Zimmer": 626,
            "Kundereferenz": 2
        },
        "Charges": [
            {
                "Datum": "23.09.19",
                "Uhrzeit": "16:36",
                "Beschreibung": "Übernachtung",
                "MwSt.%": 7.0,
                "Betrag": 77.0,
                "Zahlung": null
            },
            {
                "Datum": "24.09.19",
                "Uhrzeit": null,
                "Beschreibung": "Übernachtung",
                "MwSt.%": 7.0,
                "Betrag": 135.0,
                "Zahlung": null
            },
            {
                "Datum": "25.09.19",
                "Uhrzeit": null,
                "Beschreibung": "Übernachtung",
                "MwSt.%": 7.0,
                "Betrag": 82.0,
                "Zahlung": null
            },
            {
                "Datum": "26.09.19",
                "Uhrzeit": null,
                "Beschreibung": "Übernachtung",
                "MwSt.%": 7.0,
                "Betrag": 217.0,
                "Zahlung": null
            },
            {
                "Datum": "24.09.19",
                "Uhrzeit": "9:50",
                "Beschreibung": "Premier Inn Frühstücksbuffet",
                "MwSt.%": 19.0,
                "Betrag": 9.9,
                "Zahlung": null
            },
            {
                "Datum": "25.09.19",
                "Uhrzeit": "9:50",
                "Beschreibung": "Premier Inn Frühstücksbuffet",
                "MwSt.%": 19.0,
                "Betrag": 9.9,
                "Zahlung": null
            },
            {
                "Datum": "26.09.19",
                "Uhrzeit": "9:50",
                "Beschreibung": "Premier Inn Frühstücksbuffet",
                "MwSt.%": 19.0,
                "Betrag": 9.9,
                "Zahlung": null
            },
            {
                "Datum": "27.09.19",
                "Uhrzeit": "9:50",
                "Beschreibung": "Premier Inn Frühstücksbuffet",
                "MwSt.%": 19.0,
                "Betrag": 9.9,
                "Zahlung": null
            }
        ],
        "Payment Information": {
            "Zahlung": "550,60",
            "Gesamt (Rechnungsbetrag)": "550,60",
            "Offener Betrag": "0,00",
            "Bezahlart": "Mastercard-Kreditkarte"
        },
        "Tax Information": {
            "MwSt.%": [
                {
                    "Rate": 19.0,
                    "Netto": 33.28,
                    "MwSt.": 6.32,
                    "Brutto": 39.6
                },
                {
                    "Rate": 7.0,
                    "Netto": 477.57,
                    "MwSt.": 33.43,
                    "Brutto": 511.0
                }
            ]
        }
    }
]

[
    {
        "hotel_information": {
            "name": "string",
            "address": {
                "street": "string",
                "city": "string",
                "country": "string",
                "postal_code": "string"
            },
            "contact": {
                "phone": "string",
                "fax": "string",
                "email": "string",
                "website": "string"
            }
        },
        "guest_information": {
            "company": "string",
            "address": "string",
            "guest_name": "string"
        },
        "invoice_information": {
            "invoice_number": "string",
            "reservation_number": "string",
            "date": "YYYY-MM-DD",  
            "room_number": "string",
            "check_in_date": "YYYY-MM-DD",  
            "check_out_date": "YYYY-MM-DD"  
        },
        "charges": [
            {
                "date": "YYYY-MM-DD", 
                "description": "string",
                "charge": "number",
                "credit": "number"
            }
        ],
        "totals_summary": {
            "currency": "string",
            "total_net": "number",
            "total_tax": "number",
            "total_gross": "number",
            "total_charge": "number",
            "total_credit": "number",
            "balance_due": "number"
        },
        "taxes": [
            {
                "tax_type": "string",
                "tax_rate": "string",
                "net_amount": "number",
                "tax_amount": "number",
                "gross_amount": "number"
            }
        ]
    }
]

def transform_invoice_data(json_raw, json_schema):
    system_prompt = f"""
    You are a data transformation tool that takes in JSON data and a reference JSON schema, and outputs JSON data according to the schema.
    Not all of the data in the input JSON will fit the schema, so you may need to omit some data or add null values to the output JSON.
    Translate all data into English if not already in English.
    Ensure values are formatted as specified in the schema (e.g. dates as YYYY-MM-DD).
    Here is the schema:
    {json_schema}

    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Transform the following raw JSON data according to the provided schema. Ensure all data is in English and formatted as specified by values in the schema. Here is the raw JSON: {json_raw}"}
                ]
            }
        ],
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)



def main_transform(extracted_invoice_json_path, json_schema_path, save_path):
    # Load the JSON schema
    with open(json_schema_path, 'r', encoding='utf-8') as f:
        json_schema = json.load(f)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Process each JSON file in the extracted invoices directory
    for filename in os.listdir(extracted_invoice_json_path):
        if filename.endswith(".json"):
            file_path = os.path.join(extracted_invoice_json_path, filename)

            # Load the extracted JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_raw = json.load(f)

            # Transform the JSON data
            transformed_json = transform_invoice_data(json_raw, json_schema)

            # Save the transformed JSON to the save directory
            transformed_filename = f"transformed_{filename}"
            transformed_file_path = os.path.join(save_path, transformed_filename)
            with open(transformed_file_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_json, f, ensure_ascii=False, indent=2)

   
    extracted_invoice_json_path = "./data/hotel_invoices/extracted_invoice_json"
    json_schema_path = "./data/hotel_invoices/invoice_schema.json"
    save_path = "./data/hotel_invoices/transformed_invoice_json"

    main_transform(extracted_invoice_json_path, json_schema_path, save_path)

import os
import json
import sqlite3

def ingest_transformed_jsons(json_folder_path, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create necessary tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Hotels (
        hotel_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        street TEXT,
        city TEXT,
        country TEXT,
        postal_code TEXT,
        phone TEXT,
        fax TEXT,
        email TEXT,
        website TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Invoices (
        invoice_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hotel_id INTEGER,
        invoice_number TEXT,
        reservation_number TEXT,
        date TEXT,
        room_number TEXT,
        check_in_date TEXT,
        check_out_date TEXT,
        currency TEXT,
        total_net REAL,
        total_tax REAL,
        total_gross REAL,
        total_charge REAL,
        total_credit REAL,
        balance_due REAL,
        guest_company TEXT,
        guest_address TEXT,
        guest_name TEXT,
        FOREIGN KEY(hotel_id) REFERENCES Hotels(hotel_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Charges (
        charge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_id INTEGER,
        date TEXT,
        description TEXT,
        charge REAL,
        credit REAL,
        FOREIGN KEY(invoice_id) REFERENCES Invoices(invoice_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Taxes (
        tax_id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_id INTEGER,
        tax_type TEXT,
        tax_rate TEXT,
        net_amount REAL,
        tax_amount REAL,
        gross_amount REAL,
        FOREIGN KEY(invoice_id) REFERENCES Invoices(invoice_id)
    )
    ''')

    # Loop over all JSON files in the specified folder
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder_path, filename)

            # Load the JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Insert Hotel Information
            cursor.execute('''
            INSERT INTO Hotels (name, street, city, country, postal_code, phone, fax, email, website) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data["hotel_information"]["name"],
                data["hotel_information"]["address"]["street"],
                data["hotel_information"]["address"]["city"],
                data["hotel_information"]["address"]["country"],
                data["hotel_information"]["address"]["postal_code"],
                data["hotel_information"]["contact"]["phone"],
                data["hotel_information"]["contact"]["fax"],
                data["hotel_information"]["contact"]["email"],
                data["hotel_information"]["contact"]["website"]
            ))
            hotel_id = cursor.lastrowid

            # Insert Invoice Information
            cursor.execute('''
            INSERT INTO Invoices (hotel_id, invoice_number, reservation_number, date, room_number, check_in_date, check_out_date, currency, total_net, total_tax, total_gross, total_charge, total_credit, balance_due, guest_company, guest_address, guest_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                hotel_id,
                data["invoice_information"]["invoice_number"],
                data["invoice_information"]["reservation_number"],
                data["invoice_information"]["date"],
                data["invoice_information"]["room_number"],
                data["invoice_information"]["check_in_date"],
                data["invoice_information"]["check_out_date"],
                data["totals_summary"]["currency"],
                data["totals_summary"]["total_net"],
                data["totals_summary"]["total_tax"],
                data["totals_summary"]["total_gross"],
                data["totals_summary"]["total_charge"],
                data["totals_summary"]["total_credit"],
                data["totals_summary"]["balance_due"],
                data["guest_information"]["company"],
                data["guest_information"]["address"],
                data["guest_information"]["guest_name"]
            ))
            invoice_id = cursor.lastrowid

            # Insert Charges
            for charge in data["charges"]:
                cursor.execute('''
                INSERT INTO Charges (invoice_id, date, description, charge, credit) 
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    invoice_id,
                    charge["date"],
                    charge["description"],
                    charge["charge"],
                    charge["credit"]
                ))

            # Insert Taxes
            for tax in data["taxes"]:
                cursor.execute('''
                INSERT INTO Taxes (invoice_id, tax_type, tax_rate, net_amount, tax_amount, gross_amount) 
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    invoice_id,
                    tax["tax_type"],
                    tax["tax_rate"],
                    tax["net_amount"],
                    tax["tax_amount"],
                    tax["gross_amount"]
                ))

    conn.commit()
    conn.close()


def execute_query(db_path, query, params=()):
    """
    Execute a SQL query and return the results.

    Parameters:
    db_path (str): Path to the SQLite database file.
    query (str): SQL query to be executed.
    params (tuple): Parameters to be passed to the query (default is an empty tuple).

    Returns:
    list: List of rows returned by the query.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute the query with parameters
        cursor.execute(query, params)
        results = cursor.fetchall()

        # Commit if it's an INSERT/UPDATE/DELETE query
        if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            conn.commit()

        return results
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        # Close the connection
        if conn:
            conn.close()


# Example usage
transformed_invoices_path = "./data/hotel_invoices/transformed_invoice_json"
db_path = "./data/hotel_invoices/hotel_DB.db"
ingest_transformed_jsons(transformed_invoices_path, db_path)

query = '''
    SELECT 
        h.name AS hotel_name,
        i.total_gross AS max_spent
    FROM 
        Invoices i
    JOIN 
        Hotels h ON i.hotel_id = h.hotel_id
    ORDER BY 
        i.total_gross DESC
    LIMIT 1;
    '''

results = execute_query(db_path, query)
for row in results:
    print(row)