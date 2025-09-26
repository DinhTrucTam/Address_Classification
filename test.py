import json
import pandas as pd
import re
import time
from collections import defaultdict
from rapidfuzz import fuzz, process
import functools
from unidecode import unidecode  # Install: pip install unidecode
from datetime import datetime

# Load test cases from public.json
def load_test_cases(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load hierarchical data from .xlsx file
def load_hierarchical_data(file_path, sheet_name="Sheet1"):
    address_db = {
        "provinces": {},
        "abbreviations": {
            "tp": "thành phố", "t": "tỉnh", "p": "phường", "q": "quận", "h": "huyện",
            "x": "xã", "tt": "thị trấn", "ng": "nguyễn", "kp": "khu phố", "f": "phường",
            "t.x": "thị xã", "t.p": "thành phố", "tx": "thị xã",
            "q1": "quận 1", "q2": "quận 2", "q3": "quận 3",  # Added for common cases
            "p1": "phường 1", "p2": "phường 2", "p3": "phường 3"
        }
    }
    
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    # Ensure expected columns exist
    expected_columns = ['province', 'district', 'ward']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Excel file must contain columns: {expected_columns}")
    
    # Process each row
    for _, row in df.iterrows():
        province = str(row["province"]).strip()
        district = str(row["district"]).strip()
        ward = str(row["ward"]).strip() if pd.notna(row["ward"]) else ""
        
        province_key = unidecode(province).lower()
        district_key = unidecode(district).lower() if district else ""
        ward_key = unidecode(ward).lower() if ward else ""
        
        # Initialize province if not exists
        if province_key not in address_db["provinces"]:
            address_db["provinces"][province_key] = {"name": province, "districts": {}}
        
        # Initialize district if not exists
        if district_key and district_key not in address_db["provinces"][province_key]["districts"]:
            address_db["provinces"][province_key]["districts"][district_key] = {"name": district, "wards": {}}
        
        # Add ward if present
        if ward_key:
            address_db["provinces"][province_key]["districts"][district_key]["wards"][ward_key] = ward
    
    return address_db

# Precompiled regex patterns
regex_special_chars = re.compile(r'[\-\./,;:\s]+')  # Handle more separators
regex_spaces = re.compile(r'\s+')
regex_numbers = re.compile(r'\d+[^\s]*|bis|tieu khu')  # Remove "bis", "tieu khu"
regex_noise = re.compile(r'\b(j|xt)\b', re.IGNORECASE)  # Remove noise like "J", "XT"

# Normalize input text
def normalize_text(text):
    # Remove noise (e.g., "J", "XT")
    text = regex_noise.sub('', text)
    # Remove numbers, "bis", "tieu khu"
    text = regex_numbers.sub('', text)
    # Replace abbreviations (case-insensitive, word boundaries)
    text = text.lower()
    for abbr, full in address_db["abbreviations"].items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
        text = re.sub(r'\b' + re.escape(abbr) + r'\.', full, text)
    # Remove special characters and normalize spaces
    text = regex_special_chars.sub(' ', text)
    text = regex_spaces.sub(' ', text).strip()
    return unidecode(text)  # Normalize to ASCII

# Log failed cases
def log_failure(text, result, expected, elapsed_time, normalized_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("failed_cases.log", "a", encoding='utf-8') as f:
        f.write(f"[{timestamp}] Input: {text}\n")
        f.write(f"Normalized: {normalized_text}\n")
        f.write(f"Output: {result}\n")
        f.write(f"Expected: {expected}\n")
        f.write(f"Time: {elapsed_time:.4f}s\n\n")

# Cache matching function for performance
@functools.lru_cache(maxsize=1000)
def match_address(text):
    start_time = time.time()
    normalized_text = normalize_text(text)
    
    # Initialize result in the correct order: province, district, ward
    result = {"province": "", "district": "", "ward": ""}
    
    # Try exact matching for province
    for province_key, province_data in address_db["provinces"].items():
        if province_key in normalized_text:
            result["province"] = province_data["name"]
            # Try exact matching for district
            for district_key, district_data in province_data["districts"].items():
                if district_key in normalized_text:
                    result["district"] = district_data["name"]
                    # Try exact matching for ward (only if ward exists in district)
                    for ward_key, ward_name in district_data["wards"].items():
                        if ward_key in normalized_text and ward_key != district_key:
                            result["ward"] = ward_name
                            break
                    break
            break
    
    # Fallback to fuzzy matching if exact match fails
    if not result["province"]:
        province_matches = process.extractOne(
            normalized_text,
            address_db["provinces"].keys(),
            scorer=fuzz.token_set_ratio,  # Better for typos and word order
            score_cutoff=80  # Lowered to capture more matches
        )
        if province_matches:
            province_key = province_matches[0]
            result["province"] = address_db["provinces"][province_key]["name"]
            # Try district matching within province
            district_matches = process.extractOne(
                normalized_text,
                address_db["provinces"][province_key]["districts"].keys(),
                scorer=fuzz.token_set_ratio,
                score_cutoff=80
            )
            if district_matches:
                district_key = district_matches[0]
                result["district"] = address_db["provinces"][province_key]["districts"][district_key]["name"]
                # Try ward matching within district
                ward_matches = process.extractOne(
                    normalized_text,
                    address_db["provinces"][province_key]["districts"][district_key]["wards"].keys(),
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=80
                )
                if ward_matches and ward_matches[0] != district_key:
                    result["ward"] = address_db["provinces"][province_key]["districts"][district_key]["wards"][ward_matches[0]]
    
    elapsed_time = time.time() - start_time
    if elapsed_time > 0.1:
        print(f"Warning: Processing time {elapsed_time:.4f}s exceeds 0.1s for input: {text}")
    
    return result

# Load files
test_case_file = "public.json"  # Update with your file path
xlsx_file = "Book1.xlsx"  # Update with your file path
address_db = load_hierarchical_data(xlsx_file, sheet_name="Sheet1")
test_cases = load_test_cases(test_case_file)

# Run tests
total_time = 0
num_requests = len(test_cases)
passed_tests = 0

for test in test_cases:
    start_time = time.time()
    result = match_address(test["text"])
    elapsed_time = time.time() - start_time
    expected = test["result"]
    match = result == expected
    if match:
        passed_tests += 1
    else:
        log_failure(test["text"], result, expected, elapsed_time, normalize_text(test["text"]))
        print(f"Failed: Input: {test['text']}")
        print(f"Normalized: {normalize_text(test['text'])}")
        print(f"Output: {result}")
        print(f"Expected: {expected}")
    print(f"Input: {test['text']}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {match}")
    print(f"Processing time: {elapsed_time:.4f}s\n")
    total_time += elapsed_time

average_time = total_time / num_requests
print(f"Test Summary:")
print(f"Passed: {passed_tests}/{num_requests} ({passed_tests/num_requests*100:.2f}%)")
print(f"Average processing time: {average_time:.4f}s")
print(f"Total processing time: {total_time:.4f}s")
print(address_db)