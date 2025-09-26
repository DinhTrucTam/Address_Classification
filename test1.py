# address_parser.py
# Main module for address parsing. Loads data from xlsx files and parses addresses.
# Improved: Enhanced abbreviation expansion, robust prefix removal, ward matching from end.

import pandas as pd
import unicodedata
import re

# Section 1: Data Loading
def load_data(file_path, sheet_name='Sheet1'):
    """Load names from xlsx file, converting all values to strings."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)
    names = df[0].dropna().tolist()
    return [name.strip() for name in names if name.strip()]

# Load data from provided xlsx files
provinces = load_data('province.xlsx', sheet_name='provinces')
districts = load_data('district.xlsx', sheet_name='districts')
wards = load_data('ward.xlsx', sheet_name='wards')

# Section 2: Normalization Functions
def remove_accents(text):
    """Remove diacritics from text."""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def normalize_text(input_text):
    """Normalize input: lower, remove punctuation, fix misspellings, remove prefixes."""
    # Step 1: Lowercase
    text = input_text.lower()
    print(f"After lowercase: {text}")
    
    # Step 2: Remove punctuation
    text = re.sub(r'[.,;!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"After punctuation removal: {text}")
    
    # Step 3: Expand abbreviations
    abbreviations = {
        'tp': 'thanh pho', 'q': 'quan', 'p': 'phuong', 'h': 'huyen',
        'tt': 'thi tran', 'tx': 'thi xa', 'tp.': 'thanh pho',
        't.': 'tien '  # For T.Giang -> tien giang
    }
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{abbr}\b', full, text)
    # Handle numbered abbreviations like p1 -> phuong 1, then remove phuong
    text = re.sub(r'\b(p|q|h|tt)(\d+)\b', r'\2', text)  # Replace p1 with 1, q3 with 3, etc.
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"After abbreviation expansion: {text}")
    
    # Step 4: Fix common misspellings
    misspellings = {
        'manh': 'minh', 'ho chi manh': 'ho chi minh', 'hochiminh': 'ho chi minh'
    }
    for wrong, correct in misspellings.items():
        text = text.replace(wrong, correct)
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"After misspelling fix: {text}")
    
    # Step 5: Remove common prefixes
    prefixes = [
        'thanh pho', 'tinh', 'quan', 'huyen', 'phuong', 'xa', 'thi tran', 'duong', 'so', 'ngo', 'hem',
        'đuong', 'phường', 'quận', 'thành phố', 'thị trấn', 'huyện', 'tỉnh'
    ]
    for prefix in prefixes:
        text = re.sub(rf'\b{prefix}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"After prefix removal: {text}")
    
    # Step 6: Remove accents
    text = remove_accents(text)
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"After accent removal: {text}")
    
    return text

# Section 3: Fuzzy Matching
def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fuzzy_match(candidate, name_list, threshold=80):
    """Find best match with score >= threshold, prefer same token count."""
    candidate_no_accent = remove_accents(candidate.lower())
    best_match = None
    best_score = 0
    for name in name_list:
        name_no_accent = remove_accents(name.lower())
        dist = levenshtein_distance(candidate_no_accent, name_no_accent)
        score = 100 * (1 - dist / max(len(candidate_no_accent), len(name_no_accent)))
        if len(candidate_no_accent.split()) == len(name_no_accent.split()):
            score += 5
        if score > best_score:
            best_score = score
            best_match = name
        if score > 70:
            print(f"Matching '{candidate}' with '{name}', Score: {score:.2f}")
    if best_score >= threshold:
        return best_match
    return None

# Section 4: Parsing Logic
def parse_address(input_text):
    """Parse address and return dict."""
    print(f"Processing input: {input_text}")
    normalized = normalize_text(input_text)
    tokens = normalized.split()
    print(f"Tokens: {tokens}")
    
    # Match province from the end
    province = None
    province_end_idx = len(tokens)
    for length in range(1, min(4, len(tokens) + 1)):
        for start in range(len(tokens) - length, len(tokens)):
            candidate = ' '.join(tokens[start:start + length])
            print(f"Trying province candidate: {candidate}")
            match = fuzzy_match(candidate, provinces)
            if match:
                province = match
                province_end_idx = start
                print(f"Matched province: {province}")
                break
        if province:
            break
    
    if not province:
        print("No province matched")
        return {"province": "", "district": "", "ward": ""}
    
    remaining_tokens = tokens[:province_end_idx]
    print(f"Remaining tokens for district: {remaining_tokens}")
    
    # Match district from the end of remaining
    district = None
    district_end_idx = len(remaining_tokens)
    for length in range(1, min(4, len(remaining_tokens) + 1)):
        for start in range(len(remaining_tokens) - length, len(remaining_tokens)):
            candidate = ' '.join(remaining_tokens[start:start + length])
            print(f"Trying district candidate: {candidate}")
            match = fuzzy_match(candidate, districts)
            if match:
                district = match
                district_end_idx = start
                print(f"Matched district: {district}")
                break
        if district:
            break
    
    ward_remaining = remaining_tokens[:district_end_idx] if district else remaining_tokens
    print(f"Remaining tokens for ward: {ward_remaining}")
    
    # Match ward: try candidates from the end (1-4 tokens)
    ward = None
    for length in range(1, min(5, len(ward_remaining) + 1)):
        for start in range(len(ward_remaining) - length, len(ward_remaining)):
            candidate = ' '.join(ward_remaining[start:start + length])
            print(f"Trying ward candidate: {candidate}")
            match = fuzzy_match(candidate, wards, threshold=80)
            if match:
                ward = match
                print(f"Matched ward: {ward}")
                break
        if ward:
            break
    
    return {
        "province": province or "",
        "district": district or "",
        "ward": ward or ""
    }

# Example usage
if __name__ == "__main__":
    # Test inputs
    test_addresses = [
        "Khang Thọ Hưng Hoằng Đức, Hoằng Hóa, Thanh Hóa", # province = Đúc
        "357/28,Ng-T- Thuật,P1,Q3,TP.HồChíMinh.",
        "284DBis Ng Văn Giáo, P3, Mỹ Tho, T.Giang.",
        "123 Đường ABC, p XYZ, q Bình Thạnh, thành phố Hồ Chí Manh",
        "Số 4, Đường Nguyễn Tất Thành, Phường 12, Quận 4, TP.HồChíMinh",
        "Thị trấn Sóc Sơn, Huyện Sóc Sơn, Tỉnh Hà Giang",
        "Thi trấ Ea. Knốp,H. Ea Kar," # no province
    ]
    
    for address in test_addresses:
        print(f"\nInput: {address}")
        result = parse_address(address)
        print(f"Output: {result}\n")