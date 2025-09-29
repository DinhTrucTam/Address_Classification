import re
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional
import unicodedata
from fuzzywuzzy import fuzz
import json

# Check if a string contains Vietnamese accented characters
def has_accents(s: str) -> bool:
    """Return True if the string contains any Vietnamese accented characters."""
    return any('WITH' in unicodedata.name(c, '') for c in s if c.isalpha())

# Select the best candidate among multiple matches
def break_ties(candidates: List[str], original: str) -> str:
    """Choose the best match by preferring accented names and higher fuzz ratio.
    
    Args:
        candidates: List of potential matching names.
        original: Original string to compare against.
    
    Returns:
        The best matching name.
    """
    # Prefer candidates with Vietnamese accents
    accented = [c for c in candidates if has_accents(c)]
    if accented:
        candidates = accented
    # Sort by fuzz ratio (similarity) in descending order
    candidates.sort(key=lambda c: fuzz.ratio(c.lower(), original), reverse=True)
    return candidates[0]

# Calculate Levenshtein distance between two strings
def levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein distance between two strings.
    
    Args:
        s1: First string.
        s2: Second string.
    
    Returns:
        Integer distance (minimum number of edits to transform s1 into s2).
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    # Initialize previous row for dynamic programming
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        # Initialize current row with index i+1
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate costs for insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            # Append minimum cost to current row
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Load names from an Excel file
def load_data(file_path: str, sheet_name: str) -> Set[str]:
    """Load a set of names from a specified Excel file and sheet.
    
    Args:
        file_path: Path to the Excel file.
        sheet_name: Name of the sheet to read.
    
    Returns:
        Set of stripped, non-empty names from the first column.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)
        names = df[0].dropna().tolist()
        return {name.strip() for name in names if name.strip()}
    except Exception:
        return set()

# Load administrative names from Excel files
provinces: Set[str] = load_data('province.xlsx', sheet_name='provinces')
districts: Set[str] = load_data('district.xlsx', sheet_name='districts')
wards: Set[str] = load_data('ward.xlsx', sheet_name='wards')

# Create lists of (condensed, original) names for matching
def create_list(names: Set[str]) -> List[Tuple[str, str]]:
    """Convert a set of names into a list of tuples with condensed and original forms.
    
    Args:
        names: Set of names to process.
    
    Returns:
        List of tuples (condensed_name, original_name).
    """
    return [(name.lower().replace(' ', ''), name) for name in names]

# Prepare lists for provinces, districts, and wards
province_list = create_list(provinces)
district_list = create_list(districts)
ward_list = create_list(wards)

# Define prefixes for administrative levels (lowercase for matching)
province_prefixes = ['', 'thành phố', 'tp', 'tp.', 't.p', 'tỉnh', 't', 't.', 'tinh', 'thanhpho']
district_prefixes = ['', 'quận', 'q', 'q.', 'huyện', 'h', 'h.', 'thị xã', 'tx', 'tx.', 't.x', 'thành phố', 'tp', 'tp.', 'quan', 'huyen', 'thixa', 'thanhpho']
ward_prefixes = ['', 'phường', 'p', 'p.', 'xã', 'x', 'x.', 'thị trấn', 'tt', 'tt.', 't.t', 'phuong', 'xa', 'thitran']

# Find the best match for a given administrative level
def find_match(tokens: List[str], reference_list: List[Tuple[str, str]], prefixes: List[str]) -> Tuple[Optional[str], int]:
    """Find the best matching name from tokens for a given administrative level.
    
    Args:
        tokens: List of tokenized words from the address.
        reference_list: List of (condensed, original) names for the level.
        prefixes: List of valid prefixes for the level.
    
    Returns:
        Tuple of (best matching name or None, number of tokens consumed).
    """
    all_matches: List[Tuple[str, int, int, int]] = []

    # Try prefix-driven matching
    for i in range(len(tokens)):
        token = tokens[i].lower()
        if token in [p.lower() for p in prefixes if p]:
            max_words = min(4, len(tokens) - i - 1)
            for num_name_words in range(max_words, 0, -1):
                # Form candidate name from subsequent tokens
                name_cand = ' '.join(tokens[i + 1:i + 1 + num_name_words])
                cand_lower = name_cand.lower()
                cond_cand = cand_lower.replace(' ', '')
                if not cond_cand:
                    continue

                # Set threshold for Levenshtein distance
                ref_len = len(cond_cand)
                thresh = max(1, ref_len // 2)
                matches = []
                for cond_ref, orig in reference_list:
                    dist = levenshtein(cond_cand, cond_ref)
                    if dist <= thresh:
                        matches.append((orig, dist))
                if matches:
                    # Select candidates with minimum distance
                    min_dist = min(m[1] for m in matches)
                    best_candidates = [m[0] for m in matches if m[1] == min_dist]
                    best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                    score = fuzz.ratio(best_orig.lower(), cand_lower)
                    if score >= 80:
                        all_matches.append((best_orig, score, min_dist, num_name_words + 1))
                        if score == 100 and num_name_words >= 2:
                            return best_orig, num_name_words + 1

    # Fallback to concatenated/no-prefix logic
    for has_separate_prefix in [True, False]:
        for num_name_words in range(1, 5):
            if has_separate_prefix:
                num_tokens = num_name_words + 1
                if len(tokens) < num_tokens:
                    continue
                prefix_cand = tokens[-num_tokens].lower()
                if prefix_cand not in [p.lower() for p in prefixes if p]:
                    continue
                name_cand = ' '.join(tokens[-num_name_words:])
                cand_lower = name_cand.lower()
                cand_stripped = cand_lower
            else:
                num_tokens = num_name_words
                if len(tokens) < num_tokens:
                    continue
                name_cand = ' '.join(tokens[-num_name_words:])
                cand_lower = name_cand.lower()
                cand_stripped = cand_lower
                for prefix in [p.lower().replace(' ', '') for p in prefixes if p]:
                    if cand_stripped.startswith(prefix):
                        ref_len = len(cand_lower.replace(' ', ''))
                        thresh = max(1, ref_len // 2)
                        matches = []
                        for cond_ref, orig in reference_list:
                            dist = levenshtein(cand_lower.replace(' ', ''), cond_ref)
                            if dist <= thresh:
                                matches.append((orig, dist))
                        if matches:
                            min_dist = min(m[1] for m in matches)
                            best_candidates = [m[0] for m in matches if m[1] == min_dist]
                            best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                            score = fuzz.ratio(best_orig.lower(), cand_lower)
                            if score >= 80:
                                all_matches.append((best_orig, score, min_dist, num_tokens))
                                if score == 100 and num_name_words >= 2:
                                    return best_orig, num_tokens
                        cand_stripped = cand_stripped[len(prefix):].strip()
                        break

            cond_cand = cand_stripped.replace(' ', '')
            if not cond_cand:
                continue

            ref_len = len(cond_cand)
            thresh = max(1, ref_len // 2)
            matches = []
            for cond_ref, orig in reference_list:
                dist = levenshtein(cond_cand, cond_ref)
                if dist <= thresh:
                    matches.append((orig, dist))
            if matches:
                min_dist = min(m[1] for m in matches)
                best_candidates = [m[0] for m in matches if m[1] == min_dist]
                best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                score = fuzz.ratio(best_orig.lower(), cand_lower)
                if score >= 80:
                    all_matches.append((best_orig, score, min_dist, num_tokens))
                    if score == 100 and num_name_words >= 2:
                        return best_orig, num_tokens

    if all_matches:
        best_match = max(all_matches, key=lambda x: (x[3], x[1], -x[2]))
        return best_match[0], best_match[3]

    return None, 0

# Extract address components from text
def extract_address(text: str) -> Dict[str, str]:
    """Extract province, district, and ward from a Vietnamese address string.
    
    Args:
        text: Input address string.
    
    Returns:
        Dictionary with province, district, and ward (empty strings if not found).
    """
    # Preprocess: lowercase, replace punctuation with spaces, normalize whitespace
    text = text.lower()
    text = re.sub(r'[.,/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()

    # Initialize result dictionary
    result: Dict[str, str] = {
        "province": "",
        "district": "",
        "ward": ""
    }

    # Process levels in order: province, district, ward
    levels = [
        ("province", province_list, province_prefixes),
        ("district", district_list, district_prefixes),
        ("ward", ward_list, ward_prefixes)
    ]

    for level_name, reference_list, prefixes in levels:
        match, num_pop = find_match(tokens, reference_list, prefixes)
        if match:
            result[level_name] = match
            tokens = tokens[:-num_pop]  # Remove consumed tokens

    return result

# Test the manual test case
def test_address_extraction():
    """Test address extraction with a manual test case."""
    # Define the manual test case
    manual_test = {
        "text": "Phưng Khâm Thiên Quận Đ.Đa T.Phố HàNội",
        "result": {
            "district": "Đống Đa",
            "ward": "Khâm Thiên",
            "province": "Hà Nội"
        }
    }
    
    # Process the manual test case
    print("\nManual Test Case:")
    print(f"Input: {manual_test['text']}")
    actual = extract_address(manual_test['text'])
    expected = manual_test['result']
    is_correct = actual == expected
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    print(f"Correct: {is_correct}")
    if not is_correct:
        print("Differences:")
        for key in ["province", "district", "ward"]:
            if expected[key] != actual[key]:
                print(f"  {key.capitalize()}: Expected '{expected[key]}', got '{actual[key]}'")

# Run tests if data files are loaded successfully
if __name__ == "__main__":
    if not provinces or not districts or not wards:
        print("Error: One or more data files failed to load.")
    else:
        test_address_extraction()