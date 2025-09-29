import re
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional
import unicodedata
from fuzzywuzzy import fuzz
import json

# Initialize log file
log_file = open('address_extraction.log', 'w', encoding='utf-8')

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
    accented = [c for c in candidates if has_accents(c)]
    if accented:
        candidates = accented
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
    log_file.write(f"\nMatching for level with prefixes: {prefixes}\n")
    log_file.write(f"Tokens: {tokens}\n")
    all_matches: List[Tuple[str, int, int, int]] = []

    # Try prefix-driven matching
    for i in range(len(tokens)):
        token = tokens[i].lower()
        if token in [p.lower() for p in prefixes if p]:
            log_file.write(f"Found prefix '{token}' at index {i}\n")
            max_words = min(4, len(tokens) - i - 1)
            for num_name_words in range(max_words, 0, -1):
                name_cand = ' '.join(tokens[i + 1:i + 1 + num_name_words])
                cand_lower = name_cand.lower()
                cond_cand = cand_lower.replace(' ', '')
                if not cond_cand:
                    continue
                log_file.write(f"Trying candidate: '{name_cand}' (condensed: '{cond_cand}')\n")
                
                ref_len = len(cond_cand)
                thresh = max(1, ref_len // 2)
                matches = []
                for cond_ref, orig in reference_list:
                    dist = levenshtein(cond_cand, cond_ref)
                    if dist <= thresh:
                        matches.append((orig, dist))
                if matches:
                    log_file.write(f"Matches found: {[(orig, dist) for orig, dist in matches]}\n")
                    min_dist = min(m[1] for m in matches)
                    best_candidates = [m[0] for m in matches if m[1] == min_dist]
                    best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                    score = fuzz.ratio(best_orig.lower(), cand_lower)
                    log_file.write(f"Best candidate: '{best_orig}', Fuzz ratio: {score}, Distance: {min_dist}\n")
                    if score >= 80:
                        all_matches.append((best_orig, score, min_dist, num_name_words + 1))
                        if score == 100 and num_name_words >= 2:
                            log_file.write(f"Exact match found: '{best_orig}', consuming {num_name_words + 1} tokens\n")
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
                log_file.write(f"Trying prefix '{prefix_cand}' with candidate: '{name_cand}'\n")
            else:
                num_tokens = num_name_words
                if len(tokens) < num_tokens:
                    continue
                name_cand = ' '.join(tokens[-num_name_words:])
                cand_lower = name_cand.lower()
                cand_stripped = cand_lower
                log_file.write(f"Trying no-prefix candidate: '{name_cand}'\n")
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
                            log_file.write(f"Matches found: {[(orig, dist) for orig, dist in matches]}\n")
                            min_dist = min(m[1] for m in matches)
                            best_candidates = [m[0] for m in matches if m[1] == min_dist]
                            best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                            score = fuzz.ratio(best_orig.lower(), cand_lower)
                            log_file.write(f"Best candidate: '{best_orig}', Fuzz ratio: {score}, Distance: {min_dist}\n")
                            if score >= 80:
                                all_matches.append((best_orig, score, min_dist, num_tokens))
                                if score == 100 and num_name_words >= 2:
                                    log_file.write(f"Exact match found: '{best_orig}', consuming {num_tokens} tokens\n")
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
                log_file.write(f"Matches found: {[(orig, dist) for orig, dist in matches]}\n")
                min_dist = min(m[1] for m in matches)
                best_candidates = [m[0] for m in matches if m[1] == min_dist]
                best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                score = fuzz.ratio(best_orig.lower(), cand_lower)
                log_file.write(f"Best candidate: '{best_orig}', Fuzz ratio: {score}, Distance: {min_dist}\n")
                if score >= 80:
                    all_matches.append((best_orig, score, min_dist, num_tokens))
                    if score == 100 and num_name_words >= 2:
                        log_file.write(f"Exact match found: '{best_orig}', consuming {num_tokens} tokens\n")
                        return best_orig, num_tokens

    if all_matches:
        best_match = max(all_matches, key=lambda x: (x[3], x[1], -x[2]))
        log_file.write(f"Selected best match: '{best_match[0]}', consuming {best_match[3]} tokens\n")
        return best_match[0], best_match[3]

    log_file.write("No match found\n")
    return None, 0

# Extract address components from text
def extract_address(text: str) -> Dict[str, str]:
    """Extract province, district, and ward from a Vietnamese address string.
    
    Args:
        text: Input address string.
    
    Returns:
        Dictionary with province, district, and ward (empty strings if not found).
    """
    log_file.write(f"\nProcessing address: '{text}'\n")
    # Preprocess: lowercase, replace punctuation with spaces, normalize whitespace
    text = text.lower()
    log_file.write(f"Lowercase: '{text}'\n")
    text = re.sub(r'[.,/-]', ' ', text)
    log_file.write(f"After punctuation removal: '{text}'\n")
    text = re.sub(r'\s+', ' ', text).strip()
    log_file.write(f"Normalized whitespace: '{text}'\n")
    tokens = text.split()
    log_file.write(f"Tokens: {tokens}\n")

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
        log_file.write(f"\nAttempting to match {level_name}\n")
        match, num_pop = find_match(tokens, reference_list, prefixes)
        if match:
            log_file.write(f"Matched {level_name}: '{match}', consumed {num_pop} tokens\n")
            result[level_name] = match
            tokens = tokens[:-num_pop]
            log_file.write(f"Remaining tokens: {tokens}\n")
        else:
            log_file.write(f"No match found for {level_name}\n")

    log_file.write(f"\nExtracted result: {result}\n")
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
    log_file.write("\nManual Test Case:\n")
    log_file.write(f"Input: {manual_test['text']}\n")
    actual = extract_address(manual_test['text'])
    expected = manual_test['result']
    is_correct = actual == expected
    log_file.write(f"Expected: {expected}\n")
    log_file.write(f"Actual: {actual}\n")
    log_file.write(f"Correct: {is_correct}\n")
    if not is_correct:
        log_file.write("Differences:\n")
        for key in ["province", "district", "ward"]:
            if expected[key] != actual[key]:
                log_file.write(f"  {key.capitalize()}: Expected '{expected[key]}', got '{actual[key]}'\n")

# Run tests if data files are loaded successfully
if __name__ == "__main__":
    if not provinces or not districts or not wards:
        print("Error: One or more data files failed to load.")
    else:
        test_address_extraction()
    log_file.close()