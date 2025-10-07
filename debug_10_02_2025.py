import re
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional
import unicodedata
from fuzzywuzzy import fuzz
import json
import time

# Helper: Check if string has Vietnamese accents
def has_accents(s: str) -> bool:
    return any('WITH' in unicodedata.name(c, '') for c in s if c.isalpha())

# Tie-breaker: Prefer accented names, then by fuzz ratio
def break_ties(candidates: List[str], original: str) -> str:
    accented = [c for c in candidates if has_accents(c)]
    if accented:
        candidates = accented
    candidates.sort(key=lambda c: fuzz.ratio(c.lower(), original), reverse=True)
    return candidates[0]

# Levenshtein distance
def levenshtein(s1: str, s2: str) -> int:
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

# Load names from Excel file
def load_data(file_path: str, sheet_name: str) -> Set[str]:
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)
        names = df[0].dropna().tolist()
        return {name.strip() for name in names if name.strip()}
    except Exception as e:
        return set()

# Load data from Excel files
provinces: Set[str] = load_data('province.xlsx', sheet_name='provinces')
districts: Set[str] = load_data('district.xlsx', sheet_name='districts')
wards: Set[str] = load_data('ward.xlsx', sheet_name='wards')

# Create lists of (condensed, original)
def create_list(names: Set[str]) -> List[Tuple[str, str]]:
    return [(name.lower().replace(' ', ''), name) for name in names]

province_list = create_list(provinces)
district_list = create_list(districts)
ward_list = create_list(wards)

# Define prefixes for administrative levels (lowercase for matching)
province_prefixes = ['', 'thành phố', 'thanh pho', 'thanhpho', 'tp', 'tp.', 't.p', 'th', 't', 'than', 'thanh', 'tph', 'thnàh phố', 'thànhphố', 'tỉnh', 'tinh', 't.', 'tí', 'tinhf', 'tin', 'tnh', 't.phố', 't phố', 't.pho', 't pho']
district_prefixes = ['', 'quận', 'quan', 'q', 'q.', 'qu', 'qaun', 'qận', 'huyện', 'huyen', 'h', 'h.', 'hu', 'hyuen', 'huyệ', 'thị xã', 'thi xa', 'thixa', 'tx', 'tx.', 't.x', 'th', 'txa', 'thịxa', 'thành phố', 'thanh pho', 'thanhpho', 'tp', 'tp.', 't.p', 'than', 'thanh', 'tph', 'thnàh phố', 'thànhphố', 't.xã', 't xã', 't.xa', 't xa', 't.phố', 't phố', 't.pho', 't pho']
ward_prefixes = ['', 'phường', 'phuong', 'p', 'p.', 'ph', 'ph.', 'phuờng', 'phưng', 'puong', 'xã', 'xa', 'x', 'x.', 'xá', 'xạ', 'xãa', 'thị trấn', 'thi tran', 'thitran', 'tt', 'tt.', 't.t', 'th', 'ttr', 'thịtrấn', 't.trấn', 't tran', 't trấn', 't.tran']

# Find best match for a level (province, district, ward)
def find_match(tokens: List[str], reference_list: List[Tuple[str, str]], prefixes: List[str]) -> Tuple[Optional[str], int]:
    all_matches: List[Tuple[str, int, int, int]] = []

    # Try prefix-driven matching
    for i in range(len(tokens)):
        token = tokens[i].lower()
        if token in [p.lower() for p in prefixes if p]:
            max_words = min(4, len(tokens) - i - 1)
            for num_name_words in range(max_words, 0, -1):
                name_cand = ' '.join(tokens[i + 1:i + 1 + num_name_words])
                cand_lower = name_cand.lower()
                cond_cand = cand_lower.replace(' ', '')
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
                        all_matches.append((best_orig, score, min_dist, num_name_words + 1))
                        if score == 100 and num_name_words >= 2:
                            return best_orig, num_name_words + 1

    # Fallback to concatenated/no prefix logic
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

# Extract address components
def extract_address(text: str) -> Dict[str, str]:
    text = text.lower()
    text = re.sub(r'[.,/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()

    result: Dict[str, str] = {
        "province": "",
        "district": "",
        "ward": ""
    }

    levels = [
        ("province", province_list, province_prefixes),
        ("district", district_list, district_prefixes),
        ("ward", ward_list, ward_prefixes)
    ]

    for level_name, reference_list, prefixes in levels:
        match, num_pop = find_match(tokens, reference_list, prefixes)
        if match:
            result[level_name] = match
            tokens = tokens[:-num_pop]

    return result

# Testing with test cases from test.json
def test_address_extraction():
    try:
        with open('tests.json', 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading test.json: {e}")
        return

    total = len(test_cases)
    print(f"Total test cases: {total}")

    correct = 0
    total_time = 0.0
    test_times = []

    # Open log file for failed test cases
    with open('failed_test_cases.log', 'w', encoding='utf-8') as log_file:
        for i, case in enumerate(test_cases, 1):
            text = case["text"]
            expected = case["result"]

            # Measure execution time for this test case
            start_time = time.time()
            actual = extract_address(text)
            end_time = time.time()

            test_time = end_time - start_time
            test_times.append(test_time)
            total_time += test_time

            is_correct = actual == expected
            if is_correct:
                correct += 1
            else:
                # Log failed test case
                log_file.write(f"\nTest Case {i}:\n")
                log_file.write(f"Input: {text}\n")
                log_file.write(f"Expected: {expected}\n")
                log_file.write(f"Actual: {actual}\n")
                log_file.write("Differences:\n")
                for key in ["province", "district", "ward"]:
                    if expected[key] != actual[key]:
                        log_file.write(f"  {key.capitalize()}: Expected '{expected[key]}', got '{actual[key]}'\n")
                log_file.write(f"Notes: {case.get('notes', 'N/A')}\n")

    # Calculate average time per test case
    avg_time = total_time / total if total > 0 else 0

    # Print results
    print(f"{correct}/{total} test cases passed")
    print(f"Failed test cases: {total - correct}")
    print(f"Average time per test case: {avg_time:.6f} seconds")
    print(f"Total execution time: {total_time:.6f} seconds")

# Run tests
if __name__ == "__main__":
    if not provinces or not districts or not wards:
        print("Error: One or more data files failed to load.")
    else:
        test_address_extraction()