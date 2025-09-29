import re
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional
import unicodedata
from fuzzywuzzy import fuzz

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
        deduped_names = {name.strip() for name in names if name.strip()}
        print(f"Loaded {len(names)} names, {len(deduped_names)} unique names from {file_path} (sheet: {sheet_name})")
        return deduped_names
    except Exception as e:
        print(f"Error loading {file_path} (sheet: {sheet_name}): {e}")
        return set()

# Load data from Excel files
provinces: Set[str] = load_data('province.xlsx', sheet_name='provinces')
districts: Set[str] = load_data('district.xlsx', sheet_name='districts')
wards: Set[str] = load_data('ward.xlsx', sheet_name='wards')

# Debug: Print loaded districts
print("Loaded districts:", sorted(districts))

# Create lists of (condensed, original)
def create_list(names: Set[str]) -> List[Tuple[str, str]]:
    return [(name.lower().replace(' ', ''), name) for name in names]

province_list = create_list(provinces)
district_list = create_list(districts)
ward_list = create_list(wards)

# Prefix lists (lowercase for matching)
province_prefixes = ['', 'thành phố', 'tp', 'tp.', 't.p', 'tỉnh', 't', 't.', 'tinh', 'thanhpho']
district_prefixes = ['', 'quận', 'q', 'q.', 'huyện', 'h', 'h.', 'thị xã', 'tx', 'tx.', 't.x', 'thành phố', 'tp', 'tp.', 'quan', 'huyen', 'thixa', 'thanhpho']
ward_prefixes = ['', 'phường', 'p', 'p.', 'xã', 'x', 'x.', 'thị trấn', 'tt', 'tt.', 't.t', 'phuong', 'xa', 'thitran']

# Find best match for a level (province, district, ward)
def find_match(tokens: List[str], reference_list: List[Tuple[str, str]], prefixes: List[str], level_name: str) -> Tuple[Optional[str], int]:
    print(f"\n--- Matching {level_name} ---")
    print(f"Input tokens: {tokens}")

    # Store all valid matches: (match, fuzz_score, levenshtein_dist, num_tokens)
    all_matches: List[Tuple[str, int, int, int]] = []

    # Try prefix-driven matching
    for i in range(len(tokens)):
        token = tokens[i].lower()
        if token in [p.lower() for p in prefixes if p]:
            print(f"Found prefix '{token}' at index {i} for {level_name}")
            max_words = min(4, len(tokens) - i - 1)
            for num_name_words in range(max_words, 0, -1):
                print(f"  Trying {num_name_words} name words after prefix")
                name_cand = ' '.join(tokens[i + 1:i + 1 + num_name_words])
                cand_lower = name_cand.lower()
                cond_cand = cand_lower.replace(' ', '')
                print(f"    Name candidate: {name_cand}")
                print(f"    Condensed candidate: {cond_cand}")
                if not cond_cand:
                    print(f"    Empty candidate, skipping")
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
                    print(f"    Best candidates (distance={min_dist}): {best_candidates}")
                    best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                    score = fuzz.ratio(best_orig.lower(), cand_lower)
                    print(f"    Fuzz ratio check: {best_orig} (score={score})")
                    if score >= 80:
                        all_matches.append((best_orig, score, min_dist, num_name_words + 1))
                        # Early return for perfect match with reasonable token count
                        if score == 100 and num_name_words >= 2:
                            print(f"    ✅ Perfect match found early: {best_orig} (fuzz ratio={score}, tokens={num_name_words + 1})")
                            return best_orig, num_name_words + 1

    # Fallback to concatenated/no prefix logic
    for has_separate_prefix in [True, False]:
        prefix_type = "separate" if has_separate_prefix else "concatenated/no"
        print(f"Trying prefix type: {prefix_type}")
        for num_name_words in range(1, 5):
            print(f"  Trying {num_name_words} name words")
            if has_separate_prefix:
                num_tokens = num_name_words + 1
                if len(tokens) < num_tokens:
                    print(f"    Not enough tokens ({len(tokens)} < {num_tokens})")
                    continue
                prefix_cand = tokens[-num_tokens].lower()
                print(f"    Prefix candidate: {prefix_cand}")
                if prefix_cand not in [p.lower() for p in prefixes if p]:
                    print(f"    Invalid prefix, skipping")
                    continue
                name_cand = ' '.join(tokens[-num_name_words:])
                cand_lower = name_cand.lower()
                cand_stripped = cand_lower
                print(f"    Name candidate: {name_cand}")
            else:
                num_tokens = num_name_words
                if len(tokens) < num_tokens:
                    print(f"    Not enough tokens ({len(tokens)} < {num_tokens})")
                    continue
                name_cand = ' '.join(tokens[-num_name_words:])
                cand_lower = name_cand.lower()
                cand_stripped = cand_lower
                print(f"    Name candidate: {name_cand}")
                for prefix in [p.lower().replace(' ', '') for p in prefixes if p]:
                    if cand_stripped.startswith(prefix):
                        print(f"    Found concatenated prefix: {prefix}")
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
                            print(f"    Fuzz ratio check: {best_orig} (score={score})")
                            if score >= 80:
                                all_matches.append((best_orig, score, min_dist, num_tokens))
                                if score == 100 and num_name_words >= 2:
                                    print(f"    ✅ Perfect match found early: {best_orig} (fuzz ratio={score}, tokens={num_tokens})")
                                    return best_orig, num_tokens
                        cand_stripped = cand_stripped[len(prefix):].strip()
                        break

            cond_cand = cand_stripped.replace(' ', '')
            print(f"    Condensed candidate: {cond_cand}")
            if not cond_cand:
                print(f"    Empty candidate, skipping")
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
                print(f"    Best candidates (distance={min_dist}): {best_candidates}")
                best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                score = fuzz.ratio(best_orig.lower(), cand_lower)
                print(f"    Fuzz ratio check: {best_orig} (score={score})")
                if score >= 80:
                    all_matches.append((best_orig, score, min_dist, num_tokens))
                    if score == 100 and num_name_words >= 2:
                        print(f"    ✅ Perfect match found early: {best_orig} (fuzz ratio={score}, tokens={num_tokens})")
                        return best_orig, num_tokens

    # Select best match from all_matches
    if all_matches:
        print(f"All matches: {all_matches}")
        # Sort by: (1) num_tokens (desc), (2) fuzz_score (desc), (3) levenshtein_dist (asc)
        best_match = max(all_matches, key=lambda x: (x[3], x[1], -x[2]))
        print(f"    ✅ Best match selected: {best_match[0]} (fuzz ratio={best_match[1]}, tokens={best_match[3]})")
        return best_match[0], best_match[3]

    print(f"No match found for {level_name}")
    return None, 0

# Extract address components
def extract_address(text: str) -> Dict[str, str]:
    print(f"\n=== Processing Input: {text} ===")
    text = text.lower()
    print(f"Step 1 - After lowercase: {text}")
    text = re.sub(r'[.,/-]', ' ', text)
    print(f"Step 2 - After punctuation replacement: {text}")
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"Step 3 - After whitespace normalization: {text}")
    tokens = text.split()
    print(f"Step 4 - Tokens: {tokens}")

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
        match, num_pop = find_match(tokens, reference_list, prefixes, level_name)
        if match:
            result[level_name] = match
            tokens = tokens[:-num_pop]
            print(f"Updated tokens after {level_name}: {tokens}")

    print(f"Final result: {result}")
    return result

# Testing with first 50 test cases from public.json
def test_address_extraction():
    test_cases = [
        {
            "text": "TT Tân Bình Huyện YYên Sơn, Tuyên Quang",
            "result": {"province": "Tuyên Quang", "district": "Yên Sơn", "ward": "Tân Bình"}
        },
        {
            "text": "TT Tân Bình Huyện KYên Sơn, Tuyên Quang",
            "result": {"province": "Tuyên Quang", "district": "Yên Sơn", "ward": "Tân Bình"}
        },
        {
            "text": "357/28,Ng-T- Thuật,P1,Q3,TP.HồChíMinh.",
            "result": {"province": "Hồ Chí Minh", "district": "", "ward": ""}
        },
        {
            "text": "284DBis Ng Văn Giáo, P3, Mỹ Tho, T.Giang.",
            "result": {"province": "Tiền Giang", "district": "Mỹ Tho", "ward": "3"}
        },
        {
            "text": ",H.Tuy An,Tinh Phú yên",
            "result": {"province": "Phú Yên", "district": "Tuy An", "ward": ""}
        },
        {
            "text": "T18,Cẩm Bonh, Cẩm Phả, Quảng Nomh.",
            "result": {"province": "Quảng Ninh", "district": "Cẩm Phả", "ward": "Cẩm Bình"}
        }
    ]

    correct = 0
    total = min(len(test_cases), 50)
    for i, case in enumerate(test_cases[:50], 1):
        print(f"\n=== Test Case {i} ===")
        text = case["text"]
        expected = case["result"]
        actual = extract_address(text)
        is_correct = actual == expected
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Correct: {is_correct}")
        if is_correct:
            correct += 1

    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")

# Run tests
if __name__ == "__main__":
    if not provinces or not districts or not wards:
        print("Error: One or more data files failed to load.")
    else:
        test_address_extraction()