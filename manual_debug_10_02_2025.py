import re
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional
import unicodedata
from fuzzywuzzy import fuzz
import time
import json

# Helper: Check if string has Vietnamese accents
def has_accents(s: str) -> bool:
    return any('WITH' in unicodedata.name(c, '') for c in s if c.isalpha())

# Tie-breaker: Prefer exact numeric matches, then accented names, then fuzz ratio
def break_ties(candidates: List[str], original: str) -> str:
    # If original is numeric, prioritize exact numeric match
    if re.match(r'^\d+$', original.strip()):
        for cand in candidates:
            if cand == original.strip():
                print(f"Break_ties: Selected exact numeric match '{cand}' for original '{original}'")
                return cand
    # Otherwise, prefer accented names
    accented = [c for c in candidates if has_accents(c)]
    if accented:
        candidates = accented
    candidates.sort(key=lambda c: fuzz.ratio(c.lower(), original), reverse=True)
    print(f"Break_ties: Selected '{candidates[0]}' for original '{original}', fuzz ratio: {fuzz.ratio(candidates[0].lower(), original)}")
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
        names = {name.strip() for name in names if name.strip()}
        print(f"Loaded {len(names)} names from {file_path} ({sheet_name})")
        return names
    except Exception as e:
        print(f"Error loading {file_path} ({sheet_name}): {e}")
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
province_prefixes = ['', 'thành phố', 'thanh pho', 'thanhpho', 'tp', 'tp.', 't.p', 't', 'thnàh phố', 'thànhphố', 'tỉnh', 'tinh', 't.', 'tí', 'tinhf', 'tin', 'tnh', 't.phố', 't phố', 't.pho', 't pho']
district_prefixes = ['', 'quận', 'quan', 'q', 'q.', 'qu', 'qaun', 'qận', 'huyện', 'huyen', 'h', 'h.', 'hu', 'hyuen', 'huyệ', 'thị xã', 'thi xa', 'thixa', 'tx', 'tx.', 't.x', 'th', 'txa', 'thịxa', 'thành phố', 'thanh pho', 'thanhpho', 'tp', 'tp.', 't.p', 'than', 'thanh', 'tph', 'thnàh phố', 'thànhphố', 't.xã', 't xã', 't.xa', 't xa', 't.phố', 't phố', 't.pho', 't pho']
ward_prefixes = ['', 'phường', 'phuong', 'p', 'p.', 'ph', 'ph.', 'phuờng', 'phưng', 'puong', 'xã', 'xa', 'x', 'x.', 'xá', 'xạ', 'xãa', 'thị trấn', 'thi tran', 'thitran', 'tt', 'tt.', 't.t', 'th', 'ttr', 'thịtrấn', 't.trấn', 't tran', 't trấn', 't.tran']

# Find best match for a level (province, district, ward)
# def find_match(tokens: List[str], reference_list: List[Tuple[str, str]], prefixes: List[str], level: str) -> Tuple[Optional[str], int]:
#     print(f"\nMatching for {level} with tokens: {tokens}")
#     all_matches: List[Tuple[str, int, int, int]] = []

#     # Try prefix-driven matching
#     for i in range(len(tokens)):
#         token = tokens[i].lower()
#         if token in [p.lower() for p in prefixes if p]:
#             max_words = min(4, len(tokens) - i - 1)
#             for num_name_words in range(max_words, 0, -1):
#                 name_cand = ' '.join(tokens[i + 1:i + 1 + num_name_words])
#                 cand_lower = name_cand.lower()
#                 cond_cand = cand_lower.replace(' ', '')
#                 if not cond_cand:
#                     continue
#                 print(f"Trying candidate for {level}: '{name_cand}' (condensed: '{cond_cand}')")

#                 # Special handling for numeric wards
#                 if level == "ward" and re.match(r'^\d+$', cand_lower.strip()):
#                     if cand_lower.strip() in [orig for _, orig in reference_list]:
#                         print(f"Exact numeric ward match: '{cand_lower.strip()}'")
#                         return cand_lower.strip(), num_name_words + 1

#                 ref_len = len(cond_cand)
#                 thresh = max(1, ref_len // 2)
#                 matches = []
#                 for cond_ref, orig in reference_list:
#                     dist = levenshtein(cond_cand, cond_ref)
#                     if dist <= thresh:
#                         matches.append((orig, dist))
#                 if matches:
#                     min_dist = min(m[1] for m in matches)
#                     best_candidates = [m[0] for m in matches if m[1] == min_dist]
#                     best_orig = break_ties(best_candidates, cand_lower)
#                     score = fuzz.ratio(best_orig.lower(), cand_lower)
#                     print(f"Match for {level}: '{best_orig}', fuzz ratio: {score}, distance: {min_dist}")
#                     if score >= 80:
#                         all_matches.append((best_orig, score, min_dist, num_name_words + 1))
#                         if score == 100 and num_name_words >= 2:
#                             print(f"Exact match for {level}: '{best_orig}', consuming {num_name_words + 1} tokens")
#                             return best_orig, num_name_words + 1

#     # Fallback to concatenated/no prefix logic
#     for has_separate_prefix in [True, False]:
#         for num_name_words in range(1, 5):
#             if has_separate_prefix:
#                 num_tokens = num_name_words + 1
#                 if len(tokens) < num_tokens:
#                     continue
#                 prefix_cand = tokens[-num_tokens].lower()
#                 if prefix_cand not in [p.lower() for p in prefixes if p]:
#                     continue
#                 name_cand = ' '.join(tokens[-num_name_words:])
#                 cand_lower = name_cand.lower()
#                 cand_stripped = cand_lower
#                 print(f"Trying prefix '{prefix_cand}' with candidate for {level}: '{name_cand}'")
#             else:
#                 num_tokens = num_name_words
#                 if len(tokens) < num_tokens:
#                     continue
#                 name_cand = ' '.join(tokens[-num_name_words:])
#                 cand_lower = name_cand.lower()
#                 cand_stripped = cand_lower
#                 print(f"Trying no-prefix candidate for {level}: '{name_cand}'")
#                 for prefix in [p.lower().replace(' ', '') for p in prefixes if p]:
#                     if cand_stripped.startswith(prefix):
#                         cand_stripped = cand_stripped[len(prefix):].strip()
#                         break

#             cond_cand = cand_stripped.replace(' ', '')
#             if not cond_cand:
#                 continue

#             ref_len = len(cond_cand)
#             thresh = max(1, ref_len // 2)
#             matches = []
#             for cond_ref, orig in reference_list:
#                 dist = levenshtein(cond_cand, cond_ref)
#                 if dist <= thresh:
#                     matches.append((orig, dist))
#             if matches:
#                 min_dist = min(m[1] for m in matches)
#                 best_candidates = [m[0] for m in matches if m[1] == min_dist]
#                 best_orig = break_ties(best_candidates, cand_lower)
#                 score = fuzz.ratio(best_orig.lower(), cand_lower)
#                 print(f"Match for {level}: '{best_orig}', fuzz ratio: {score}, distance: {min_dist}")
#                 if score >= 70:
#                     all_matches.append((best_orig, score, min_dist, num_tokens))
#                     if score == 100 and num_name_words >= 2:
#                         print(f"Exact match for {level}: '{best_orig}', consuming {num_tokens} tokens")
#                         return best_orig, num_tokens

#     if all_matches:
#         best_match = max(all_matches, key=lambda x: (x[3], x[1], -x[2]))
#         print(f"Selected best match for {level}: '{best_match[0]}', consuming {best_match[3]} tokens")
#         return best_match[0], best_match[3]

#     print(f"No match found for {level}")
#     return None, 0

def find_match(tokens: List[str], reference_list: List[Tuple[str, str]], prefixes: List[str], level: str) -> Tuple[Optional[str], int]:
    print(f"\nMatching for {level} with tokens: {tokens}")
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
                print(f"Trying candidate for {level}: '{name_cand}' (condensed: '{cond_cand}')")

                # Special handling for numeric wards
                if level == "ward" and re.match(r'^\d+$', cand_lower.strip()):
                    if cand_lower.strip() in [orig for _, orig in reference_list]:
                        print(f"Exact numeric ward match: '{cand_lower.strip()}'")
                        return cand_lower.strip(), num_name_words + 1

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
                    best_orig = break_ties(best_candidates, cand_lower)
                    score = fuzz.ratio(best_orig.lower(), cand_lower)
                    print(f"Match for {level}: '{best_orig}', fuzz ratio: {score}, distance: {min_dist}")
                    if score >= 80:
                        all_matches.append((best_orig, score, min_dist, num_name_words + 1))
                        if score == 100 and num_name_words >= 2:
                            print(f"Exact match for {level}: '{best_orig}', consuming {num_name_words + 1} tokens")
                            return best_orig, num_name_words + 1

    # Fallback to no-prefix logic, but limit to tokens after the last prefix
    last_prefix_idx = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].lower() in [p.lower() for p in prefixes if p]:
            last_prefix_idx = i
            break
    for num_name_words in range(1, min(4, len(tokens) - last_prefix_idx) + 1):
        name_cand = ' '.join(tokens[-num_name_words:])
        cand_lower = name_cand.lower()
        cand_stripped = cand_lower
        print(f"Trying no-prefix candidate for {level}: '{name_cand}'")
        for prefix in [p.lower().replace(' ', '') for p in prefixes if p]:
            if cand_stripped.startswith(prefix):
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
            best_orig = break_ties(best_candidates, cand_lower)
            score = fuzz.ratio(best_orig.lower(), cand_lower)
            print(f"Match for {level}: '{best_orig}', fuzz ratio: {score}, distance: {min_dist}")
            if score >= 80:  # Stricter threshold for no-prefix matches
                all_matches.append((best_orig, score, min_dist, num_name_words))

    if all_matches:
        # Prioritize higher fuzz ratio, then fewer tokens, then lower distance
        best_match = max(all_matches, key=lambda x: (x[1], -x[3], -x[2]))
        print(f"Selected best match for {level}: '{best_match[0]}', consuming {best_match[3]} tokens")
        return best_match[0], best_match[3]

    print(f"No match found for {level}")
    return None, 0

# Extract address components
def extract_address(text: str) -> Dict[str, str]:
    print(f"\nProcessing address: '{text}'")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"Normalized whitespace: '{text}'")
    # Split by commas first
    segments = [segment.strip() for segment in text.split(',') if segment.strip()]
    print(f"Comma-separated segments: {segments}")

    # Initialize prefix tracking
    prefix_counts = {'province': 0, 'district': 0, 'ward': 0}
    segment_prefixes = []  # Store (segment, prefix_type, matched_prefix) for each segment
    tokens = []

    # Define prefix lists to check hierarchically
    prefix_lists = {
        'province': sorted([p.lower().replace(' ', '') for p in province_prefixes if p], key=len, reverse=True),
        'district': sorted([p.lower().replace(' ', '') for p in district_prefixes if p], key=len, reverse=True),
        'ward': sorted([p.lower().replace(' ', '') for p in ward_prefixes if p], key=len, reverse=True)
    }

    # Track expected level based on hierarchy
    expected_levels = ['province', 'district', 'ward']  # Order to check
    current_level_idx = 0  # Start with province

    # Process segments in reverse order
    for segment in reversed(segments):
        segment_lower = segment.lower()
        segment_condensed = segment_lower.replace(' ', '')
        segment_tokens = []
        prefix_found = False
        prefix_type = None
        matched_prefix = None

        # Try all levels, prioritizing the current expected level
        levels_to_check = [expected_levels[current_level_idx]] + [level for level in expected_levels if level != expected_levels[current_level_idx]]
        for level in levels_to_check:
            prefixes_to_check = prefix_lists[level]
            for prefix in prefixes_to_check:
                if segment_condensed.startswith(prefix):
                    prefix_with_spaces = next(
                        (p for p in (province_prefixes if level == 'province' else
                                    district_prefixes if level == 'district' else
                                    ward_prefixes)
                        if p.lower().replace(' ', '') == prefix),
                        prefix
                    )
                    if segment_lower.startswith(prefix_with_spaces.lower()):
                        prefix_type = level
                        prefix_counts[prefix_type] += 1
                        matched_prefix = prefix_with_spaces
                        name_part = segment[len(prefix_with_spaces):].strip()
                        if not name_part and segment_condensed[len(prefix):].isdigit():
                            name_part = segment_condensed[len(prefix):]  # Handle numeric wards like 'p3'
                        if name_part:
                            segment_tokens.append(prefix_with_spaces.lower().replace(' ', ''))
                            segment_tokens.extend(name_part.split() if ' ' in name_part else [name_part])
                            print(f"Split segment '{segment}' into prefix '{prefix_with_spaces}' and name '{name_part}'")
                        else:
                            segment_tokens.append(prefix_with_spaces.lower().replace(' ', ''))
                            print(f"Split segment '{segment}' into prefix '{prefix_with_spaces}' (no name part)")
                        prefix_found = True
                        # Move to the next level if matched the current expected level
                        if level == expected_levels[current_level_idx] and current_level_idx < len(expected_levels) - 1:
                            current_level_idx += 1
                        break
            if prefix_found:
                break

        if not prefix_found:
            segment_tokens = segment.split()
            print(f"No prefix found in segment '{segment}', tokens: {segment_tokens}")
            # Only reset to province if no prefixes have been found
            if sum(prefix_counts.values()) == 0:
                current_level_idx = 0
            # Stay at current level or move back if the previous level wasn't matched
            elif current_level_idx > 0 and prefix_counts[expected_levels[current_level_idx - 1]] == 0:
                current_level_idx -= 1

        segment_prefixes.append((segment, prefix_type, matched_prefix))
        tokens = segment_tokens + tokens

    # Print prefix counts
    total_prefixes = sum(prefix_counts.values())
    print(f"Prefix counts: {prefix_counts}, Total: {total_prefixes}")
    print(f"Final tokens: {tokens}")

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
            print(f"Matched {level_name}: '{match}', remaining tokens: {tokens}")
        else:
            print(f"No {level_name} matched, remaining tokens: {tokens}")

    print(f"Extracted result: {result}")
    return result


def test_address_extraction():
    # try:
    #     with open('tests.json', 'r', encoding='utf-8') as f:
    #         test_cases = json.load(f)
    # except Exception as e:
    #     print(f"Error loading test.json: {e}")
    #     return

    test_cases = [
    {
        "text": "XãQuỳn1 Lâm, H. Quỳnh Phụ,  Thai Bình",
        "result": {
            "province": "Thái Bình",
            "district": "Quỳnh Phụ",
            "ward": "Quỳnh Lâm"
        },
        "notes": "public test case"
    },
    ]

    total = len(test_cases)
    print(f"Total test cases: {total}")

    correct = 0
    total_time = 0.0
    test_times = []

    # Open log file for failed test cases
    with open('failed_test_cases.log', 'w', encoding='utf-8') as log_file:
        for i, case in enumerate(test_cases, 21):  # Start numbering at 21
            text = case["text"]
            expected = case["result"]
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {text}")
            print(f"Expected: {expected}")

            # Measure execution time for this test case
            start_time = time.time()
            actual = extract_address(text)
            end_time = time.time()

            test_time = end_time - start_time
            test_times.append(test_time)
            total_time += test_time

            is_correct = actual == expected
            print(f"Actual: {actual}")
            print(f"Correct: {is_correct}")
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
    print(f"\n--- Final Results ---")
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

#Only print out the result and steps to debug failed test cases in the log file "failed_test_cases.log"