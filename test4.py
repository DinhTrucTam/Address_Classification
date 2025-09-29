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

# Trie Node
class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.original: str = None

# Trie for fuzzy matching
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, original: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.original = original

    def find_matches(self, word: str, max_dist: int) -> List[Tuple[str, int]]:
        print(f"  Searching trie for: {word}, max_dist={max_dist}")
        results = []
        self._search_recursive(self.root, word, 0, "", max_dist, results)
        # print(f"  Trie results: {results}")
        return results

    def _search_recursive(self, node: TrieNode, word: str, pos: int, current: str, max_dist: int, results: List[Tuple[str, int]]):
        if pos == len(word):
            if node.is_end and max_dist >= 0:
                results.append((node.original, max_dist))
            return
        if max_dist < 0:
            return
        char = word[pos] if pos < len(word) else None
        for next_char, child in node.children.items():
            cost = 0 if char == next_char else 1
            self._search_recursive(child, word, pos + 1, current + next_char, max_dist - cost, results)
            if char is not None:
                self._search_recursive(child, word, pos + 1, current + next_char, max_dist - 1, results)
        self._search_recursive(node, word, pos, current, max_dist - 1, results)

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
        print(f"Error loading {file_path} (sheet: {sheet_name}): {e}")
        return set()

# Load data from Excel files
provinces: Set[str] = load_data('province.xlsx', sheet_name='provinces')
districts: Set[str] = load_data('district.xlsx', sheet_name='districts')
wards: Set[str] = load_data('ward.xlsx', sheet_name='wards')

# Debug: Print loaded districts
print("Loaded districts:", sorted(districts))

# Create tries
def create_trie(names: Set[str]) -> Trie:
    trie = Trie()
    for name in names:
        condensed = name.lower().replace(' ', '')
        trie.insert(condensed, name)
    return trie

province_trie = create_trie(provinces)
district_trie = create_trie(districts)
ward_trie = create_trie(wards)

# Prefix lists (lowercase for matching)
province_prefixes = ['', 'thành phố', 'tp', 'tp.', 't.p', 'tỉnh', 't', 't.', 'tinh', 'thanhpho', 'tinh']
district_prefixes = ['', 'quận', 'q', 'q.', 'huyện', 'h', 'h.', 'thị xã', 'tx', 'tx.', 't.x', 'thành phố', 'tp', 'tp.', 'quan', 'huyen', 'thixa', 'thanhpho']
ward_prefixes = ['', 'phường', 'p', 'p.', 'xã', 'x', 'x.', 'thị trấn', 'tt', 'tt.', 't.t', 'phuong', 'xa', 'thitran']

# Find best match for a level (province, district, ward)
def find_match(tokens: List[str], trie: Trie, prefixes: List[str], level_name: str) -> Tuple[Optional[str], int]:
    print(f"\n--- Matching {level_name} ---")
    print(f"Input tokens: {tokens}")

    # Try prefix-driven matching, starting with all tokens after prefix
    for i in range(len(tokens)):
        token = tokens[i].lower()
        if token in [p.lower() for p in prefixes if p]:
            print(f"Found prefix '{token}' at index {i} for {level_name}")
            best_score = 0
            best_match = None
            best_num_tokens = 0
            # Try from max tokens to 1
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
                thresh = max(1, ref_len // 2)  # Increased threshold for flexibility
                matches = trie.find_matches(cond_cand, thresh)
                if matches:
                    min_dist = min(m[1] for m in matches)
                    best_candidates = [m[0] for m in matches if m[1] == min_dist]
                    print(f"    Best candidates (distance={min_dist}): {best_candidates}")
                    best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                    score = fuzz.ratio(best_orig.lower(), cand_lower)
                    print(f"    Fuzz ratio check: {best_orig} (score={score})")
                    if score >= 80 and score > best_score:
                        best_score = score
                        best_match = best_orig
                        best_num_tokens = num_name_words + 1
            if best_match:
                print(f"    Best match found: {best_match} (fuzz ratio={best_score})")
                return best_match, best_num_tokens

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
                        matches = trie.find_matches(cand_lower.replace(' ', ''), thresh)
                        if matches:
                            min_dist = min(m[1] for m in matches)
                            best_candidates = [m[0] for m in matches if m[1] == min_dist]
                            best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                            score = fuzz.ratio(best_orig.lower(), cand_lower)
                            print(f"    Fuzz ratio check: {best_orig} (score={score})")
                            if score >= 80:
                                print(f"    ✅ High-confidence match: {best_orig}")
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
            matches = trie.find_matches(cond_cand, thresh)
            if matches:
                min_dist = min(m[1] for m in matches)
                best_candidates = [m[0] for m in matches if m[1] == min_dist]
                # print(f"    Best candidates (distance={min_dist}): {best_candidates}")
                best_orig = max(best_candidates, key=lambda c: fuzz.ratio(c.lower(), cand_lower))
                score = fuzz.ratio(best_orig.lower(), cand_lower)
                print(f"    Fuzz ratio check: {best_orig} (score={score})")
                if score >= 80:
                    print(f"    ✅ High-confidence match: {best_orig}")
                    return best_orig, num_tokens
                print(f"    Low fuzz ratio, continuing")

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
        ("province", province_trie, province_prefixes),
        ("district", district_trie, district_prefixes),
        ("ward", ward_trie, ward_prefixes)
    ]

    for level_name, trie, prefixes in levels:
        match, num_pop = find_match(tokens, trie, prefixes, level_name)
        if match:
            result[level_name] = match
            tokens = tokens[:-num_pop]  # Remove tokens from end
            print(f"Updated tokens after {level_name}: {tokens}")

    print(f"Final result: {result}")
    return result

# Testing with first 50 test cases from public.json
def test_address_extraction():
    test_cases = [
        {
            "text": "TT Tân Bình Huyện YYên Sơn, Ruyên Quang",
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
        if i == 1:
            break

    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")

# Run tests
if __name__ == "__main__":
    if not provinces or not districts or not wards:
        print("Error: One or more data files failed to load.")
    else:
        test_address_extraction()