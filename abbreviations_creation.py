import re
import json
import pandas as pd
from typing import Dict, Set, Tuple, List

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
        results = []
        self._search_recursive(self.root, word, 0, "", max_dist, results)
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
            # Match (substitution or exact)
            cost = 0 if char == next_char else 1
            self._search_recursive(child, word, pos + 1, current + next_char, max_dist - cost, results)
            # Insertion (skip char in word)
            if char is not None:
                self._search_recursive(child, word, pos + 1, current + next_char, max_dist - 1, results)
        # Deletion (skip char in trie)
        self._search_recursive(node, word, pos, current, max_dist - 1, results)

# Levenshtein distance for fuzzy matching (kept for reference)
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

# Create tries instead of condensed maps
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
def find_match(tokens: list[str], trie: Trie, prefixes: list[str], level_name: str) -> Tuple[str | None, int]:
    print(f"\n--- Matching {level_name} ---")
    print(f"Input tokens: {tokens}")
    for has_separate_prefix in [True, False]:
        prefix_type = "separate" if has_separate_prefix else "concatenated/no"
        print(f"Trying prefix type: {prefix_type}")
        for num_name_words in range(1, 5):  # 1 to 4 as requested
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
                        cand_stripped = cand_stripped[len(prefix):].strip()
                        break

            cond_cand = cand_stripped.replace(' ', '')
            print(f"    Condensed candidate: {cond_cand}")
            if not cond_cand:
                print(f"    Empty candidate, skipping")
                continue

            # Use trie for fuzzy matching
            ref_len = len(cond_cand)
            thresh = 0 if ref_len <= 3 else (ref_len // 3)
            matches = trie.find_matches(cond_cand, thresh)
            if matches:
                best_orig, best_dist = min(matches, key=lambda x: x[1])
                print(f"    Match found: {best_orig} (distance={best_dist}, threshold={thresh})")
                print(f"    Selected match: {best_orig}, consuming {num_tokens} tokens")
                return best_orig, num_tokens

    print(f"No match found for {level_name}")
    return None, 0

# Extract address components
def extract_address(text: str) -> Dict[str, str]:
    print(f"\n=== Processing Input: {text} ===")
    # Step 1: Convert to lowercase
    text = text.lower()
    print(f"Step 1 - After lowercase: {text}")

    # Step 2: Replace punctuation with spaces
    text = re.sub(r'[.,/-]', ' ', text)
    print(f"Step 2 - After punctuation replacement: {text}")

    # Step 3: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"Step 3 - After whitespace normalization: {text}")

    # Step 4: Tokenize
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
            tokens = tokens[:-num_pop]
            print(f"Updated tokens after {level_name}: {tokens}")

    print(f"Final result: {result}")
    return result

# Testing with first 50 test cases from public.json
def test_address_extraction():
    test_cases = [
        {
            "text": "357/28,Ng-T- Thuật,P1,Q3,TP.HồChíMinh.",
            "result": {"province": "Hồ Chí Minh", "district": "", "ward": ""}
        },
        {
            "text": "TT Tân Bình Huyện Yên Sơn, Tuyên Quang",
            "result": {"province": "Tuyên Quang", "district": "Yên Sơn", "ward": "Tân Bình"}
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
            "text": "T18,Cẩm Bình, Cẩm Phả, Quảng Ninh.",
            "result": {"province": "Quảng Ninh", "district": "Cẩm Phả", "ward": "Cẩm Bình"}
        }
        # Add remaining 45 test cases from public.json here
    ]

    correct = 0
    total = min(len(test_cases), 50)  # Limit to 50 test cases
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
        # Break after first case for detailed demonstration
        # if i == 1:
        #     break

    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")

# Run tests
if __name__ == "__main__":
    if not provinces or not districts or not wards:
        print("Error: One or more data files failed to load.")
    else:
        test_address_extraction()