from rapidfuzz import fuzz
# Check the similarity score
name = "Tiền Giang"
full_name = "t. giang"
full_name1 = "Tiền Giang"

print(f"Similarity score: {fuzz.ratio(name, full_name)}")
print(f"Similarity score: {fuzz.ratio(name, full_name1)}")