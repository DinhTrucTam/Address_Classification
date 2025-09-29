from rapidfuzz import fuzz
# Check the similarity score
name = "Đống Đa"
full_name = "Đ Đa"
full_name1 = "Đ.Đa,"

print(f"Similarity score: {fuzz.ratio(name, full_name)}")
print(f"Similarity score: {fuzz.ratio(name, full_name1)}")