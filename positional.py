import os
os.environ["PYSPARK_PYTHON"] = "C:\\Users\\doaaa\\anaconda3\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\doaaa\\anaconda3\\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, monotonically_increasing_id, col, regexp_extract
from pyspark.sql.functions import input_file_name
import re

spark = SparkSession.builder \
    .appName("PositionalIndex") \
    .master("local[*]") \
    .config("spark.local.dir", "E:/spark_temp") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.text("dataset_text_processing/*.txt")
df = df.withColumn(
    "filename",
    regexp_extract(input_file_name(), r"([^/\\]+\.txt)$", 1)
)
df = df.orderBy("filename")
df.show(truncate=False)

rdd = df.rdd.flatMap(lambda row: [(row["filename"], idx, word) for idx, word in enumerate(re.findall(r"\w+", row["value"].lower()))])

positional_index = (
    rdd.map(lambda x: (x[2], (x[0], x[1])))
       .groupByKey()
       .mapValues(list)
)

def format_output_pretty(term, postings):
    doc_positions = {}
    for doc, pos in postings:
        doc_positions.setdefault(doc, []).append(pos)
    result = f"Term: {term}\n"
    for doc, positions in doc_positions.items():
        result += f" {doc} -> positions: {positions}\n"
    return result

formatted_output_pretty = positional_index.map(lambda x: format_output_pretty(x[0], x[1]))

import shutil
shutil.rmtree("positional_index", ignore_errors=True)
for line in formatted_output_pretty.take(20):
    print(line)

lines = formatted_output_pretty.collect()
with open("positional_index.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

# part 2
import math
import re

input_file = "positional_index.txt"
term_docs = {}
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    term = None
    for line in lines:
        line = line.strip()
        if line.startswith("Term:"):
            term = line.split("Term:")[1].strip()
            term_docs[term] = {}
        elif line and term:
            parts = line.split("-> positions:")
            if len(parts) == 2:
                doc = parts[0].strip()
                positions = eval(parts[1].strip())
                term_docs[term][doc] = positions

docs = sorted({doc for postings in term_docs.values() for doc in postings.keys()}, key=lambda x: int(re.findall(r'\d+', x)[0]))

tf_matrix = {}
tf_raw_matrix = {}

for term, counts in term_docs.items():
    tf_matrix[term] = []
    tf_raw_matrix[term] = []
    for doc in docs:
        tf = len(counts.get(doc, []))
        tf_raw_matrix[term].append(tf)
        weight = int(1 + math.log10(tf)) if tf > 0 else 0
        tf_matrix[term].append(weight)

N = len(docs)
idf = {}
idf_table = []

for term, counts in tf_matrix.items():
    df = sum(1 for c in counts if c > 0)
    idf_value = math.log10(N / df) if df > 0 else 0
    idf[term] = idf_value
    idf_table.append([term, df, idf_value])

print("term\t df\t idf")
for row in idf_table:
    print(f"{row[0]}\t {row[1]}\t {row[2]}")

tfidf_matrix = {
    term: [tf * idf[term] for tf in counts]
    for term, counts in tf_matrix.items()
}

def save_ascii_table(filename, header, rows):
    col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([header] + rows))]
    with open(filename, "w", encoding="utf-8") as f:
        sep = "+" + "+".join("-"*w for w in col_widths) + "+"
        f.write(sep + "\n")
        f.write("|" + "|".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header))) + "|\n")
        f.write(sep + "\n")
        for row in rows:
            f.write("|" + "|".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + "|\n")
        f.write(sep + "\n")

tf_rows_raw = [[term] + [tf_raw_matrix[term][i] for i in range(len(docs))] for term in tf_raw_matrix]
save_ascii_table("TF_table.txt", ["Term"] + docs, tf_rows_raw)

tf_rows_weighted = [[term] + [tf_matrix[term][i] for i in range(len(docs))] for term in tf_matrix]
save_ascii_table("TF_weighted_table.txt", ["Term"] + docs, tf_rows_weighted)

idf_rows = [[term, df, idf[term]] for term, df, idf_value in idf_table]
save_ascii_table("IDF_table.txt", ["Term", "DF", "IDF"], idf_rows)

tfidf_rows = [[term] + [round(v, 5) for v in values] for term, values in tfidf_matrix.items()]
save_ascii_table("TFIDF_table.txt", ["Term"] + docs, tfidf_rows)

doc_lengths = {}
for i, doc in enumerate(docs):
    length = math.sqrt(sum((tfidf_matrix[term][i])**2 for term in tfidf_matrix))
    doc_lengths[doc] = length

with open("TFIDF_table.txt", "a", encoding="utf-8") as f:
    f.write("\n")
    for doc, length in doc_lengths.items():
        f.write(f"{doc} length {length}\n")

normalized_tfidf_matrix = {}
for term in tfidf_matrix:
    normalized_tfidf_matrix[term] = []
    for i, doc in enumerate(docs):
        length = doc_lengths[doc]
        if length > 0:
            normalized_value = tfidf_matrix[term][i] / length
        else:
            normalized_value = 0
        normalized_tfidf_matrix[term].append(round(normalized_value, 5))

normalized_rows = [[term] + values for term, values in normalized_tfidf_matrix.items()]
save_ascii_table("Normalized_TFIDF_table.txt", ["Term"] + docs, normalized_rows)

print("Saved TF, IDF, and TF-IDF tables as ASCII files:")
print(" - TF_table.txt")
print(" - IDF_table.txt")
print(" - TFIDF_table.txt")

# PART 3 — Phrase Query Searching
import re
import math

output_file = "query_results.txt"
BOOLEAN_OPS = ["AND NOT", "AND", "OR NOT", "OR"]

def words_in_document(term_docs, words, doc):
    for word in words:
        if word not in term_docs or doc not in term_docs[word]:
            return False
    return True

def search_query_words(term_docs, docs, query):
    words = re.findall(r"\w+", query.lower())
    words = [w for w in words if w.upper() not in BOOLEAN_OPS]
    matched_docs = set()
    for doc in docs:
        if words_in_document(term_docs, words, doc):
            matched_docs.add(doc)
    return matched_docs

def check_invalid_boolean(query):
    query_upper = query.upper()
    for op in BOOLEAN_OPS:
        if re.search(rf"\b{op}\b\s+\b{op}\b", query_upper):
            return f"Error: Boolean operator '{op}' is repeated consecutively."
    for op in BOOLEAN_OPS:
        if op.lower() in query and op not in query:
            return f"Error: Boolean operator '{op}' must be uppercase."
    return None

def evaluate_query(term_docs, docs, query):
    query = query.strip()
    q_upper = query.upper()
    if " AND NOT " in q_upper:
        p1, p2 = query.split(" AND NOT ")
        return search_query_words(term_docs, docs, p1) - search_query_words(term_docs, docs, p2)
    elif " AND " in q_upper:
        p1, p2 = query.split(" AND ")
        return search_query_words(term_docs, docs, p1) & search_query_words(term_docs, docs, p2)
    elif " OR NOT " in q_upper:
        p1, p2 = query.split(" OR NOT ")
        return search_query_words(term_docs, docs, p1) | (set(docs) - search_query_words(term_docs, docs, p2))
    elif " OR " in q_upper:
        p1, p2 = query.split(" OR ")
        return search_query_words(term_docs, docs, p1) | search_query_words(term_docs, docs, p2)
    else:
        return search_query_words(term_docs, docs, query)

def compute_query_tf(query_terms, vocabulary):
    query_terms = [t for t in query_terms if t.upper() not in BOOLEAN_OPS]
    return [query_terms.count(term) for term in vocabulary]

def save_table_aligned(filename, lines):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")

while True:
    query = input("\nEnter your query (type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break

    error_msg = check_invalid_boolean(query)
    if error_msg:
        print(error_msg)
        continue

    vocabulary = list(tf_matrix.keys())
    query_terms = re.findall(r"\w+", query.lower())
    query_tf = compute_query_tf(query_terms, vocabulary)
    query_tfidf = [query_tf[i] * idf[vocabulary[i]] for i in range(len(vocabulary))]

    query_length = math.sqrt(sum(val**2 for val in query_tfidf))
    normalized_query_tfidf = [round(val / query_length, 5) if query_length > 0 else 0 for val in query_tfidf]

    matched_docs = sorted(evaluate_query(term_docs, docs, query)) 

    product_matrix = {}
    sum_per_doc = {}
    similarities = {}
    for doc in matched_docs:
        doc_index = docs.index(doc)
        doc_vec = [tf_matrix[vocabulary[i]][doc_index] * idf[vocabulary[i]] for i in range(len(vocabulary))]
        doc_vec_norm = [val / doc_lengths[doc] if doc_lengths[doc] > 0 else 0 for val in doc_vec]
        product_values = [round(normalized_query_tfidf[i] * doc_vec_norm[i], 5) for i in range(len(vocabulary))]
        product_matrix[doc] = product_values
        sum_per_doc[doc] = round(sum(product_values), 5)
        similarities[doc] = sum(product_values)

    ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    col_width = 12
    header = ["Term", "tf_raw", "tf_weighted", "idf", "tf*idf", "normalized"] + matched_docs
    lines = ["".join(h.ljust(col_width) for h in header)]

    for term in query_terms:
        if term in vocabulary:
            idx = vocabulary.index(term)
            row = [
                term.ljust(col_width),
                str(query_tf[idx]).ljust(col_width),
                str(1 + int(math.log10(query_tf[idx])) if query_tf[idx] > 0 else 0).ljust(col_width),
                f"{idf[term]:.4f}".ljust(col_width),
                f"{query_tfidf[idx]:.4f}".ljust(col_width),
                f"{normalized_query_tfidf[idx]:.4f}".ljust(col_width)
            ]
            for doc in matched_docs:
                row.append(f"{product_matrix[doc][idx]:.4f}".ljust(col_width))
            lines.append("".join(row))

    sum_row = ["Sum".ljust(col_width)] + ["".ljust(col_width)] * 5
    for doc in matched_docs:
        sum_row.append(f"{sum_per_doc[doc]:.4f}".ljust(col_width))
    lines.append("".join(sum_row))

    lines.append(f"Query length: {round(query_length, 5)}")
    lines.append("Similarity per document:")
    for doc, sim in ranked_docs:
        lines.append(f"{doc.ljust(col_width)} -> {sim:.4f}")

    returned_docs = [doc for doc, sim in ranked_docs]
    lines.append("Returned docs: " + ", ".join(returned_docs))

    print("\n".join(lines))
    save_table_aligned(output_file, lines)
    print(f"\nResults appended to {output_file}")

spark.stop()