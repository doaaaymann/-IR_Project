import os

os.environ["PYSPARK_PYTHON"] = "C:\\Users\\doaaa\\anaconda3\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\doaaa\\anaconda3\\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, input_file_name
import re
import shutil
import math
from functools import reduce

OUTPUT_DIR = "outputs"
RESULTS_DIR = "results"

def ensure_dirs_exist():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

ensure_dirs_exist()
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

rdd = df.rdd.flatMap(lambda row: [(row["filename"], idx, word)
                                  for idx, word in enumerate(re.findall(r"\w+", row["value"].lower()))])

positional_index = (
    rdd.map(lambda x: (x[2], (x[0], x[1])))
    .groupByKey()
    .mapValues(list)
)


def format_output_pretty(term, postings):
    doc_positions = {}
    for doc, pos in postings:
        doc_positions.setdefault(doc, []).append(pos + 1)

    sorted_docs = sorted(doc_positions.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]))

    doc_list_parts = [f"{doc}: {len(doc_positions[doc])}" for doc in sorted_docs]
    doc_list_string = "; ".join(doc_list_parts)

    result = f"< {term.ljust(10)}{doc_list_string} >"
    return result


formatted_output_pretty = (
    positional_index
    .sortBy(lambda x: x[0])
    .map(lambda x: format_output_pretty(x[0], x[1]))
)

positional_index_path = os.path.join(OUTPUT_DIR, "positional_index.txt")
shutil.rmtree(positional_index_path, ignore_errors=True)

lines = formatted_output_pretty.collect()
with open(positional_index_path, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")
        print(line)

input_file = positional_index_path

term_docs = {}
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"<\s*(\w+)\s*(.*?)\s*>", line)
        if match:
            term = match.group(1)
            doc_list_str = match.group(2)
            term_docs[term] = {}

            parts = re.findall(r'(\d+\.txt):\s*(\d+)', doc_list_str)
            for doc, count in parts:
                term_docs[term][doc] = [0] * int(count)

docs = sorted({doc for postings in term_docs.values() for doc in postings.keys()},
              key=lambda x: int(re.findall(r'\d+', x)[0]))

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
for term in sorted(tf_matrix.keys()):
    counts = tf_matrix[term]
    df = sum(1 for c in counts if c > 0)
    idf_value = math.log10(N / df) if df > 0 else 0
    idf[term] = idf_value
    idf_table.append([term, df, idf_value])

tfidf_matrix = {
    term: [tf * idf[term] for tf in counts]
    for term, counts in tf_matrix.items()
}


def save_ascii_table(filename, header, rows):
    filepath = os.path.join(OUTPUT_DIR, filename)
    col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([header] + rows))]
    with open(filepath, "w", encoding="utf-8") as f:
        sep = "+" + "+".join("-" * w for w in col_widths) + "+"
        f.write(sep + "\n")
        f.write("|" + "|".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header))) + "|\n")
        f.write(sep + "\n")
        for row in rows:
            f.write("|" + "|".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + "|\n")
        f.write(sep + "\n")


tf_rows_raw = [[term] + [tf_raw_matrix[term][i] for i in range(len(docs))] for term in sorted(tf_raw_matrix.keys())]
save_ascii_table("TF_table.txt", ["Term"] + docs, tf_rows_raw)

tf_rows_weighted = [[term] + [tf_matrix[term][i] for i in range(len(docs))] for term in sorted(tf_matrix.keys())]
save_ascii_table("TF_weighted_table.txt", ["Term"] + docs, tf_rows_weighted)

idf_rows = [[term, df, idf[term]] for term, df, idf_value in idf_table]
save_ascii_table("IDF_table.txt", ["Term", "DF", "IDF"], idf_rows)

tfidf_rows = [[term] + [round(v, 5) for v in values] for term, values in tfidf_matrix.items()]
save_ascii_table("TFIDF_table.txt", ["Term"] + docs, tfidf_rows)

doc_lengths = {}
for i, doc in enumerate(docs):
    length = math.sqrt(sum((tfidf_matrix[term][i]) ** 2 for term in tfidf_matrix))
    doc_lengths[doc] = length

tfidf_table_path = os.path.join(OUTPUT_DIR, "TFIDF_table.txt")
with open(tfidf_table_path, "a", encoding="utf-8") as f:
    f.write("\n")
    for doc, length in doc_lengths.items():
        f.write(f"{doc} length {length}\n")

normalized_tfidf_matrix = {}
for term in tfidf_matrix:
    normalized_tfidf_matrix[term] = []
    for i, doc in enumerate(docs):
        length = doc_lengths[doc]
        normalized_value = tfidf_matrix[term][i] / length if length > 0 else 0
        normalized_tfidf_matrix[term].append(round(normalized_value, 5))

normalized_rows = [[term] + values for term, values in normalized_tfidf_matrix.items()]
save_ascii_table("Normalized_TFIDF_table.txt", ["Term"] + docs, normalized_rows)

output_file = os.path.join(RESULTS_DIR, "query_results.txt")
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
        break

    error_msg = check_invalid_boolean(query)
    if error_msg:
        print(error_msg)
        continue

    vocabulary = sorted(tf_matrix.keys())
    query_terms_list = re.findall(r"\w+", query.lower())
    query_terms = [t for t in query_terms_list if t.upper() not in BOOLEAN_OPS]

    query_tf = compute_query_tf(query_terms_list, vocabulary)

    query_tfidf = [query_tf[i] * idf[vocabulary[i]] for i in range(len(vocabulary))]

    query_length = math.sqrt(sum(val ** 2 for val in query_tfidf))
    normalized_query_tfidf = [round(val / query_length, 5) if query_length > 0 else 0 for val in query_tfidf]

    matched_docs = sorted(evaluate_query(term_docs, docs, query))
    if not matched_docs:
        print("No matching documents.")
        continue

    product_matrix = {}
    sum_per_doc = {}
    similarities = {}
    for doc in matched_docs:
        doc_index = docs.index(doc)
        doc_vec = [tf_matrix[vocabulary[i]][doc_index] * idf[vocabulary[i]] for i in range(len(vocabulary))]
        doc_vec_norm = [val / doc_lengths[doc] if doc_lengths[doc] > 0 else 0 for val in doc_vec]

        product_values = [normalized_query_tfidf[i] * doc_vec_norm[i] for i in range(len(vocabulary))]
        product_matrix[doc] = product_values
        sum_per_doc[doc] = sum(product_values)
        similarities[doc] = sum(product_values)

    ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    col_width = 15
    lines = []

    lines.append("=" * 5 + " QUERY TABLE " + "=" * 5)

    query_header = ["term", "tf-raw", "wtf(1+log tf)", "idf", "tf*idf"]
    lines.append("".join(h.ljust(col_width) for h in query_header))
    lines.append("-" * (col_width * len(query_header)))

    unique_query_terms = sorted(list(set(query_terms)))

    for term in unique_query_terms:
        if term in vocabulary:
            idx = vocabulary.index(term)
            tf_weighted_val = 1 + math.log10(query_tf[idx]) if query_tf[idx] > 0 else 0

            row = [
                term.ljust(col_width),
                str(query_terms_list.count(term)).ljust(col_width),
                f"{tf_weighted_val:.4f}".ljust(col_width),
                f"{idf[term]:.4f}".ljust(col_width),
                f"{query_tfidf[idx]:.4f}".ljust(col_width)
            ]
            lines.append("".join(row))

    lines.append("\nnormalized vector:")

    for term in unique_query_terms:
        if term in vocabulary:
            idx = vocabulary.index(term)
            lines.append(f"{term.ljust(col_width)}: {normalized_query_tfidf[idx]:.4f}")

    lines.append(f"\nquery length = {query_length:.6f}")

    lines.append("\n" + "=" * 5 + " PRODUCT TABLE " + "=" * 5)

    product_header = ["term"] + matched_docs
    lines.append("".join(h.ljust(col_width) for h in product_header))
    lines.append("-" * (col_width * (len(product_header))))

    for term in unique_query_terms:
        if term in vocabulary:
            idx = vocabulary.index(term)
            row = [term.ljust(col_width)]
            for doc in matched_docs:
                row.append(f"{product_matrix[doc][idx]:.4f}".ljust(col_width))
            lines.append("".join(row))

    lines.append("-" * (col_width * (len(product_header))))
    sum_row = ["SUM".ljust(col_width)]
    for doc in matched_docs:
        sum_row.append(f"{sum_per_doc[doc]:.4f}".ljust(col_width))
    lines.append("".join(sum_row))

    lines.append("\n" + "=" * 5 + " COSINE SIMILARITY " + "=" * 5)
    for doc, sim in ranked_docs:
        lines.append(f"similarity(q , {doc}) = {sim:.4f}")

    returned_docs_str = ", ".join([doc for doc, sim in ranked_docs])
    lines.append(f"\nReturned docs: {returned_docs_str}")
    lines.append("=" * (col_width * 5))

    print("\n".join(lines))
    save_table_aligned(output_file, lines)

spark.stop()