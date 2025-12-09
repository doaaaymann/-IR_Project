# Positional Index & TF-IDF Search System

## Overview

This project builds a **positional index** and computes **TF, IDF, TF-IDF, and normalized TF-IDF** for a collection of text documents. It also supports **Boolean and phrase queries** with cosine similarity ranking. The system is implemented using **Python** and **PySpark** for efficient handling of large datasets.

## Features

* Build positional index for words in a set of text documents.
* Compute:

  * Raw term frequency (TF)
  * Weighted TF
  * Inverse document frequency (IDF)
  * TF-IDF
  * Normalized TF-IDF
* Export tables as ASCII-formatted files:

  * `TF_table.txt`
  * `TF_weighted_table.txt`
  * `IDF_table.txt`
  * `TFIDF_table.txt`
  * `Normalized_TFIDF_table.txt`
* Query support:

  * Boolean operators: `AND`, `OR`, `AND NOT`, `OR NOT`
  * Cosine similarity ranking for results
* Error detection for invalid or incorrectly written Boolean operators.

## Requirements

* Python 3.8 or higher
* PySpark 3.x
* pandas (optional for debugging)

## Setup

1. Install PySpark if not already installed:

   ```bash
   pip install pyspark
   ```

2. Prepare your dataset:

   * Place all `.txt` files in a folder, e.g., `dataset_text_processing/`

3. Update Python environment paths in the script if needed:

   ```python
   os.environ["PYSPARK_PYTHON"] = "C:\\Users\\<username>\\anaconda3\\python.exe"
   ```

os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\Users\<username>\anaconda3\python.exe"

```

## Running the Code
1. Run the main script.
2. The script will:
- Read all text files and create a positional index.
- Compute TF, IDF, TF-IDF, and normalized TF-IDF.
- Save results as ASCII tables.
- Enter a query loop for searching documents.

### Query Examples
- Single word query:
```

python script.py
Enter your query: worser

```
- Boolean query:
```

Enter your query: mercy AND NOT worser
Enter your query: mercy OR worser

```
- Phrase query is supported by writing terms separated by spaces.

### Notes
- Boolean operators **must be uppercase** (`AND`, `OR`, `AND NOT`, `OR NOT`).
- Consecutive repetition of operators will produce an error.
- Results are appended to `query_results.txt`.

## Output Files
- `positional_index.txt`: Contains the positional index for all terms.
- `TF_table.txt`: Raw term frequencies.
- `TF_weighted_table.txt`: Weighted TF.
- `IDF_table.txt`: Inverse document frequency.
- `TFIDF_table.txt`: TF-IDF values.
- `Normalized_TFIDF_table.txt`: Normalized TF-IDF.
- `query_results.txt`: Results of queries, cosine similari
```
