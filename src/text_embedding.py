#!/usr/bin/env python
# coding: utf-8

# ## Text Embedding

# ### This notebook demonstrates how to generate text embeddings using transformer-based models (all-MiniLM-L6-v2) Natural Language Processing (NLP) applications. 

# In[ ]:


import os
import time
import math
import numpy as np
import pandas as pd
import pyarrow
import torch
import gc

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


# In[ ]:





# In[2]:


print(torch.__version__)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("Embedding dim:", m.get_sentence_embedding_dimension())


# In[ ]:





# ### Data loading

# In[4]:


# Load data
main_df = pd.read_csv('E://CV//Internship//Coding_Challenge_Omkar_Pawar//data//cleaned_sentiment_dataset.csv')


# In[5]:


df = main_df.copy()
df


# ### Generate Embeddings using all-MiniLM-L6-v2

# In[13]:


# -----------------------------
# Config
# -----------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = "embeddings_out"
MAX_WORDS = 50
BATCH = 128
DTYPE = np.float16

print("Starting embedding...\n")
start_time = time.time()

# -----------------------------
# prep text
# -----------------------------
texts = df["text"].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
texts = texts.apply(lambda x: " ".join(x.split()[:MAX_WORDS]) if x else x)

# -----------------------------
# Model
# -----------------------------
model = SentenceTransformer(MODEL_NAME, device="cpu")
dim = model.get_sentence_embedding_dimension()

# -----------------------------
# Safe memmap creation (Windows)
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
path_mmap = os.path.abspath(os.path.join(OUT_DIR, "miniLM_L6v2_embeddings_float16.mmap"))

dtype = DTYPE
n_rows, n_cols = len(texts), dim
itemsize = np.dtype(dtype).itemsize
total_bytes = n_rows * n_cols * itemsize

# If old file exists, remove or rename
if os.path.exists(path_mmap):
    try:
        os.remove(path_mmap)
    except PermissionError:
        path_mmap = os.path.abspath(os.path.join(OUT_DIR, "miniLM_L6v2_embeddings_float16_new.mmap"))

# Pre-size then map r+
with open(path_mmap, "wb") as f:
    if total_bytes > 0:
        f.seek(total_bytes - 1)
        f.write(b"\x00")

emb = np.memmap(path_mmap, dtype=dtype, mode="r+", shape=(n_rows, n_cols))
print("memmap created:", path_mmap, "size (MB):", round(total_bytes / (1024**2), 2))

# -----------------------------
# Embedding loop
# -----------------------------
try:
    for s in tqdm(range(0, n_rows, BATCH), desc="Embedding"):
        batch = texts.iloc[s:s+BATCH].tolist()
        vecs = model.encode(
            batch,
            batch_size=BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        emb[s:s+len(vecs)] = vecs.astype(dtype)
finally:
    emb.flush()

print("✅ Done! Saved embeddings to:", path_mmap)

# -----------------------------
# Timing summary
# -----------------------------
end_time = time.time()
total_time = end_time - start_time
per_text = total_time / max(1, n_rows)

print(f"Total time: {total_time/60:.2f} minutes for {n_rows:,} texts")
print(f"Average speed: {per_text*1000:.2f} ms per text ({1/per_text:.2f} texts/sec)\n")


# In[ ]:





# In[ ]:




