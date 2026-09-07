from pinecone import Pinecone, TextQuery
from read_configrations import get_pipeconeapi_env
from .create_ollama_embedding import get_all_documents_vector,length_of_embedding_vector,get_embedding_vector
from .raw_documents import documents

documents=get_all_documents_vector(documents)
pinecone_secret_key=get_pipeconeapi_env()
pc = Pinecone(api_key=pinecone_secret_key)
namespace="articles-ollama-published"

vector=get_embedding_vector("hi this is text")
fixed_length_of_vector=length_of_embedding_vector(vector)

# pc.create_index(name="articles-ollama",dimension=fixed_length_of_vector,metric="cosine",)
idx = pc.Index("articles-ollama")

print("\n========== UPSERT ==========")
response = idx.documents.upsert(namespace=namespace, documents=documents,)
print("Upsert response:")
print(response)
print("Documents accepted:", response.upserted_count)


print("\n========== SEARCH ==========")
response = idx.documents.search(
    namespace=namespace,
    top_k=3,
    score_by=[
        TextQuery(
            query="vector database RAG",
            fields=["content"],
        )
    ],
    include_fields=[
        "title",
        "content",
        "category",
        "author",
    ],
)


print("\nSearch results:")
for doc in response.matches:

    print("--------------------------------")
    print("ID       :", doc.id)
    print("Score    :", doc.score)
    print("Title    :", doc.title)
    print("Category :", doc.category)
    print("Author   :", doc.author)
    print("Content  :", doc.content)


# ============================================================
# 5. FETCH DOCUMENTS BY ID
# ============================================================

print("\n========== FETCH ==========")

response = idx.documents.fetch(
    namespace=namespace,
    ids=[
        "article-101",
        "article-102",
    ],
)

print("Fetched documents:")

for doc_id, doc in response.documents.items():

    print("--------------------------------")
    print("ID      :", doc_id)
    print("Title   :", doc.title)
    print("Content :", doc.content)


# ============================================================
# 6. UPDATE A DOCUMENT
# ============================================================

print("\n========== UPDATE ==========")

response = idx.documents.update(
    namespace=namespace,
    documents=[
        {
            "_id": "article-101",
            # Change this field
            "title": "Advanced Introduction to Vector Databases",
            # Add a new field
            "status": "published",
        }
    ],
)

print("Update response:")
print(response)

# ============================================================
# 7. FETCH THE UPDATED DOCUMENT
# ============================================================

print("\n========== FETCH UPDATED DOCUMENT ==========")
response = idx.documents.fetch(
    namespace=namespace,
    ids=["article-101"],
)

for doc_id, doc in response.documents.items():

    print("--------------------------------")
    print("ID     :", doc_id)
    print("Title  :", doc.title)
    print("Status :", doc.status)


# ============================================================
# 8. LIST DOCUMENTS
# ============================================================

print("\n========== LIST DOCUMENTS ==========")

for doc in idx.documents.list(
    namespace=namespace
):

    print("Document ID:", doc.id)


# ============================================================
# 9. SEARCH WITH FILTER
# ============================================================

print("\n========== SEARCH WITH FILTER ==========")

response = idx.documents.search(
    namespace=namespace,

    top_k=5,

    score_by=[
        TextQuery(
            query="AI",
            fields=["content"],
        )
    ],

    include_fields=[
        "title",
        "content",
        "category",
    ],

    # Only search documents where category == AI
    filter={
        "category": {
            "$eq": "AI"
        }
    },
)


for doc in response.matches:

    print("--------------------------------")
    print("ID       :", doc.id)
    print("Score    :", doc.score)
    print("Title    :", doc.title)
    print("Category :", doc.category)


# ============================================================
# 10. DELETE ONE DOCUMENT
# ============================================================

print("\n========== DELETE ==========")

response = idx.documents.delete(
    namespace=namespace,
    ids=["article-103"],
)

print("Delete response:")
print(response)


# ============================================================
# 11. VERIFY DELETE
# ============================================================

print("\n========== VERIFY DELETE ==========")

response = idx.documents.fetch(
    namespace=namespace,
    ids=["article-103"],
)

print("Remaining documents:")

for doc_id, doc in response.documents.items():

    print(doc_id)


print("\n========== DONE ==========")
