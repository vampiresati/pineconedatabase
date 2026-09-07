from pinecone import Pinecone, TextQuery, DenseVectorQuery

from read_configrations import get_pipeconeapi_env

from .create_ollama_embedding import (
    get_all_documents_vector,
    length_of_embedding_vector,
    get_embedding_vector,
)

from .raw_documents import documents


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = "articles-ollama"
NAMESPACE = "articles-ollama-published"


# ============================================================
# PINECONE CONNECTION
# ============================================================

def get_pinecone_client():
    pinecone_secret_key = get_pipeconeapi_env()

    return Pinecone(
        api_key=pinecone_secret_key
    )


# ============================================================
# CREATE DOCUMENT SCHEMA INDEX
# ============================================================

def create_document_index(pc):
    """
    Create a Pinecone Document Schema index.

    embedding:
        768-dimensional Ollama embedding

    content:
        Full-text searchable field

    title:
        Full-text searchable field
    """

    if pc.indexes.exists(INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists.")
        return

    pc.indexes.create(
        name=INDEX_NAME,

        schema={
            "fields": {

                "embedding": {
                    "type": "dense_vector",
                    "dimension": 768,
                    "metric": "cosine",
                },

                "content": {
                    "type": "string",
                    "full_text_search": {},
                },

                "title": {
                    "type": "string",
                    "full_text_search": {},
                },
            }
        },

        deployment={
            "deployment_type": "managed",
            "cloud": "aws",
            "region": "us-east-1",
        },
    )

    print(f"Index '{INDEX_NAME}' created.")


# ============================================================
# GET INDEX
# ============================================================

def get_index(pc):
    return pc.index(INDEX_NAME)


# ============================================================
# PREPARE DOCUMENTS
# ============================================================

def prepare_documents():
    """
    Generate Ollama embeddings and attach them
    to the Pinecone documents.
    """

    vector_documents = get_all_documents_vector(documents)

    return vector_documents


# ============================================================
# UPSERT DOCUMENTS
# ============================================================

def upsert_documents(index, documents):
    print("\n========== UPSERT ==========")

    response = index.documents.upsert(
        namespace=NAMESPACE,
        documents=documents,
    )

    print("Upsert response:")
    print(response)

    print(
        "Documents accepted:",
        response.upserted_count
    )

    return response


# ============================================================
# TEXT SEARCH
# ============================================================

def search_documents(index, query, top_k=3):
    print("\n========== TEXT SEARCH ==========")

    response = index.documents.search(
        namespace=NAMESPACE,

        top_k=top_k,

        score_by=[
            TextQuery(
                query=query,
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

    return response


# ============================================================
# DENSE VECTOR SEARCH
# ============================================================

def search_documents_by_embedding(index, query_text, top_k=3):
    print("\n========== VECTOR SEARCH ==========")

    query_vector = get_embedding_vector(
        query_text
    )

    response = index.documents.search(
        namespace=NAMESPACE,

        top_k=top_k,

        score_by=[
            DenseVectorQuery(
                field="embedding",
                values=query_vector,
            )
        ],

        include_fields=[
            "title",
            "content",
            "category",
            "author",
        ],
    )

    print("\nVector search results:")

    for doc in response.matches:

        print("--------------------------------")

        print("ID       :", doc.id)
        print("Score    :", doc.score)
        print("Title    :", doc.title)
        print("Category :", doc.category)
        print("Author   :", doc.author)

    return response


# ============================================================
# FETCH DOCUMENTS
# ============================================================

def fetch_documents(index, ids):
    print("\n========== FETCH ==========")

    response = index.documents.fetch(
        namespace=NAMESPACE,
        ids=ids,
    )

    print("Fetched documents:")

    for doc_id, doc in response.documents.items():

        print("--------------------------------")

        print("ID      :", doc_id)
        print("Title   :", doc.title)
        print("Content :", doc.content)

    return response


# ============================================================
# UPDATE DOCUMENT
# ============================================================

def update_document(index, document_id):
    print("\n========== UPDATE ==========")

    response = index.documents.update(
        namespace=NAMESPACE,

        documents=[
            {
                "_id": document_id,

                "title": (
                    "Advanced Introduction "
                    "to Vector Databases"
                ),

                "status": "published",
            }
        ],
    )

    print("Update response:")
    print(response)

    return response


# ============================================================
# FETCH UPDATED DOCUMENT
# ============================================================

def fetch_updated_document(index, document_id):
    print("\n========== FETCH UPDATED DOCUMENT ==========")

    response = index.documents.fetch(
        namespace=NAMESPACE,

        ids=[document_id],

        include_fields=[
            "title",
            "status",
        ],
    )

    for doc_id, doc in response.documents.items():

        print("--------------------------------")

        print("ID     :", doc_id)
        print("Title  :", doc.title)

        # status may not exist if update has not
        # become visible yet.
        if hasattr(doc, "status"):
            print("Status :", doc.status)
        else:
            print("Status : not visible yet")

    return response


# ============================================================
# LIST DOCUMENTS
# ============================================================

def list_documents(index):
    print("\n========== LIST DOCUMENTS ==========")

    for doc in index.documents.list(
        namespace=NAMESPACE
    ):

        print(
            "Document ID:",
            doc.id
        )


# ============================================================
# SEARCH WITH FILTER
# ============================================================

def search_documents_with_filter(
    index,
    query,
    category,
    top_k=5,
):
    print("\n========== SEARCH WITH FILTER ==========")

    response = index.documents.search(
        namespace=NAMESPACE,

        top_k=top_k,

        score_by=[
            TextQuery(
                query=query,
                fields=["content"],
            )
        ],

        include_fields=[
            "title",
            "content",
            "category",
        ],

        filter={
            "category": {
                "$eq": category
            }
        },
    )

    for doc in response.matches:

        print("--------------------------------")

        print("ID       :", doc.id)
        print("Score    :", doc.score)
        print("Title    :", doc.title)
        print("Category :", doc.category)

    return response


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(index, document_id):
    print("\n========== DELETE ==========")

    response = index.documents.delete(
        namespace=NAMESPACE,

        ids=[
            document_id
        ],
    )

    print("Delete response:")
    print(response)

    return response


# ============================================================
# VERIFY DELETE
# ============================================================

def verify_delete(index, document_id):
    print("\n========== VERIFY DELETE ==========")

    response = index.documents.fetch(
        namespace=NAMESPACE,

        ids=[
            document_id
        ],
    )

    if document_id in response.documents:
        print(
            "Document still exists:",
            document_id
        )
    else:
        print(
            "Document deleted:",
            document_id
        )

    return response


# ============================================================
# MAIN
# ============================================================

def main():

    # ------------------------------------------
    # 1. Ollama embedding test
    # ------------------------------------------

    vector = get_embedding_vector(
        "hi this is text"
    )

    fixed_length_of_vector = (
        length_of_embedding_vector(vector)
    )

    print(
        "Embedding dimension:",
        fixed_length_of_vector
    )

    # ------------------------------------------
    # 2. Pinecone
    # ------------------------------------------

    pc = get_pinecone_client()

    # ------------------------------------------
    # 3. Create index
    # ------------------------------------------

    create_document_index(pc)

    # ------------------------------------------
    # 4. Get index
    # ------------------------------------------

    index = get_index(pc)

    # ------------------------------------------
    # 5. Generate Ollama embeddings
    # ------------------------------------------

    vector_documents = prepare_documents()

    # ------------------------------------------
    # 6. UPSERT
    # ------------------------------------------

    upsert_documents(
        index,
        vector_documents
    )

    # ------------------------------------------
    # 7. TEXT SEARCH
    # ------------------------------------------

    search_documents(
        index,
        "vector database RAG"
    )

    # ------------------------------------------
    # 8. VECTOR SEARCH
    # ------------------------------------------

    search_documents_by_embedding(
        index,
        "What is a vector database?"
    )

    # ------------------------------------------
    # 9. FETCH
    # ------------------------------------------

    fetch_documents(
        index,
        [
            "article-101",
            "article-102",
        ]
    )

    # ------------------------------------------
    # 10. UPDATE
    # ------------------------------------------

    update_document(
        index,
        "article-101"
    )

    # ------------------------------------------
    # 11. FETCH UPDATED
    # ------------------------------------------

    fetch_updated_document(
        index,
        "article-101"
    )

    # ------------------------------------------
    # 12. LIST
    # ------------------------------------------

    list_documents(index)

    # ------------------------------------------
    # 13. FILTER SEARCH
    # ------------------------------------------

    search_documents_with_filter(
        index,
        "AI",
        "AI"
    )

    # ------------------------------------------
    # 14. DELETE
    # ------------------------------------------

    delete_document(
        index,
        "article-103"
    )

    # ------------------------------------------
    # 15. VERIFY DELETE
    # ------------------------------------------

    verify_delete(
        index,
        "article-103"
    )

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
