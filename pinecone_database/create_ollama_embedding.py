from langchain_ollama import OllamaEmbeddings
from .raw_documents import documents
model="nomic-embed-text"
embeddings = OllamaEmbeddings( model=model)

def get_embedding_vector(content_str):
    vector = embeddings.embed_query(content_str)
    return vector
def length_of_embedding_vector(vector):
    print("Dimension:", len(vector))
    return len(vector)
def get_all_documents_vector(documents):
    vector_document=[]
    for document in documents:
        vector = get_embedding_vector(document["content"])
        vector_d=document
        vector_d['vector']=vector
        vector_document.append(vector_d)
    return vector_document


if __name__=='__main__':
    vector=get_embedding_vector("hi this is text")
    fixed_length_of_vector=length_of_embedding_vector(vector)
    print(fixed_length_of_vector)
    vector_document=get_all_documents_vector(documents)
    #print(vector_document)
    for v in vector_document:
        print(v['_id'],v['title'],v['content'],v['category'],v['author'],v['vector'])
        break
