from typing import Any
from read_configrations import get_pipeconeapi_env
pinecone_secret_key=get_pipeconeapi_env()
from pinecone import Pinecone

class PinconeOwn:
    def __init__(self):
        self.env=pinecone_secret_key
        self.pc= Pinecone(api_key=pinecone_secret_key)
    def list_index(self):
        indexes=self.pc.list_indexes()
        return indexes
    def delete_index(self,idx):
        self.pc.delete_index(idx)
    def create_index(self,name,dimension = 3,metric= 'cosine'):
         self.pc.create_index(name=name,dimension=dimension,metric=metric)
    def has_index(self,name):
        self.pc.has_index(name)
    def upsert(self, index, vectors):
        self.index = self.pc.Index(index)
        return self.index.upsert(
            vectors=vectors,
            namespace="default"
        )

    def query(self, index, vector, top_k):
        self.index = self.pc.Index(index)
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_values=True,
            namespace="default"
        )
if __name__=='__main__':
    pinecone_secret_key=get_pipeconeapi_env()
    pineobj=PinconeOwn()
    indexpc='movie-recommendations-3dimension-cosine'
#     if pineobj.has_index('movie-recommendations-3dimension-cosine'):
#         pineobj.delete_index('movie-recommendations-3dimension-cosine')
#         for p in pineobj.list_index():
#             print(p.name)
#     else:
#         pineobj.create_index(name="movie-recommendations-3dimension-cosine", dimension=3, metric="cosine")
#     for p in pineobj.list_index():
#         print(p.name)
    A=[1, 2, 3]
    B=[2, 4, 6]
    C=[1, 1, 1]
    D=[1,2,4]
    vectors = [
        {
            "id": "A",
            "values": A,
        },
        {
            "id": "B",
            "values": B,
        },
        {
            "id": "C",
            "values": C,
        },
        {
            "id": "D",
            "values": D,
        },
    ]
#     v=pineobj.upsert(index=indexpc,vectors=vectors)
    results_A=pineobj.query(index=indexpc,vector=A,top_k=3)
    results_B=pineobj.query(index=indexpc,vector=[1,1,1],top_k=3)
    print('results_A')
#     print(results_A)
    for match in results_A.matches:
        print(
            f"ID: {match.id}, "
            f"Score: {match.score:.4f}, "
            f"Vector: {match.values}"
        )
    print('results_B')
#     print(results_B)
    for match in results_B.matches:
        print(
            f"ID: {match.id}, "
            f"Score: {match.score:.4f}, "
            f"Vector: {match.values}"
        )
