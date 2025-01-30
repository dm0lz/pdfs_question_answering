from typing import List, Optional
from openai import OpenAI
from vector_store import VectorStore


class QASystem:
    def __init__(self, embedding_model: str, llm_model: str):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        self.llm_model = llm_model
        self.vectorstore = VectorStore(embedding_model=embedding_model)

    def setup(self, urls: Optional[List[str]] = None):
        vectorstore = self.vectorstore.load_or_create_index(urls)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def get_answer(self, question: str) -> str:
        relevant_chunks = self.retriever.invoke(question)
        context = " ".join([chunk.page_content for chunk in relevant_chunks])

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the given context.",
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {question}",
                },
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or "No answer found"
