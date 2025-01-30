from typing import List
from qa_system import QASystem
import gradio as gr


class GradioInterface:
    def __init__(self, embedding_model: str, llm_model: str, urls: List[str]):
        self.qa_system = QASystem(embedding_model=embedding_model, llm_model=llm_model)
        self.qa_system.setup(urls)

    def create_interface(self):
        return gr.Interface(
            fn=self.qa_system.get_answer,
            inputs=gr.Textbox(label="Enter your question"),
            outputs="text",
            title="Document Question Answering",
            description="Ask a question and get an answer based on the pdfs content.",
        )
