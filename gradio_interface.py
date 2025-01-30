from typing import List
from qa_system import QASystem
import gradio as gr


class GradioInterface:
    def __init__(self, urls: List[str]):
        self.qa_system = QASystem()
        self.qa_system.setup(urls)

    def create_interface(self):
        return gr.Interface(
            fn=self.qa_system.get_answer,
            inputs=gr.Textbox(label="Enter your question"),
            outputs="text",
            title="Document Question Answering",
            description="Ask a question about yoga and get an answer based on the loaded texts.",
        )
