import os
import warnings
from gradio_interface import GradioInterface
from qa_system import QASystem

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def cli(embedding_model, llm_model, urls):
    qa_system = QASystem(embedding_model, llm_model, urls)
    print("PDF Question Answering System Ready !")
    while True:
        question = input("\nEnter your question (type 'exit' to quit) : ").strip()
        if question.lower() == "exit":
            break
        else:
            answer = qa_system.get_answer(question)
            print(f"\nAnswer: {answer}")


def gui(embedding_model, llm_model, urls):
    gradio_interface = GradioInterface(embedding_model, llm_model, urls)
    interface = gradio_interface.create_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    embedding_model = "sentence-transformers/all-MiniLM-L12-v2"
    llm_model = "deepseek-r1-distill-qwen-7b"
    urls = [
        "http://www.marijoga.lt/Yoga_and_Kriya_Swami_Satyananda_Saraswati.pdf",
        "https://mantrayogameditation.org/wp-content/uploads/2019/12/Swami-Satyananda-Saraswati-Kundalini-Tantra.pdf",
    ]
    # gui(embedding_model, llm_model, urls)
    cli(embedding_model, llm_model, urls)
