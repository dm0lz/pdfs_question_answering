import os
import warnings
from gradio_interface import GradioInterface
from qa_system import QASystem

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def cli(urls):
    qa_system = QASystem()
    qa_system.setup(urls)
    answer = qa_system.get_answer("What is kriya yoga?")
    print(f"Answer: {answer}")


def gui(urls):
    gradio_interface = GradioInterface(urls)
    interface = gradio_interface.create_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    urls = [
        "http://www.marijoga.lt/Yoga_and_Kriya_Swami_Satyananda_Saraswati.pdf",
        "https://mantrayogameditation.org/wp-content/uploads/2019/12/Swami-Satyananda-Saraswati-Kundalini-Tantra.pdf",
    ]
    # cli(urls)
    gui(urls)
