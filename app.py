import gradio as gr
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

from langchain.document_loaders import PyPDFLoader
import os

import fitz
from PIL import Image


# Global variables
COUNT, N = 0, 0
chat_history = []
chain = ''

enable_box = gr.Textbox.update(value=None, placeholder='Upload your HF key',
                               interactive=True)
disable_box = gr.Textbox.update(value="HF API key is Set", interactive=False)

# Function to set the OpenAI API key
def set_apikey(api_key):
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    return disable_box

# Function to enable the API key input box
def enable_api_box():
    return enable_box

# Function to add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

# Create Chain
# Function to process the PDF file and create a conersation chain
def process_file(file):
    # Raise an error if API key is not provided
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        raise gr.Error("Upload your HF API key")
    
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    # Split texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 0)
    texts = text_splitter.split_documents(documents)

    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"
                                        , model_kwargs = {'device': 'cpu'})

    # Create a ConversationRetrievalChain with ChatOpenAI language model
    # and PDF search retriever
    pdfsearch = Chroma.from_documents(texts, embeddings)

    # Load LLM
    repo_id = "google/flan-t5-xxl"

    llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 512, "context_length": 1000}
    )

    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                retriever=
                                                pdfsearch.as_retriever(search_kwargs={'k': 4}),
                                                return_source_documents=True)
    return chain

# Function to generate a response based on the chat history and query
def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain

    # Check if a PDF file is uploaded
    if not btn:
        raise gr.Error(message="Upload a PDF")
    
    # Initialize the conversation chain only once
    if COUNT == 0:
        chain = process_file(btn)
        COUNT+=1

    # Generate a response using the conversation chain
    result = chain({"question": query, "chat_history": chat_history}, return_only_outputs=True)

    # Update the chat history with the query and its corresponding answer
    chat_history += [(query, result['answer'])]

    # Retrieve the page number from the source document
    N = list(result['source_documents'][0])[1][1]['page']

    # Append each character of the answer to the last message in the history
    for char in result['answer']:
        history[-1][-1] +=char

        # Yield the updated history and an empty sring
        yield history, ''

# Render Image of A PDF File
# Function to render a specific page of a PDF file as an image
def render_file(file):
    global N

    # Open the PDF document using fitz
    doc = fitz.open(file.name)

    # Get the specific page to render
    page = doc[N]

    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))

    # Create an Image object from the rendered pixel data
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

    # Return the rendered image
    return image

def render_first(file):
        doc = fitz.open(file.name)
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]



# Gradio application setup
# Build Chat Interface
with gr.Blocks() as demo:
    # Create a Gradio block (low-level API to create custom webapp)

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder='Enter HF API key',
                    show_label=False,
                    interactive=True
                ).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')
            
        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select').style(height=800)

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label = False,
                placeholder="Enter text and press enter or click Submit"
            ).style(container=False)
        
        with gr.Column(scale=0.15):
            submit_btn = gr.Button("Submit")
        
        with gr.Column(scale=0.15):
            btn = gr.UploadButton("Upload a PDF", file_types=[".pdf"]).style()

    # Set up event handlers

    # Event handler for submitting the OpenAI API key
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])

    # Event handler for changing the API key
    change_api_key.click(fn=enable_api_box, outputs=[api_key])

    # Event handler for uploading a PDF
    btn.upload(fn=render_file, inputs=[btn], outputs=[show_img])

    # Event handler for submitting text and generating response by clicking Submit
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )


    # Event handler for submitting text and generating response by pressing ENTER
    txt.submit(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(share=True)