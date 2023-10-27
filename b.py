import gradio as gr
def update(name):
    return f"Welcome to Gradio, {name}!"

with gr.Blocks() as demo:
    # Create a Gradio block (low-level API to create custom webapp)

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder='Enter OpenAI API key',
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
                placeholder="Enter text and press enter"
            ).style(container=False)
        
        with gr.Column(scale=0.15):
            submit_btn = gr.Button("Submit")
        
        with gr.Column(scale=0.15):
            btn = gr.UploadButton("Upload a PDF", file_types=[".pdf"]).style()

demo.launch()