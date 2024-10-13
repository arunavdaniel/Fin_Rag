import gradio as gr
import main


index = None
text_chunks = None


def app_interface_load(ticker, email):
    global index, text_chunks
    cik = main.company_search(ticker, email)
    accession, document_id = main.reports_find_docs_recent_10k(cik, email)
    text = main.get_reports(cik, accession, document_id, email)
    text_chunks = main.chunking_normal(text, 500)
    embeddings = main.embeddings_gen(text_chunks)
    index = main.creation_and_adding(embeddings)

    return "Data Loaded "


def app_interface_chat(query):
    if index is None or text_chunks is None:
        return "Please load company data first!"

    query_result = main.quarying(query, index, text_chunks)
    response = main.google_gen_ai(query, query_result)

    return response







with gr.Blocks() as app:
    gr.Markdown("# SEC Report Querying App")

    with gr.Tab("Load Company Data"):
        ticker = gr.Textbox(label="Ticker Symbol")
        email = gr.Textbox(label="Your Email")
        load_button = gr.Button("Load Data")
        load_output = gr.Textbox(label="Load Status", interactive=False)

        load_button.click(fn=app_interface_load, inputs=[ticker, email], outputs=[load_output])

    with gr.Tab("Query Data"):
        query = gr.Textbox(label="Enter your query")
        submit_button = gr.Button("Submit Query")
        chat_output = gr.Markdown(label="Response")


        submit_button.click(fn=app_interface_chat, inputs=[query], outputs=chat_output)


app.launch()
