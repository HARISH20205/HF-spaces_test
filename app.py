import gradio as gr
from main import main as process_files_and_query
import os
import shutil

os.makedirs('/src', exist_ok=True)

def process_and_query(files, query):
    try:
        file_paths = []
        for file in files:
            filename = os.path.join('/src', os.path.basename(file.name))
            counter = 1
            while os.path.exists(filename):
                name, ext = os.path.splitext(os.path.basename(file.name))
                filename = os.path.join('/src', f"{name}_{counter}{ext}")
                counter += 1
            
            shutil.copy(file.name, filename)
            file_paths.append(filename)
        
        # Process the files and query
        result = process_files_and_query(file_paths, query)
        
        # Ensure the result is a string for display
        return str(result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

iface = gr.Interface(
    fn=process_and_query,
    inputs=[
        gr.File(file_count="multiple", label="Upload files (PDF, audio, image)"),
        gr.Textbox(label="Enter your query")
    ],
    outputs=gr.Textbox(label="Result"),
    title="File Processing and Query System",
    description="Upload files (PDF, audio, image) and enter a query to get relevant information."
)

iface.launch()