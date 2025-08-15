import base64
import json
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from openai import OpenAI
import matplotlib.pyplot as plt
import duckdb
import requests
from bs4 import BeautifulSoup
import re
import os
import tempfile

app = FastAPI()

# Configure OpenAI API client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define the data analyst agent function
def run_data_analysis(questions_text, files):
    """
    This function prompts an LLM to act as a data analyst,
    generates Python code, and executes it to get the final answer.
    """
    
    # Pre-process files for the LLM prompt
    file_info = ""
    for filename, content in files.items():
        # For CSVs, show a sample
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            file_info += f"File '{filename}' is a CSV with columns: {list(df.columns)}. Here's a sample:\n{df.head().to_string()}\n\n"
        # For images, mention their presence
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            file_info += f"File '{filename}' is an image.\n\n"
        # For other files, mention them
        else:
            file_info += f"File '{filename}' is attached.\n\n"
   # Corrected prompt snippet for `run_data_analysis` function
# ...
    # Construct the system message for the LLM
    system_prompt = f"""
    You are an expert data analyst. Your task is to answer a user's questions by writing and executing Python code.
    
    The user will provide a set of questions in a text file and may include additional data files.
    
    Follow these instructions carefully:
    1.  Your final output must be a single, complete Python script. Do not include any text before or after the script.
    2.  The script must perform all necessary data operations, including web scraping (if needed), data loading, cleaning, analysis, and visualization. Use the 'networkx' library for network analysis.
    3.  Load all provided file data directly from the in-memory content. The files dictionary `files` is already available to you. For example, to read `data.csv`, use `pd.read_csv(io.StringIO(files['data.csv'].decode('utf-8')))`
    4.  If the task requires a visualization, generate a plot, save it to an in-memory buffer, and convert it to a base64 encoded data URI. The size must be under 100,000 bytes. Do not save to a local file. Use the provided plotting code snippet for this purpose.
    5.  Print the final answer to standard output. The output must be in the exact JSON format specified in the user's questions. Do not print any other text.
    
    Example plotting code snippet for base64 encoding:
    ```python
    import io
    import base64
    import matplotlib.pyplot as plt
    import networkx as nx
    
    def get_base64_plot(fig):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
    
    # Example usage:
    # fig_network, ax_network = plt.subplots(figsize=(8, 6))
    # nx.draw(G, ax=ax_network, ...)
    # network_graph_b64 = get_base64_plot(fig_network)
    # fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    # ax_hist.bar(...)
    # degree_histogram_b64 = get_base64_plot(fig_hist)
    # 
    # final_output = {
    #    "edge_count": 7,
    #    "highest_degree_node": "Bob",
    #    ...
    #    "network_graph": network_graph_b64,
    #    "degree_histogram": degree_histogram_b64
    # }
    # print(json.dumps(final_output))
    ```
    
    Here is the user's request:
    
    ---
    
    User Questions:
    {questions_text}
    
    Attached files:
    {file_info}
    
    ---
    
    Your Python script should be a single block of code that directly produces the final JSON output.
    Begin your response with the Python code block.
    """
# ...
    # Ask the LLM to generate the script
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Or another suitable model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate the Python script to answer the user's request. The user provided the following questions: {questions_text} and files: {list(files.keys())}"}
            ]
        )
        # Extract the generated Python code from the response
        generated_code = response.choices[0].message.content.strip("```python\n").strip("\n```")
        
        # Create a temporary namespace for execution
        namespace = {'pd': pd, 'np': np, 'plt': plt, 'io': io, 'base64': base64, 'json': json, 'requests': requests, 'BeautifulSoup': BeautifulSoup, 're': re, 'duckdb': duckdb, 'files': files}
        
        # Capture standard output
        import sys
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        # Execute the generated code
        exec(generated_code, namespace)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Get the output and parse it as JSON
        output_str = redirected_output.getvalue()
        
        # In case the LLM adds text before or after the JSON, try to find and extract the JSON
        match = re.search(r'\[.*\]|\{.*\}', output_str, re.DOTALL)
        if match:
            json_output = match.group(0)
            return json.loads(json_output)
        else:
            raise ValueError("LLM output did not contain a valid JSON object or array.")

    except Exception as e:
        return {"error": str(e), "generated_code": generated_code}

@app.post("/")
async def analyze_data(
    questions: UploadFile = File(..., alias="questions.txt"),
    additional_files: list[UploadFile] = File(None)
):
    """
    Endpoint for data analysis.
    Accepts a 'questions.txt' file and optional additional files.
    """
    
    # Read the questions file
    questions_content = await questions.read()
    questions_text = questions_content.decode('utf-8')
    
    # Prepare a dictionary of file contents
    files_dict = {}
    if questions.filename:
        files_dict["questions.txt"] = questions_content
    
    if additional_files:
        for file in additional_files:
            file_content = await file.read()
            files_dict[file.filename] = file_content
            
    # Run the data analysis task
    try:
        result = run_data_analysis(questions_text, files_dict)
        return result
    except Exception as e:
        return {"error": f"An error occurred during analysis: {str(e)}"}
