# filepath: b:\agentic_ai\run_streamlit_from_notebook.py
import subprocess

if __name__ == '__main__':
    subprocess.run(["streamlit", "run", "chatbot_multiple_tools.py"])