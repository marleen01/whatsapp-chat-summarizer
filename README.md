# WhatsApp Chat Summarizer with Local LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This Python script parses WhatsApp chat exports (`.txt`), lets you pick a date range, and uses a local LLM (e.g., via LM Studio) to generate daily conversation summaries. It handles long chats by chunking them for the LLM.

## Key Features

*   Parses WhatsApp `.txt` exports (including multi-line messages).
*   Parses sender names.
*   User-defined date range for summarization.
*   Interacts with local LLMs via an OpenAI-compatible API (e.g., LM Studio).
*   Chunks long daily chats to fit LLM context windows.
*   Outputs daily summaries to a text file, attributing points to configured senders.

## Tech Stack

*   Python 3.x
*   Pandas
*   Requests
*   LM Studio (or similar local LLM server)

## Quick Start

1.  **Clone:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/whatsapp-chat-summarizer.git
    cd whatsapp-chat-summarizer
    ```
    *(Replace `YOUR_USERNAME`)*

2.  **Setup Environment:**
    ```bash
    # Create & activate a virtual environment (e.g., python -m venv venv; source venv/bin/activate)
    pip install -r requirements.txt
    ```

3.  **Configure `main.py`:**
    Open `main.py` and **edit these critical lines** at the top:
    ```python
    CHAT_FILE_PATH = r"YOUR_WHATSAPP_CHAT_EXPORT.txt" # Path to your chat.txt
    LM_STUDIO_MODEL_ID = "your-actual-model-id-from-lm-studio" # Model ID from LM Studio
    ```
    *   **`CHAT_FILE_PATH`**: Path to your exported WhatsApp `.txt` file (Export chat "Without Media").
    *   **`LM_STUDIO_MODEL_ID`**: The exact model ID running on your LM Studio server.

4.  **Run Local LLM Server:**
    Start your LM Studio (or similar) server with a suitable model loaded. Ensure it's accessible (default: `http://127.0.0.1:8000/v1`).

5.  **Run the Script:**
    ```bash
    python main.py
    ```
    Follow prompts for the date range. Summaries will be printed and saved to `whatsapp_chat_summaries_custom_range.txt`.

## How it Works

The script loads the chat, groups messages by day. For each day in the selected range:
1.  If short, the day's chat is sent directly to the LLM for summarization.
2.  If long, the chat is split into overlapping chunks. Each chunk is summarized.
3.  These chunk summaries are then combined and sent to the LLM for a final daily summary.

---
Author: Emre Yilmaz / github.com/marleen01