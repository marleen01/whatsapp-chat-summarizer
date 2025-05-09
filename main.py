# Copyright (c) 2025 Emre Yilmaz
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

import re
import pandas as pd
from datetime import datetime, date as DDate, timedelta
import requests
from collections import Counter # Import Counter

# --- CONFIGURATION---
LINE_PATTERN = r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*)$"
CHAT_FILE_PATH = r"example_chat.txt" # <<< Default, change this
BASE_URL = "http://127.0.0.1:8000/v1"
HEADERS = {"Content-Type": "application/json"}
LM_STUDIO_MODEL_ID = "gemma-3-12b-it-qat" # !!! CHANGE THIS to your actual model ID !!!
SUMMARY_OUTPUT_FILENAME_CUSTOM = "whatsapp_chat_summaries_custom_range.txt"
CHUNK_TARGET_CHAR_LENGTH = 10000
CHUNK_OVERLAP_CHAR_LENGTH = 500
MAX_TOKENS_PER_CHUNK_SUMMARY = 250
MAX_TOKENS_FINAL_SUMMARY = 400
# --- END CONFIGURATION ---


def get_date_from_user(prompt_message: str) -> DDate | None:
    """Helper function to get a valid date from user input."""
    while True:
        date_str = input(prompt_message + " (YYYY-MM-DD): ").strip()
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

def get_primary_senders(df: pd.DataFrame, num_senders: int = 2) -> list[str]:
    """
    Identifies the most frequent senders from the DataFrame.
    Returns a list of sender names, or defaults if insufficient data.
    """
    if df.empty or 'sender' not in df.columns:
        print("Warning: Cannot determine primary senders from empty or invalid DataFrame.")
        return [f"Sender{i+1}" for i in range(num_senders)] # Default names

    sender_counts = Counter(df['sender'])
    most_common_senders = sender_counts.most_common(num_senders)

    if not most_common_senders:
        print("Warning: No senders found in the chat data.")
        return [f"Sender{i+1}" for i in range(num_senders)]

    primary_senders = [sender[0] for sender in most_common_senders]

    # Ensure we always return the requested number of senders, padding if necessary
    while len(primary_senders) < num_senders:
        primary_senders.append(f"OtherSender{len(primary_senders) + 1 - len(most_common_senders)}")
        print(f"Warning: Fewer than {num_senders} distinct senders found. Using default names for missing ones.")


    return primary_senders


def main():
    """
    Main function to load chat, get user input for a custom date range,
    group messages, and summarize.
    """
    print(f"Loading WhatsApp chat data from: {CHAT_FILE_PATH}")
    print("If this is not your chat file, please edit CHAT_FILE_PATH in the script.")
    df = load_chat_to_df(CHAT_FILE_PATH)

    if df.empty:
        print(f"No messages were parsed from {CHAT_FILE_PATH}. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(df)} messages.")

    # --- AUTO-DETECT PRIMARY SENDERS ---
    primary_senders = get_primary_senders(df, num_senders=2)
    if len(primary_senders) < 2: # Should be handled by get_primary_senders padding
        print("Error: Could not reliably determine two primary senders. Exiting.")
        # Or provide a fallback to manual input if desired
        # primary_senders = ["DefaultSender1", "DefaultSender2"] # Fallback
        return # Or handle differently

    sender1, sender2 = primary_senders[0], primary_senders[1]
    print(f"Auto-detected primary senders: {sender1} and {sender2}")
    print("If these are incorrect, the chat data might be unusual or have many participants.")


    print("\nGrouping messages by day...")
    daily_chunks = group_by_day(df.copy())

    if not daily_chunks:
        print("No daily message chunks found. Exiting.")
        return

    all_available_dates = sorted(list(daily_chunks.keys()))

    if not all_available_dates:
        print("No dates with messages found after grouping. Exiting.")
        return

    print(f"\nData available from {all_available_dates[0].strftime('%Y-%m-%d')} to {all_available_dates[-1].strftime('%Y-%m-%d')}.")

    start_date = None
    end_date = None

    while True:
        print("\nEnter the date range you want to summarize.")
        start_date_input = get_date_from_user("Start date")
        if start_date_input is None:
            print("Start date not provided. Exiting.")
            return

        end_date_input = get_date_from_user("End date (or leave empty to use start date as a single day)")
        if end_date_input is None:
            end_date_input = start_date_input
            print(f"End date not provided, will summarize for single day: {start_date_input.strftime('%Y-%m-%d')}")

        if start_date_input > end_date_input:
            print("Error: Start date cannot be after end date. Please try again.")
            continue
        else:
            start_date = start_date_input
            end_date = end_date_input
            break

    print(f"\nSelected range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    dates_to_process = [
        d for d in all_available_dates if start_date <= d <= end_date
    ]

    if not dates_to_process:
        print(f"No messages found within the specified date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
        return

    print(f"\nFound {len(dates_to_process)} day(s) within the selected range to summarize...")

    all_summaries = {}
    for date_obj in dates_to_process:
        day_df = daily_chunks[date_obj]
        if day_df.empty:
            print(f"Skipping {date_obj.strftime('%Y-%m-%d')}: No parsable messages for this day in the chunk.")
            all_summaries[date_obj] = "No parsable messages for this day."
            continue

        text_for_summary_lines = []
        for _, row in day_df.iterrows():
            text_for_summary_lines.append(f"{row['sender']}: {row['message']}")
        text_for_summary = "\n".join(text_for_summary_lines)

        print(f"\n--- Summarizing chat for {date_obj.strftime('%Y-%m-%d')} ({len(day_df)} messages) ---")
        # Pass the detected senders to the summarization function
        summary = summarize_day_with_llm(date_obj, text_for_summary, LM_STUDIO_MODEL_ID, sender1, sender2)
        all_summaries[date_obj] = summary
        print(f"Summary for {date_obj.strftime('%Y-%m-%d')}:\n{summary}")

    print("\n\n--- All Generated Summaries for Custom Range (Chronological) ---")
    sorted_summary_items = sorted(all_summaries.items(), key=lambda item: item[0])

    for date_obj, summary_text in sorted_summary_items:
        print(f"\nDate: {date_obj.strftime('%Y-%m-%d')}")
        print(summary_text)

    if all_summaries:
        try:
            with open(SUMMARY_OUTPUT_FILENAME_CUSTOM, "w", encoding="utf-8") as f_out:
                f_out.write(f"WhatsApp Chat Summaries for Custom Range\n")
                f_out.write(f"Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
                f_out.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_out.write(f"Summarized {len(sorted_summary_items)} day(s).\n")
                f_out.write(f"Primary Senders Considered: {sender1}, {sender2}\n") # Add this info
                f_out.write("=" * 30 + "\n\n")

                for date_obj, summary_text in sorted_summary_items:
                    f_out.write(f"Date: {date_obj.strftime('%B %d, %Y (%A)')}\n")
                    f_out.write("-" * len(f"Date: {date_obj.strftime('%B %d, %Y (%A)')}") + "\n")
                    f_out.write(f"{summary_text}\n\n\n")
            print(f"\nSuccessfully exported summaries to: {SUMMARY_OUTPUT_FILENAME_CUSTOM}")
        except IOError as e:
            print(f"\nError: Could not write summaries to file '{SUMMARY_OUTPUT_FILENAME_CUSTOM}'. Reason: {e}")
    else:
        print("\nNo summaries were generated for the selected custom range.")


# --- Minimal stubs for helper functions (replace with your full versions) ---
def parse_whatsapp_line(line: str):
    match = re.match(LINE_PATTERN, line)
    if not match: return None
    date_str, time_str, sender, message = match.groups()
    date_format_str = "%m/%d/%y"
    if len(date_str.split('/')[-1]) == 4: date_format_str = "%m/%d/%Y"
    time_format_str = "%H:%M"
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", f"{date_format_str} {time_format_str}".strip())
    except ValueError:
        try:
            alt_date_format_str = "%d/%m/%y" if date_format_str == "%m/%d/%y" else "%d/%m/%Y"
            dt = datetime.strptime(f"{date_str} {time_str}", f"{alt_date_format_str} {time_format_str}".strip())
        except ValueError:
            return None
    return dt, sender.strip(), message.strip()

def load_chat_to_df(path: str) -> pd.DataFrame:
    records = []
    current_message_parts = []
    last_dt, last_sender = None, None
    try:
        with open(path, encoding="utf-8") as f:
            for line_content in f:
                line = line_content.strip()
                if not line: continue
                parsed_header = parse_whatsapp_line(line)
                if parsed_header:
                    if last_dt and current_message_parts:
                        records.append((last_dt, last_sender, "\n".join(current_message_parts)))
                    last_dt, last_sender, first_message_part = parsed_header
                    current_message_parts = [first_message_part]
                elif last_dt:
                    current_message_parts.append(line)
            if last_dt and current_message_parts:
                records.append((last_dt, last_sender, "\n".join(current_message_parts)))
    except FileNotFoundError:
        print(f"Error: Chat file not found at {path}")
        print("Please ensure CHAT_FILE_PATH in the script points to your WhatsApp chat export .txt file.")
        return pd.DataFrame(columns=["datetime", "sender", "message"])
    df = pd.DataFrame(records, columns=["datetime", "sender", "message"])
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by="datetime").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["datetime", "sender", "message"])
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def group_by_day(df: pd.DataFrame) -> dict[DDate, pd.DataFrame]:
    if df.empty or "datetime" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        return {}
    df_copy = df.copy()
    df_copy["date_only"] = df_copy["datetime"].dt.date
    grouped = {}
    for date_val, group in df_copy.groupby("date_only"):
        grouped[date_val] = group.drop(columns=["date_only"]).reset_index(drop=True)
    return grouped

def send_to_llm(prompt_text: str, system_message: str, max_summary_tokens: int, model_id: str, is_final_summary: bool = False) -> str | None:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_summary_tokens,
        "temperature": 0.15,
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=HEADERS, timeout=240)
        resp.raise_for_status()
        response_json = resp.json()
        if "choices" in response_json and len(response_json["choices"]) > 0 and \
           "message" in response_json["choices"][0] and "content" in response_json["choices"][0]["message"]:
            return response_json["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error: Unexpected LLM response structure: {response_json}")
            return None
    except requests.exceptions.Timeout:
        print("Error: LLM API request timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error: LLM API request failed with HTTPError: {e}")
        if e.response is not None:
            print(f"LLM Response (if any): {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: LLM API request failed: {e}")
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing LLM API response: {e}")
        return None

# Modified to accept primary_sender1 and primary_sender2
def summarize_day_with_llm(date_obj: DDate, day_messages_text: str, model_id: str,
                           primary_sender1: str, primary_sender2: str) -> str:
    """
    Summarizes a day's messages using the provided primary sender names.
    If the text is too long, it chunks the text, summarizes each chunk,
    and then summarizes the summaries.
    """
    total_chars = len(day_messages_text)

    if total_chars <= CHUNK_TARGET_CHAR_LENGTH * 1.2:
        print(f"Attempting direct summary for {date_obj} (length: {total_chars} chars).")
        system_prompt_direct = (
            "You are an expert assistant that summarizes chat conversations. "
            f"Pay close attention to attributing statements to the correct speaker, especially distinguishing between '{primary_sender1}' and '{primary_sender2}'. " # Use variables
            "The chat transcript is for a single day."
        )
        user_prompt_direct = (
            f"The following is a transcript of WhatsApp messages primarily between {primary_sender1} and {primary_sender2} from {date_obj.strftime('%B %d, %Y')}. " # Use variables
            "Please provide a concise summary of the main topics they discussed. "
            f"It is crucial to correctly attribute statements, questions, or opinions to the specific person ({primary_sender1} or {primary_sender2}) who expressed them. " # Use variables
            "Focus on key events, decisions, or significant information shared.\n\n"
            "Transcript (each line starts with the sender's name followed by a colon):\n"
            "----------\n"
            f"{day_messages_text}\n"
            "----------\n"
            f"Concise Summary of the Day (strictly attributing actions and words to either {primary_sender1} or {primary_sender2} based on the transcript):" # Use variables
        )
        summary = send_to_llm(user_prompt_direct, system_prompt_direct, MAX_TOKENS_FINAL_SUMMARY, model_id, is_final_summary=True)
        return summary if summary else f"Error: Could not generate direct summary for {date_obj}."

    print(f"Text for {date_obj} is too long ({total_chars} chars). Implementing chunked summarization.")
    chunk_summaries = []
    current_pos = 0
    chunk_num = 1
    temp_full_text = day_messages_text

    while current_pos < total_chars:
        end_pos = min(current_pos + CHUNK_TARGET_CHAR_LENGTH, total_chars)
        actual_end_pos = temp_full_text.rfind('\n', current_pos, end_pos + 1)
        if actual_end_pos == -1 or actual_end_pos < current_pos + CHUNK_TARGET_CHAR_LENGTH * 0.7:
            actual_end_pos = end_pos

        chunk_text = temp_full_text[current_pos:actual_end_pos].strip()
        if not chunk_text:
            current_pos = actual_end_pos
            continue

        print(f"  Summarizing chunk {chunk_num} for {date_obj} (chars: {len(chunk_text)} from {current_pos}-{actual_end_pos})...")
        system_prompt_chunk = (
            "You are an assistant that summarizes parts of a day's chat conversation. "
            "Focus on key information, decisions, and questions in this specific segment. "
            f"Mention who ('{primary_sender1}' or '{primary_sender2}') said what. This is one part of a larger conversation." # Use variables
        )
        user_prompt_chunk = (
            f"This is a segment of a WhatsApp conversation between {primary_sender1} and {primary_sender2} from {date_obj.strftime('%B %d, %Y')}. " # Use variables
            "Please summarize the key points discussed in THIS SEGMENT ONLY. Be concise.\n\n"
            "Chat Segment:\n"
            "------------\n"
            f"{chunk_text}\n"
            "------------\n"
            "Summary of this Segment:"
        )
        chunk_summary = send_to_llm(user_prompt_chunk, system_prompt_chunk, MAX_TOKENS_PER_CHUNK_SUMMARY, model_id)
        if chunk_summary:
            chunk_summaries.append(chunk_summary)
        else:
            chunk_summaries.append(f"[Error summarizing chunk {chunk_num}]")

        current_pos = max(current_pos + CHUNK_TARGET_CHAR_LENGTH - CHUNK_OVERLAP_CHAR_LENGTH, actual_end_pos)
        if current_pos >= actual_end_pos and actual_end_pos < total_chars:
             current_pos = actual_end_pos
        chunk_num += 1
        if current_pos >= total_chars:
            break

    if not chunk_summaries:
        return f"Error: No summaries generated from chunks for {date_obj}."

    print(f"  Generated {len(chunk_summaries)} chunk summaries for {date_obj}. Now summarizing the summaries...")
    combined_chunk_summaries_text = "\n\n".join(
        f"Summary of Segment {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    if len(combined_chunk_summaries_text) > CHUNK_TARGET_CHAR_LENGTH * 1.5 :
        print(f"Warning: Combined chunk summaries for {date_obj} are very long ({len(combined_chunk_summaries_text)} chars). Truncating for final summary.")
        combined_chunk_summaries_text = combined_chunk_summaries_text[:int(CHUNK_TARGET_CHAR_LENGTH*1.5)] + "\n... (Summaries Truncated)"

    system_prompt_final = (
        "You are an expert summarizer. You will be given a series of summaries, each covering a segment of a day's WhatsApp conversation between {primary_sender1} and {primary_sender2}. " # Use variables
        "Your task is to synthesize these segment summaries into a single, coherent, and concise overview of the entire day's discussion. "
        f"Ensure to attribute key points to {primary_sender1} or {primary_sender2}. Highlight the most important topics, decisions, and outcomes." # Use variables
    )
    user_prompt_final = (
        f"The following are summaries of consecutive segments from a WhatsApp conversation that occurred on {date_obj.strftime('%B %d, %Y')} between {primary_sender1} and {primary_sender2}. " # Use variables
        "Please synthesize these into a single, well-organized, and concise summary for the entire day. "
        "Attribute statements to the correct person.\n\n"
        "Segment Summaries:\n"
        "------------------\n"
        f"{combined_chunk_summaries_text}\n"
        "------------------\n"
        "Overall Concise Summary of the Day:"
    )
    final_summary = send_to_llm(user_prompt_final, system_prompt_final, MAX_TOKENS_FINAL_SUMMARY, model_id, is_final_summary=True)
    return final_summary if final_summary else f"Error: Could not generate final summary from chunks for {date_obj}."

if __name__ == "__main__":
    main()