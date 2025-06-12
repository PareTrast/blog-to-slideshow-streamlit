# summarizer_logic.py
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline, AutoTokenizer
import streamlit as st
import torch

# --- Constants for model and lengths ---
# CHANGED MODEL_NAME TO DISTILBART (for performance)
MODEL_NAME = "sshleifer/distilbart-cnn-12-6" 
MAX_MODEL_INPUT_TOKENS = 1024 # DistilBART also typically has a 1024 token limit
CHUNK_OVERLAP_TOKENS = 50

# --- Output Summary Lengths (adjustable for longer summaries) ---
MIN_CHUNK_SUMMARY_LENGTH = 10
MAX_CHUNK_SUMMARY_LENGTH = 150

MIN_FINAL_SUMMARY_LENGTH = 150 # Aim for a more substantial final summary
MAX_FINAL_SUMMARY_LENGTH = 400 # Allow the final summary to be quite long


# --- Function to Load Summarizer Model and Tokenizer (remains the same) ---
@st.cache_resource
def load_summarizer_resources():
    print("DEBUG summarizer_logic: Attempting to load summarization model and tokenizer...")
    with st.status("Loading summarization model and tokenizer...", expanded=True) as status:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"DEBUG summarizer_logic: Using device: {device}")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            summarizer_pipeline = pipeline(
                "summarization",
                model=MODEL_NAME,
                tokenizer=tokenizer,
                device=device
            )
            status.update(label="Summarization model and tokenizer loaded!", state="complete", expanded=False)
            print("DEBUG summarizer_logic: Summarization model and tokenizer loaded successfully!")
            return tokenizer, summarizer_pipeline
        except Exception as e:
            status.update(label=f"Failed to load summarizer resources: {e}", state="error", expanded=True)
            print(f"DEBUG summarizer_logic: Error loading resources: {e}")
            st.error(f"Error loading model resources: {e}")
            raise

tokenizer, summarizer_pipeline_instance = load_summarizer_resources()


# --- All other functions (split_text_into_chunks_by_tokens, get_webpage_text, generate_summary) remain the same ---
# They will now use the newly loaded DistilBART model and its tokenizer.

def split_text_into_chunks_by_tokens(text, max_tokens, overlap_tokens):
    if not text:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    effective_max_tokens = max_tokens - 2
    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + effective_max_tokens, len(tokens))
        chunk_tokens = tokens[start_index:end_index]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        if end_index == len(tokens):
            break
        start_index += (effective_max_tokens - overlap_tokens)
        if start_index < 0:
            start_index = 0
    return chunks

@st.cache_data
def get_webpage_text(url):
    print(f"DEBUG summarizer_logic: Fetching content from URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content_tag = soup.find('article') or \
                           soup.find('main') or \
                           soup.find('div', {'id': 'main-content'}) or \
                           soup.find('div', {'class': 'article-body'}) or \
                           soup.find('div', {'class': 'post-content'})

        if main_content_tag:
            print("DEBUG summarizer_logic: Found a potential main content tag. Extracting text from it.")
            for tag in main_content_tag(['script', 'style', 'nav', 'footer', 'aside', 'header', 'form', 'button', 'img', 'svg']):
                tag.decompose()

            full_text = main_content_tag.get_text()
        else:
            print("DEBUG summarizer_logic: No specific main content tag found. Falling back to paragraph/heading extraction.")
            paragraphs = soup.find_all('p')
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            text_parts = [p.get_text() for p in paragraphs]
            text_parts.extend([h.get_text() for h in headings])
            full_text = "\n".join(text_parts)

        boilerplate_phrases = [
            r"About Apple\b",
            r"Apple Media Helpline",
            r"Apple’s more than \d+ employees are dedicated to making the best products on earth.",
            r"contact information",
            r"press contact",
            r"copyright",
            r"all rights reserved",
            r"terms of use",
            r"privacy policy",
            r"subscribe to our newsletter",
            r"share this article",
            r"related articles",
            r"read more",
            r"sources?",
            r"\[\d+\]"
        ]
        for phrase in boilerplate_phrases:
            full_text = re.sub(phrase, '', full_text, flags=re.IGNORECASE | re.DOTALL)
            
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        if not full_text or len(full_text) < 100:
            st.warning("Could not extract enough meaningful text from the URL. "
                       "This might be a non-textual page (e.g., image, video), paywalled content, or an issue with parsing.")
            print("DEBUG summarizer_logic: Not enough meaningful text extracted after cleaning.")
            return None
        print(f"DEBUG summarizer_logic: Successfully extracted and cleaned text (length: {len(full_text)} characters).")
        return full_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        print(f"DEBUG summarizer_logic: Request error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the URL: {e}")
        print(f"DEBUG summarizer_logic: Unexpected error during webpage processing: {e}")
        return None

@st.cache_data
def generate_summary(text):
    if text is None:
        print("DEBUG summarizer_logic: No text provided for summarization.")
        return "No text available to summarize."

    word_count = len(text.split())
    if word_count > (MAX_MODEL_INPUT_TOKENS * 0.75):
        st.info(f"The article is approximately {word_count} words long. "
                   "Summarizing long articles requires multiple passes and may take a moment.")
    print(f"DEBUG summarizer_logic: Input text length (chars): {len(text)}")
    print("DEBUG summarizer_logic: Starting recursive summarization process.")

    current_text_to_summarize = text
    summaries_from_passes = []
    pass_num = 1

    while True:
        print(f"DEBUG summarizer_logic: Summarization Pass {pass_num} starting.")
        
        chunks = split_text_into_chunks_by_tokens(current_text_to_summarize, MAX_MODEL_INPUT_TOKENS, CHUNK_OVERLAP_TOKENS)
        
        if not chunks:
            print("DEBUG summarizer_logic: No chunks to process. Breaking loop.")
            break 

        print(f"DEBUG summarizer_logic: Created {len(chunks)} chunks for pass {pass_num}.")

        chunk_summaries = []
        with st.status(f"Pass {pass_num}: Summarizing {len(chunks)} chunks...", expanded=True) as inference_status:
            for i, chunk in enumerate(chunks):
                print(f"DEBUG summarizer_logic:   Processing chunk {i+1}/{len(chunks)}. FULL Chunk content (first 500 chars): '{chunk[:500]}...'")
                print(f"DEBUG summarizer_logic:   Processing chunk {i+1}/{len(chunks)}. Chunk content length: {len(chunk)} chars.")

                try:
                    summary_result_list = summarizer_pipeline_instance(
                        chunk,
                        max_length=MAX_CHUNK_SUMMARY_LENGTH,
                        min_length=MIN_CHUNK_SUMMARY_LENGTH,
                        num_beams=4,
                        do_sample=False
                    )
                    
                    if summary_result_list and isinstance(summary_result_list, list) and len(summary_result_list) > 0 and 'summary_text' in summary_result_list[0]:
                        chunk_summaries.append(summary_result_list[0]['summary_text'])
                        print(f"DEBUG summarizer_logic:   Chunk {i+1} summarized. Summary length: {len(summary_result_list[0]['summary_text'])} chars.")
                    else:
                        print(f"DEBUG summarizer_logic:   WARNING: Summarizer returned empty or invalid result for chunk {i+1}. Appending empty string. Result: {summary_result_list}")
                        chunk_summaries.append("")
                    
                    inference_status.update(label=f"Pass {pass_num}: Summarized chunk {i+1}/{len(chunks)}", state="running", expanded=True)
                except Exception as e:
                    print(f"DEBUG summarizer_logic:   ERROR: Exception during summarization of chunk {i+1}: {e}. Appending empty string.")
                    st.warning(f"Warning: Could not summarize chunk {i+1} due to an error: {e}. Skipping this part.")
                    chunk_summaries.append("")

            inference_status.update(label=f"Pass {pass_num}: All chunks summarized!", state="complete", expanded=False)
            print(f"DEBUG summarizer_logic: Pass {pass_num} chunks summarized.")
            
        combined_summaries = " ".join(chunk_summaries)
        summaries_from_passes.append(combined_summaries)

        if not combined_summaries.strip():
            print("DEBUG summarizer_logic: Combined summaries are empty or only whitespace after pass. Cannot proceed. This might indicate poor source text or overly strict summarization.")
            break

        if len(tokenizer.encode(combined_summaries)) <= MAX_MODEL_INPUT_TOKENS:
            print(f"DEBUG summarizer_logic: Pass {pass_num} combined summary is short enough ({len(tokenizer.encode(combined_summaries))} tokens). Finalizing.")
            break

        current_text_to_summarize = combined_summaries
        pass_num += 1
        print(f"DEBUG summarizer_logic: Combined summary still too long ({len(tokenizer.encode(current_text_to_summarize))} tokens). Starting new recursive pass.")

    if not summaries_from_passes or not summaries_from_passes[-1].strip():
        return "Could not generate a meaningful summary for the provided content. The article might be too complex or contain non-summarizable content."
    
    final_input_text = summaries_from_passes[-1]
    
    print(f"DEBUG summarizer_logic: Performing final summarization pass on text of length {len(final_input_text)} chars.")
    with st.status("Finalizing summary...", expanded=True) as final_status:
        try:
            final_summary_list = summarizer_pipeline_instance(
                final_input_text,
                max_length=MAX_FINAL_SUMMARY_LENGTH,
                min_length=MIN_FINAL_SUMMARY_LENGTH,
                num_beams=8,
                do_sample=False
            )
            if final_summary_list and isinstance(final_summary_list, list) and len(final_summary_list) > 0 and 'summary_text' in final_summary_list[0]:
                final_summary_output = final_summary_list[0]['summary_text']
                final_status.update(label="Final summary generated!", state="complete", expanded=False)
                print("DEBUG summarizer_logic: Final summary generated successfully.")
            else:
                final_summary_output = "Could not generate a coherent final summary. The content might be too abstract."
                final_status.update(label="Failed to generate final summary.", state="error", expanded=False)
                print("DEBUG summarizer_logic: WARNING: Final summarization pass returned empty result.")
        except Exception as e:
            final_summary_output = f"An error occurred during final summary generation: {e}"
            final_status.update(label="Error during final summary generation.", state="error", expanded=False)
            print(f"DEBUG summarizer_logic: ERROR: Exception during final summary generation: {e}")

    print("DEBUG summarizer_logic: Recursive summarization complete. Final summary returned.")
    cleaned_summary = final_summary_output.replace("<n>", "\n").strip()
    print(f"DEBUG summarizer_logic: Cleaned summary. Original length: {len(final_summary_output)} chars. Cleaned length: {len(cleaned_summary)} chars.")
    return cleaned_summary