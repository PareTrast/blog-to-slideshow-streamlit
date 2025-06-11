# app.py
print("DEBUG app.py: Starting app execution.")

import os
import torch
import streamlit as st
import re
import nltk
from nltk.tokenize import sent_tokenize
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile # NEW: Import zipfile

print("DEBUG app.py: Imports successful.")

# --- Streamlit Page Configuration (MUST BE FIRST Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="Intelligent URL Summarizer",
    page_icon="📄"
)
print("DEBUG app.py: Page config set.")

# --- PATCH FOR STREAMLIT/TORCH COMPATIBILITY ---
torch.classes.__path__ = []
print("DEBUG app.py: Torch path patched.")
# --- END PATCH ---

from summarizer_logic import get_webpage_text, generate_summary
print("DEBUG app.py: summarizer_logic imported.")


# --- HELPER FUNCTION FOR SLIDESHOW CHUNKING (remains the same) ---
def split_summary_into_chunks(summary_text, max_chars_per_chunk=350):
    """
    Splits summary text into chunks, primarily by sentences using NLTK,
    but falling back to character limit if sentences are too long or absent.
    """
    if not summary_text:
        return ["No summary available."]

    chunks = []
    current_chunk_sentences = []

    sentences = sent_tokenize(summary_text)

    if not sentences:
        print("DEBUG app.py: NLTK returned no sentences. Falling back to char-based split.")
        fallback_chunks = []
        for i in range(0, len(summary_text), max_chars_per_chunk):
            fallback_chunks.append(summary_text[i:i+max_chars_per_chunk].strip())
        return fallback_chunks

    for sentence in sentences:
        potential_chunk = " ".join(current_chunk_sentences + [sentence])

        if len(potential_chunk) <= max_chars_per_chunk:
            current_chunk_sentences.append(sentence)
        else:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences).strip())
            current_chunk_sentences = [sentence]

            if len(sentence) > max_chars_per_chunk:
                print(f"DEBUG app.py: Found a very long sentence ({len(sentence)} chars). Splitting by char.")
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences).strip())
                    current_chunk_sentences = []

                for i in range(0, len(sentence), max_chars_per_chunk):
                    chunks.append(sentence[i:i+max_chars_per_chunk].strip())
                current_chunk_sentences = []

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences).strip())

    if not chunks or any(len(c) > max_chars_per_chunk * 1.5 for c in chunks) or (len(chunks) == 1 and len(summary_text) > max_chars_per_chunk * 1.5):
        print("DEBUG app.py: Final fallback to character-based splitting (post-NLTK check).")
        fallback_chunks = []
        for i in range(0, len(summary_text), max_chars_per_chunk):
            fallback_chunks.append(summary_text[i:i+max_chars_per_chunk].strip())
        return fallback_chunks
    
    return chunks

# --- Helper function to generate an image from text (remains the same) ---
# Ensure your font_path is correct for your system or a default font is used if it fails
def generate_slide_image(text, slide_number, image_width=1080, image_height=1080, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    image = Image.new("RGB", (image_width, image_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    try:
        font_path = "arial.ttf" # Adjust as per your OS or provide a full path if needed
        font_size = 40
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"DEBUG app.py: Font not found at {font_path}. Using default font.")
        font = ImageFont.load_default()
        font_size = 20

    lines = []
    words = text.split()
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if draw.textlength(test_line, font=font) < image_width - 100:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))

    total_text_height = sum(draw.textbbox((0,0), line, font=font)[3] - draw.textbbox((0,0), line, font=font)[1] for line in lines)
    y_text = (image_height - total_text_height) / 2

    for line in lines:
        text_bbox = draw.textbbox((0,0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x_text = (image_width - text_width) / 2
        draw.text((x_text, y_text), line, font=font, fill=text_color)
        y_text += (text_bbox[3] - text_bbox[1])

    slide_num_text = f"Slide {slide_number}"
    slide_num_font_size = 25
    try:
        slide_num_font = ImageFont.truetype(font_path, slide_num_font_size)
    except IOError:
        slide_num_font = ImageFont.load_default()
        slide_num_font_size = 15
    
    slide_num_bbox = draw.textbbox((0,0), slide_num_text, font=slide_num_font)
    slide_num_width = slide_num_bbox[2] - slide_num_bbox[0]
    slide_num_x = image_width - slide_num_width - 30
    slide_num_y = image_height - (slide_num_bbox[3] - slide_num_bbox[1]) - 30
    draw.text((slide_num_x, slide_num_y), slide_num_text, font=slide_num_font, fill=text_color)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

# --- NEW: Helper function to create a zip archive from image data ---
def create_zip_archive(images_data):
    """
    Creates a zip archive from a list of dictionaries, where each dict
    contains 'name' (filename for the zip) and 'data' (image bytes).
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for img_info in images_data:
            zip_file.writestr(img_info['name'], img_info['data'])
    return zip_buffer.getvalue()


# --- Streamlit UI Elements ---
st.header("📄 Intelligent Article Summarizer", divider='rainbow')
st.write(
    "Paste a URL from a news article, blog post, or any text-heavy webpage below. "
    "Our AI will fetch the content, summarize it into key points, and present it as an interactive slideshow. "
    "Perfect for quick insights into long reads!"
)

user_url = st.text_input(
    "**Enter URL to Summarize:**",
    placeholder="e.g., https://www.nytimes.com/your-article-here",
    key="url_input"
)

# --- Conditional Logic based on User Input ---
if user_url:
    print(f"DEBUG app.py: User entered URL: {user_url}")
    results_container = st.container() 
    with results_container:
        st.info(f"Processing URL: `{user_url}`")

        with st.spinner("Fetching content and summarizing... This may take a moment for long articles."):
            webpage_content = get_webpage_text(user_url)

            if webpage_content:
                summary = generate_summary(webpage_content)
                st.subheader("Generated Summary (Slideshow):")

                summary_chunks = split_summary_into_chunks(summary, max_chars_per_chunk=350)
                if summary_chunks:
                    # Store all generated image bytes to pass to the zip function
                    all_slide_images_data = [] # List to store dicts: {'name': filename, 'data': bytes}

                    col1, col2 = st.columns([3, 1]) # Left for expanders, Right for download buttons

                    with col1: # Left column for expanders
                        for i, chunk in enumerate(summary_chunks):
                            with st.expander(f"Slide {i+1} 💡", expanded=(i==0)):
                                st.write(chunk)

                    with col2: # Right column for image download buttons and new zip download
                        st.markdown("##### Download Slides as Images:")
                        
                        for i, chunk in enumerate(summary_chunks):
                            # Generate image on demand when button is created (cached by Streamlit)
                            slide_image_bytes = generate_slide_image(chunk, i + 1)
                            all_slide_images_data.append({'name': f"slide_{i+1}.png", 'data': slide_image_bytes})
                            
                            st.download_button(
                                label=f"🖼️ Slide {i+1} Image",
                                data=slide_image_bytes,
                                file_name=f"slide_{i+1}.png",
                                mime="image/png",
                                key=f"download_slide_{i+1}" # Unique key for each button
                            )

                        # --- NEW: Download All as ZIP ---
                        # Only show the "Download All" button if there are actual images to download
                        if all_slide_images_data: 
                            st.markdown("---") # Separator
                            st.markdown("##### Download All Images:")
                            
                            # Generate the zip data here (this function also needs to be cached if called often)
                            # For simplicity, we call it directly, which might be slow if many images.
                            zip_data = create_zip_archive(all_slide_images_data)

                            st.download_button(
                                label="📦 Download All Slides as ZIP",
                                data=zip_data,
                                file_name="all_slides.zip",
                                mime="application/zip",
                                key="download_all_slides_zip"
                            )

                        st.markdown("---") # Separator after image/zip buttons
                        st.download_button( # Keep full summary text download
                            label="Download Full Summary as Text",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain",
                            help="Download the complete generated summary as a plain text file."
                        )

                    st.success("Summary generated successfully! Navigate through the slides above and download images.")
                else:
                    st.warning("Could not create slideshow as the generated summary is empty or too short. Try a different URL.")
            else:
                st.error("Failed to retrieve or process content from the URL. Please check the URL or try another one.")
else:
    st.info("💡 Enter a URL above to get started!")

st.markdown("---")

# --- About Section ---
with st.expander("About This App"):
    st.markdown("""
    This application leverages advanced **Natural Language Processing (NLP)** to provide concise summaries of web articles.
    
    **How it Works:**
    1.  You provide a URL to a text-based webpage.
    2.  The app extracts the main textual content from that page.
    3.  A powerful **Transformer model (BART-Large-CNN from Hugging Face)** is used to generate a summary. For longer articles, the model intelligently summarizes sections recursively to cover the entire content.
    4.  The summary is then broken down into easy-to-digest "slides" using NLTK's sentence tokenization and Streamlit's collapsible expanders.
    
    **Built With:**
    * [Streamlit](https://streamlit.io/) for the interactive web interface
    * [Hugging Face Transformers](https://huggingface.co/transformers/) for the summarization model
    * [PyTorch](https://pytorch.org/) as the deep learning backend
    * [NLTK](https://www.nltk.org/) for advanced text processing
    * [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) and [Requests](https://requests.readthedocs.io/en/latest/) for web content extraction
    """)

st.write("Built with ❤️ by Your Name/Alias (if desired)")