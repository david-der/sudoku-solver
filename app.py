import streamlit as st
import cv2
import numpy as np
from openai import OpenAI
from PIL import Image
import io
import os
import base64
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def capture_image():
    """Capture image from webcam"""
    img_file = st.camera_input("Take a picture of the Sudoku puzzle")
    if img_file is not None:
        return Image.open(img_file)
    return None

def clean_json_response(response_text):
    """Clean the GPT response to get valid JSON"""
    # Remove markdown code blocks if present
    json_text = re.sub(r'```json\s*|\s*```', '', response_text)
    # Remove any other markdown formatting or extra text
    json_text = re.sub(r'^[^{]*', '', json_text)
    json_text = re.sub(r'[^}]*$', '', json_text)
    return json_text

def detect_and_solve_sudoku(image):
    """Send image to GPT-4 Vision API for detection and solving"""
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Convert bytes to base64
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this Sudoku puzzle image and return ONLY a JSON object with this exact format:
                                     {
                                         "initial_grid": [[n,n,n,n,n,n,n,n,n], ...],
                                         "solved_grid": [[n,n,n,n,n,n,n,n,n], ...]
                                     }
                                     Use 0 for empty cells. Do not include any explanation or markdown formatting."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Get the response text and clean it
        response_text = response.choices[0].message.content
        json_text = clean_json_response(response_text)
        
        try:
            # Parse the cleaned JSON
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response: {str(e)}")
            st.text("Raw response:")
            st.text(response_text)
            return None
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def display_grids(result):
    """Display the initial and solved grids side by side with better formatting"""
    if not result:
        return
    
    col1, col2 = st.columns(2)
    
    def format_grid(grid):
        """Format a grid with horizontal and vertical lines"""
        formatted_rows = []
        for i, row in enumerate(grid):
            # Format each number with padding
            formatted_numbers = [f" {n if n != 0 else '_'} " for n in row]
            # Add vertical lines
            row_str = '|' + '|'.join([''.join(formatted_numbers[i:i+3]) for i in range(0, 9, 3)]) + '|'
            formatted_rows.append(row_str)
            # Add horizontal line after every 3 rows
            if i % 3 == 2 and i < 8:
                formatted_rows.append('-' * len(formatted_rows[0]))
        return formatted_rows

    with col1:
        st.subheader("Original Puzzle")
        st.text('\n'.join(format_grid(result['initial_grid'])))
    
    with col2:
        st.subheader("Solved Puzzle")
        st.text('\n'.join(format_grid(result['solved_grid'])))

def main():
    st.title("Sudoku Solver with GPT Vision")
    
    upload_option = st.radio(
        "Choose input method:",
        ["Upload Image", "Camera"],
        index=0
    )
    
    image = None
    if upload_option == "Camera":
        image = capture_image()
    else:
        uploaded_file = st.file_uploader("Choose a Sudoku puzzle image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Solve Puzzle"):
            with st.spinner("Processing image and solving puzzle..."):
                result = detect_and_solve_sudoku(image)
                if result:
                    st.success("Puzzle solved!")
                    display_grids(result)  # Display formatted grids

if __name__ == "__main__":
    main()