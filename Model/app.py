import streamlit as st
from PIL import Image
from main import process_image

# Set up the Streamlit page
st.set_page_config(page_title="InternVL OCR Processor", layout="centered")
st.title("InternVL OCR Processor")

# Upload an image file
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process button
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            extracted_text = process_image(image)
            st.subheader("Extracted Text")
            st.text_area("OCR Output", extracted_text, height=300)

        st.success("Processing Complete")

if "extracted_text" in locals() and st.button("Save Output"):
    output_dir = "outputs/"
    output_filename = f"{output_dir}output_{uploaded_file.name}.txt"
    
    with open(output_filename, "w") as f:
        f.write(extracted_text)
    
    st.success(f"Output saved to {output_filename}")
