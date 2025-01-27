import gradio as gr
from tokenizer import SanskritBPETokenizer

tokenizer = SanskritBPETokenizer("sanskrit_bpe.json")

# Example Sanskrit texts
EXAMPLES = [
    "तत् त्वम् असि",  # "Tat Tvam Asi" (Chandogya Upanishad)
    "अहं ब्रह्मास्मि",  # "Aham Brahmasmi" (Brihadaranyaka Upanishad)
    "वसुधैव कुटुम्बकम्",  # "Vasudhaiva Kutumbakam" (Maha Upanishad)
    "सत्यमेव जयते",  # "Satyameva Jayate" (Mundaka Upanishad)
    "ॐ असतो मा सद्गमय",  # "Om Asato Ma Sadgamaya" (Brihadaranyaka Upanishad)
]

def process_text(text):
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    return (
        ", ".join(map(str, encoded)),  # Encoded tokens
        decoded,  # Decoded text
        len(encoded)  # Token count
    )

interface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(label="Sanskrit Input", placeholder="Enter text or click example below..."),
    outputs=[
        gr.Textbox(label="Encoded Tokens"),
        gr.Textbox(label="Decoded Text"),
        gr.Number(label="Token Count")
    ],
    title="🕉 Sanskrit BPE Tokenizer",
    description="Custom BPE tokenizer for Sanskrit with clickable examples:",
    examples=EXAMPLES
)

if __name__ == "__main__":
    interface.launch()  # Fixed syntax with parentheses
