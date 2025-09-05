# CRITICAL: This patch must be applied BEFORE any other imports
import importlib.metadata
import sys

# Store original functions
_original_version = importlib.metadata.version
_original_distribution = importlib.metadata.distribution

def safe_version(name):
    try:
        return _original_version(name)
    except Exception:
        # Only return fallback if we're in a compiled environment
        if getattr(sys, 'frozen', False):
            # Try to get reasonable version numbers for common packages
            fallback_versions = {
                'torch': '2.1.0',
                'transformers': '4.35.0',
                'tqdm': '4.65.0',
                'numpy': '1.24.0',
                'gradio': '4.0.0',
                'matplotlib': '3.7.0',
                'huggingface-hub': '0.17.0',
                'tokenizers': '0.14.0',
                'requests': '2.31.0',
                'packaging': '23.0',
                'pyyaml': '6.0',
                'regex': '2023.8.8',
                'safetensors': '0.4.0',
                'filelock': '3.12.0',
                'fsspec': '2023.9.0',
            }
            return fallback_versions.get(name, "1.0.0")
        else:
            # In development, re-raise the exception
            raise

def safe_distribution(name):
    try:
        return _original_distribution(name)
    except Exception:
        # Only use mock in compiled environment
        if getattr(sys, 'frozen', False):
            class MockDistribution:
                def __init__(self, name):
                    self.name = name
                    self.version = safe_version(name)
                def read_text(self, filename):
                    return ""
            return MockDistribution(name)
        else:
            # In development, re-raise the exception
            raise

# Only patch in compiled environment
if getattr(sys, 'frozen', False):
    importlib.metadata.version = safe_version
    importlib.metadata.distribution = safe_distribution

# Now import everything else
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, NllbTokenizer
import gradio as gr

# Define model and tokenizer path
if getattr(sys, 'frozen', False):  # Running as an EXE
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "t5")

# Notify user of loading
print("Please wait... Loading the model and tokenizer. This may take a moment.")

# Load model and tokenizer
try:

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = NllbTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, attn_implementation="eager")
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def generate_attention_heatmap(attn_weights, input_tokens, output_tokens):
    """Generates a heatmap showing attention weights between input and output tokens using matplotlib only."""
    attn_matrix = attn_weights.squeeze(0).mean(dim=0).cpu().numpy()
    
    # Get the actual dimensions of the attention matrix
    attn_height, attn_width = attn_matrix.shape
    
    # Ensure token lists match the attention matrix dimensions
    # Truncate or pad token lists to match attention matrix dimensions
    if len(output_tokens) > attn_height:
        output_tokens = output_tokens[:attn_height]
    elif len(output_tokens) < attn_height:
        output_tokens.extend(['<pad>'] * (attn_height - len(output_tokens)))
    
    if len(input_tokens) > attn_width:
        input_tokens = input_tokens[:attn_width]
    elif len(input_tokens) < attn_width:
        input_tokens.extend(['<pad>'] * (attn_width - len(input_tokens)))

    fig, ax = plt.subplots(figsize=(10, 5))
    heatmap = ax.imshow(attn_matrix, cmap="Blues")

    # Set ticks - use the actual matrix dimensions
    ax.set_xticks(np.arange(attn_width))
    ax.set_yticks(np.arange(attn_height))
    ax.set_xticklabels(input_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(output_tokens, fontsize=8)

    # Annotate each cell with value - use actual matrix dimensions
    for i in range(attn_height):
        for j in range(attn_width):
            ax.text(j, i, f"{attn_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=6)

    # Labels
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Output Tokens")
    ax.set_title("Attention Weights Heatmap")

    plt.tight_layout()
    heatmap_path = os.path.join(base_path, "attention_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    return heatmap_path

def gloss_to_text_multiple(gloss_text):
    gloss_sentences = gloss_text.split(".")
    translations = []
    alternative_translations = []
    tokenized_inputs = []
    tokenized_outputs = []
    predicted_words = []
    heatmap_paths = []

    num_beams = 10

    for sentence in gloss_sentences:
        sentence = sentence.strip()
        if sentence:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=True,
                    max_new_tokens=50,
                    num_beams=num_beams,
                    num_return_sequences=num_beams
                )
            main_translation = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            translations.append(main_translation)
            tokenized_inputs.append(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist()))
            tokenized_outputs.append(tokenizer.convert_ids_to_tokens(outputs.sequences[0].tolist()))
            beam_translations = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
            alternative_translations.append(beam_translations)
            for i, (token_id, logits) in enumerate(zip(outputs.sequences[0], outputs.scores)):
                if i < inputs.input_ids.shape[1]:  # Check if the index is within bounds
                    probs = torch.nn.functional.softmax(logits[0], dim=-1)
                    top_probs, top_indices = torch.topk(probs, 3)
                    top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices]
                    predicted_words.append([tokenizer.convert_ids_to_tokens(inputs.input_ids[0][i].item()),
                                           top_tokens[0], round(top_probs[0].item(), 4),
                                           top_tokens[1], round(top_probs[1].item(), 4),
                                           top_tokens[2], round(top_probs[2].item(), 4)])
            if "encoder_attentions" in outputs:
                attn_weights = outputs.encoder_attentions[-1]
                heatmap_path = generate_attention_heatmap(attn_weights, tokenized_inputs[-1], tokenized_outputs[-1])
                heatmap_paths.append(heatmap_path)
    return " ".join(translations), alternative_translations, tokenized_inputs, tokenized_outputs, predicted_words, heatmap_paths


guide_html = """
<div style="text-align: left; font-family: Arial; padding: 10px;">
    <h2>✨ ASL Gloss-to-Text Translator</h2>
    <p>This tool allows you to translate <strong>ASL glosses</strong> into natural English sentences using an AI model.</p>
</div>
"""

glossing_guide = """

Here are some basic grammar conventions and rules you can follow when writing glosses:

---

### 🔹 1. Use ALL CAPS  
Gloss words are always written in **UPPERCASE**.  
✅ `ME GO STORE`  
🚫 `Me go store`

---

### 🔹 2. Follow ASL Word Order (Time → Topic → Comment)  
ASL structure is usually:  
**[Time] → [Topic/Subject] → [Comment/Action]**

| English                        | ASL Gloss                   |
|-------------------------------|-----------------------------|
| I went to the store yesterday | `YESTERDAY ME GO STORE`     |
| My mom will cook tomorrow     | `TOMORROW MOTHER COOK`      |

---

### 🔹 3. Keep it Simple — Use Only Key Words  
ASL glosses do **not** include:
- Articles: `the`, `a`, `an`
- Helping verbs: `is`, `are`, `was`
- Contractions: `don't`, `won't`

| English              | ASL Gloss           |
|----------------------|---------------------|
| I don't like apples  | `ME NOT LIKE APPLE` |
| The boy is running   | `BOY RUN`           |

---

### 🔹 4. Use "-Q" or WH-words for Questions
For yes/no questions, end the gloss with -Q.
For wh- questions, place the question word at the end.

| English              | ASL Gloss           |
|----------------------|---------------------|
| Do you like pizza?   | `YOU LIKE PIZZA-Q`  |
| Where is your bag?   | `YOU BAG WHERE`    |
"""

gloss_examples = """

| ASL Gloss                   | English Translation                    |
|-----------------------------|----------------------------------------|
| `YESTERDAY ME EAT APPLE`    | I ate an apple yesterday.              |
| `YOU GO SCHOOL TOMORROW - Q`| Are you going to school tomorrow?      |
| `FATHER COOK DINNER`        | My father is cooking dinner.           |
| `YOU LIKE DOG -Q`           | Do you like dogs?                      |
| `MOTHER LOVE CHILD`         | The mother loves the child.            |
| `NIGHT ME GO DANCE`         | I'm going to dance tonight.            |
"""

# Gradio UI
with gr.Blocks() as demo:
    gr.HTML(guide_html)
    with gr.Accordion("📘 What is an ASL Gloss?", open=True):
        gr.Markdown("""
An **ASL gloss** is a way to write down American Sign Language (ASL) signs using simplified English keywords. It's not full English — instead, it's a written approximation of the signs used.

Glosses:
- Use **UPPERCASE** words.
- Do **not** follow standard English grammar.
- Represent the **sign order**, not the spoken/written English sentence.

---

### ✨ Example

- 👋 **Signs**: (I'm going to dance tonight)  
- 📝 **ASL Gloss**: `NIGHT ME GO DANCE`  
- 📢 **English Translation**: *I'm going to dance tonight.*
""")
        
    with gr.Accordion("🛠️ How to Use This Tool", open=True):
        gr.Markdown("""                   
### 1. Enter a Gloss Sentence  
Type your ASL gloss into the input box.  
Example: `YESTERDAY ME GO STORE`
                                       
### 2. Click the "Translate" Button  
The AI will process your gloss and convert it into a natural English sentence.
                    
### 3. View Your Result  
The translated English text will appear below.

                                 
### 4. Explore the Details (Optional)  
Use the tabs below to see:
- Alternative translations  
- Tokenization  
- Predicted output words  
- Attention heatmap (shows how the model focused on each gloss word)l
        """)
    
    with gr.Accordion("🧠 How to Write ASL Glosses", open=True):
        gr.Markdown(glossing_guide)
    with gr.Accordion("💬 Gloss-to-English Examples", open=True):
        gr.Markdown(gloss_examples)
    gloss_input = gr.Textbox(label="Enter Gloss Text")
    translate_btn = gr.Button("Translate")
    output_text = gr.Textbox(label="Translated Text")

    with gr.Accordion("Transformation Process", open=False):
        with gr.Tabs():
            with gr.Tab("Alternative Translations"):
                alternative_output = gr.JSON(label="Alternative Translations (Beam Search)")
            with gr.Tab("Tokenization"):
                token_input = gr.JSON(label="Tokenized Input")
                token_output = gr.JSON(label="Tokenized Output")
            with gr.Tab("Predicted Words"):
                word_table = gr.Dataframe(
                    headers=["Input Token", "Word 1", "Confidence 1", "Word 2", "Confidence 2", "Word 3", "Confidence 3"],
                    label="Predicted Words and Confidence Scores")
            with gr.Tab("Attention Heatmap"):
                heatmap_output = gr.Image(label="Attention Weights Heatmap")

    def process_translation(gloss_text):
        translated, alt_translations, token_inputs, token_outputs, predicted_words, heatmap_paths = gloss_to_text_multiple(gloss_text)
        heatmap_path = heatmap_paths[0] if heatmap_paths else None
        return translated, alt_translations, token_inputs, token_outputs, predicted_words, heatmap_path

    translate_btn.click(process_translation, inputs=[gloss_input],
                        outputs=[output_text, alternative_output, token_input, token_output, word_table, heatmap_output])

# Launch Gradio
if __name__ == "__main__":
    demo.launch(share=False)