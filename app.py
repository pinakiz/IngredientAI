from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Model setup
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model from Hugging Face
adapter_dir = "./tinyllama-howto-final"               # Your local LoRA adapter folder
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model and tokenizer (no need to download manually)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
model.eval()

# FastAPI app
app = FastAPI(title="TinyLlama How-To Ingredient Extractor API")

class PromptRequest(BaseModel):
    instruction: str

@app.get("/")
def read_root():
    return {
        "project": "TinyLlama How-To Ingredient Extractor",
        "description": "This API uses a fine-tuned TinyLlama model to extract ingredients or answer instructional queries.",
        "usage": {
            "POST /generate": {
                "description": "Generate an answer based on a given instruction.",
                "body_parameters": {
                    "instruction": "The instructional input to be answered by the model."
                },
                "example_request": {
                    "instruction": "List ingredients needed to make French toast."
                },
                "example_response": {
                    "answer": "- 2 eggs\n- 4 slices of bread\n- 1/4 cup milk\n..."
                }
            }
        }
    }

@app.post("/generate")
def generate_answer(prompt_request: PromptRequest):
    prompt = f"### Instruction:\n{prompt_request.instruction}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Answer:")[-1].strip()
    return {"answer": answer}
