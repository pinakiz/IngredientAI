from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Model paths
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use Hugging Face model ID
adapter_path = "./tinyllama-howto-final"              # LoRA adapter directory (must contain adapter_model.safetensors or similar)

# Load tokenizer and model with LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)

model.eval()

# FastAPI setup
app = FastAPI(title="TinyLlama How-To Ingredient Extractor API")

class PromptRequest(BaseModel):
    instruction: str

@app.get("/")
def read_root():
    return {
        "project": "TinyLlama How-To Ingredient Extractor",
        "description": "This API uses a fine-tuned TinyLlama model to extract ingredients or answer instructional queries.",
    }

@app.post("/generate")
def generate_answer(prompt_request: PromptRequest):
    prompt = f"### Instruction:\n{prompt_request.instruction}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("### Answer:")[-1].strip()
    return {"answer": answer}
