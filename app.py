from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import torch

# Initialize FastAPI application
app = FastAPI(
    title="Text-Summarizer App",
    description="Text Summarization using T5",
    version="1.0",
)

# Load model and tokenizer
model_path = "./saved_summary_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model.to(device)
model.eval()

# Configure HTML templates
templates = Jinja2Templates(directory=".")


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text


def summarize_dialogue(dialogue: str) -> str:
    cleaned_dialogue = clean_data(dialogue)

    if not cleaned_dialogue:
        return ""

    # T5 summarization models often expect a task prefix.
    input_text = "summarize: " + cleaned_dialogue

    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary.strip()


@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)
