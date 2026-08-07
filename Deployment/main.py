from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
import gradio as gr
import torch
import torch.nn.functional as F

tokenizer = DistilBertTokenizer.from_pretrained("../final-model")
model = DistilBertForSequenceClassification.from_pretrained("../final-model")

model.eval()

def predict(question, a, b, c, d, e):
    text = (
        f"Question: {question} "
        f"Options: "
        f"A) {a} "
        f"B) {b} "
        f"C) {c} "
        f"D) {d} "
        f"E) {e}"
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    labels = ["A", "B", "C", "D", "E"]

    prediction = labels[torch.argmax(probs).item()]

    confidence = "\n".join(
        f"{labels[i]} : {probs[i] * 100:.2f}%"
        for i in range(5)
    )

    return prediction, confidence

display = gr.Interface(
    fn = predict,
    inputs = [
        gr.Textbox(label = "Question", lines = 5),
        gr.Textbox(label = "Option A"),
        gr.Textbox(label = "Option B"),
        gr.Textbox(label = "Option C"),
        gr.Textbox(label = "Option D"),
        gr.Textbox(label = "Option E")
    ],

    outputs=[
        gr.Textbox(label="Predicted Answer"),
        gr.Textbox(label="Confidence Scores", lines=5)
    ]

)

display.launch()


