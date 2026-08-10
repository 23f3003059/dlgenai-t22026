import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import traceback
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "final-model"


if not MODEL_DIR.exists():
    zip_path = BASE_DIR / "final-model.zip"
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MODEL_DIR)
    else:
        st.error(
            "Missing model folder: neither 'final-model/' nor "
            "'final-model.zip' was found next to app.py."
        )
        st.stop()


class MCQNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_assets():

    tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    le = joblib.load(BASE_DIR / "label_encoder.pkl")
    xgb = joblib.load(BASE_DIR / "xgb_model.pkl")


    input_dim = len(tfidf.get_feature_names_out())
    mlp = MCQNet(input_dim)
    state_dict = torch.load(
        BASE_DIR / "mlp_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    mlp.load_state_dict(state_dict)
    mlp.eval()


    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    bert = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    bert.eval()

    return tfidf, le, xgb, mlp, tokenizer, bert


try:
    tfidf, le, xgb, mlp, tokenizer, bert = load_assets()
except Exception:

    st.error("Error loading models. Full details below:")
    st.code(traceback.format_exc())
    st.stop()


st.title("Smart MCQ Solver")
st.write("Enter your question and options below to get the top 3 predicted answers.")

prompt = st.text_area("Question Prompt:")
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A:")
    opt_b = st.text_input("Option B:")
with col2:
    opt_c = st.text_input("Option C:")
    opt_d = st.text_input("Option D:")
opt_e = st.text_input("Option E:")

if st.button("Predict Answer"):
    if not prompt or not opt_a:
        st.error("Please provide at least the prompt and Option A!")
    else:
        full_text = (
            f"Question: {prompt} Options: "
            f"A) {opt_a} B) {opt_b} C) {opt_c} D) {opt_d} E) {opt_e}"
        )


        tfidf_vec = tfidf.transform([full_text]).toarray()
        with torch.no_grad():
            mlp_logits = mlp(torch.FloatTensor(tfidf_vec))
            mlp_probs = torch.softmax(mlp_logits, dim=1).numpy()


        xgb_probs = xgb.predict_proba(tfidf_vec)


        inputs = tokenizer(
            full_text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            bert_logits = bert(**inputs).logits
            bert_probs = torch.softmax(bert_logits, dim=-1).numpy()

        final_probs = (0.4 * bert_probs) + (0.4 * xgb_probs) + (0.2 * mlp_probs)

        top_3_indices = np.argsort(final_probs[0])[-3:][::-1]
        label_map = {i: label for i, label in enumerate(le.classes_)}
        results = [label_map[idx] for idx in top_3_indices]

        st.success(f"Top Predictions: {', '.join(results)}")
        st.info(f"Confidence score of top choice: {np.max(final_probs):.2f}")