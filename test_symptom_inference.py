import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import os
import pickle

# Configuration - Adjusted to match your directory structure
BASE_PATH = "/content/drive/MyDrive/AI-Medical-Diagnosis-Assistant/nlp_output"
TOKENIZER_PATH = os.path.join(BASE_PATH, "tokenizer")
MODEL_PATH = os.path.join(BASE_PATH, "checkpoints_nlp/best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Note: We need the LabelEncoder to map IDs back to Disease names
# In a real deployment, you'd save the LabelEncoder to a pickle file
def get_inference_tools():
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)

    # We need to know the number of labels to initialize the architecture
    # Based on our previous execution, there are 41 classes.
    NUM_LABELS = 41

    print(f"Loading model from: {MODEL_PATH}")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def predict(text, tokenizer, model, disease_labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(
        DEVICE
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    conf, pred_idx = torch.max(probs, dim=1)

    # Get Top 3
    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)

    results = {
        "predicted_disease": disease_labels[pred_idx.item()],
        "confidence": conf.item(),
        "top3": [
            (disease_labels[idx.item()], p.item()) for p, idx in zip(top3_probs[0], top3_indices[0])
        ],
    }
    return results


if __name__ == "__main__":
    # Define labels manually for the standalone script based on the encoder used during training
    # In the notebook context, we can extract this from 'le.classes_'
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv("data/dataset.csv")
    le = LabelEncoder()
    le.fit(df["Disease"])
    disease_labels = le.classes_

    tokenizer, model = get_inference_tools()

    examples = [
        "I have a high fever, cough, and I am sweating a lot at night.",
        "Itching and skin rash with nodal eruptions.",
        "Continuous sneezing, shivering and watering from eyes.",
        "Stomach pain, loss of appetite, and yellowish skin.",
        "Joint pain, neck pain, and dizziness.",
    ]

    print("\n--- Running Inference Examples ---\n")
    for text in examples:
        res = predict(text, tokenizer, model, disease_labels)
        print(f"Input: {text}")
        print(f"Predicted: {res['predicted_disease']} ({res['confidence']:.2%})")
        print("Top 3:")
        for d, p in res["top3"]:
            print(f"  - {d}: {p:.2%}")
        print("-" * 30)
