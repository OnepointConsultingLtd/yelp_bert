## Runs a Flask endpoint which allows to call the sentiment analysis API.

import torch
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import torch
import config

import flask
from flask import Flask
from flask import request

app = Flask(__name__)

path = Path('/data/yelp/model_save')
assert path.exists()

tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
print('Created tokenizer')

def encode(sequence):
    return tokenizer.encode_plus(
                sequence,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
    )

def load_model():
    model = torch.load(path/'model')
    model.cpu()
    model.eval();
    return model
    
model = load_model()
print('Loaded model')

def predict_sentiment(sequence='I love you a lot. You are really great. You are wonderful and awesome.'):
    encoded = encode(sequence)
    with torch.no_grad():
        output = model(encoded['input_ids'].cpu(), token_type_ids=None, attention_mask=encoded['attention_mask'].cpu())[0]
        pred_flat = np.argmax(output, axis=1).flatten()
        sig_factor = torch.sigmoid(output) / torch.sigmoid(output).sum()
        return {'proportional':  sig_factor.numpy().tolist(), 'sigmoid': torch.sigmoid(output).numpy().tolist(), 'stars': pred_flat.item() + 1, 'raw': output.numpy().tolist()}

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    response = {}
    response["response"] = predict_sentiment(sentence)
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()

