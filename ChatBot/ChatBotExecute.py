from flask import Flask, request, jsonify,g
from flask_cors import CORS
from transformers import AutoTokenizer, BertModel,BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as FT
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "modelNews")
TOKENIZER_PATH = os.path.join(BASE_DIR, "modelNewsTokenizer")

#Criando o modelo multitarefa (Idenfificar classificação e assunto)

class BertSingleTask(BertPreTrainedModel):
    def __init__(self, config, numLabelsClassification):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classification = nn.Linear(self.bert.config.hidden_size, numLabelsClassification)

    def forward(self, input_ids, attention_mask, token_type_ids = None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        pooled_output = self.dropout(outputs.pooler_output)
        logitsClassification = self.classification(pooled_output)

        return {'logits': (logitsClassification)}

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
df = pd.read_json(os.path.join(BASE_DIR,"DataSet", "queimadas.json"))
labelEncoderClassification = LabelEncoder()
labelEncoderClassification.fit(df["classificacão"])

config = BertConfig.from_pretrained("modelNews")

model = BertSingleTask.from_pretrained(
    MODEL_PATH,
    config = config,
    numLabelsClassification = len(labelEncoderClassification.classes_),
    ignore_mismatched_sizes=True,
)

model.eval()


def classificationNews(message):

    inputs = tokenizer(message, return_tensors = "pt", truncation = True, padding = True, max_length = 512)
    with torch.no_grad():
        outputs = model(**inputs)
        logitsClassification = outputs["logits"]
        probsClassification = FT.softmax(logitsClassification, dim = 1)


        predictedClass = torch.argmax(probsClassification, dim = 1).item()

        classPreditd = labelEncoderClassification.inverse_transform([predictedClass])[0]

    return classPreditd

def identifierGreeting(message):
    message = message.lower()
    greeting = ['oi', 'ola', 'bom dia', 'boa tarde', 'boa noite', 'eai', 'oie', 'hi', 'hello', 'e aí', 'fala', "olá"]
    thanks = ['obrigado', 'muito obrigado', 'valeu', 'thanks', 'obrigada', 'muito obrigada']
    closures = ['tchau', 'até mais', 'falou', 'flw']

    if any(re.search(rf"\b{re.escape(word)}\b", message) for word in greeting):
        return "greeting"
    elif any(re.search(rf"\b{re.escape(word)}\b", message) for word in thanks):
        return "thanks"
    elif any(re.search(rf"\b{re.escape(word)}\b", message) for word in closures):
        return "closures"

@app.route("/mensagem", methods=["POST"])

def response():
    userMessage = request.json.get("message")
    print(userMessage)
    type = identifierGreeting(userMessage)
    if type == "greeting":
        response ="👋 Olá! Como posso te ajudar hoje? Me envie uma notícia para classificar."
    elif type == "thanks":
        response = "😊 De nada! Estou por aqui se precisar."
    elif type == "closures":
        response = "👋 Até logo! Se cuide."
    else: 
        predictedClass = classificationNews(userMessage)
        response = f"Essa noticia é {predictedClass.upper()}."

    return jsonify({
        "response": response
    })

networkLogs = []
@app.before_request
def beforeRequest():
    g.start_time = datetime.now()

@app.after_request
def afterRequest(response):
    duration = (datetime.now() - g.start_time).total_seconds()
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": request.method,
        "path": request.path,
        "status": response.status_code,
        "protocol": request.environ.get("SERVER_PROTOCOL"),
        "duration": round(duration, 3)
    }
    networkLogs.append(log)
    return response

@app.route("/network-logs")
def get_network_logs():
    return jsonify(networkLogs)


if __name__ == "__main__":
    print("🚀 Servidor Flask rodando em http://localhost:3000")
    app.run(port=3000, debug=True)