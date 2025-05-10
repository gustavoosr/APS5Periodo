from flask import Flask, request, jsonify,g
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)
#Carrega dataset em Json
url = "C:/Users/gusta/Documents/Aps5Cc/DataSet/dataSetMeioAmbiente.json"
df = pd.read_json(url)
pd.options.display.max_columns = None

#Determina o dado de entrada do treinamento com a noticía
x_data = df['noticia']
#determina a saída do dado (resposta) com a classificação.
y_data = df["classificacão"]

#Codifica a saída com números (0,1,2)
le = LabelEncoder()
y_data_encoded = le.fit_transform(y_data)

#Transforma a noticia em um vetor númerico.
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(X, y_data_encoded, test_size=0.2, random_state=42)

LRModel = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
    )
LRModel.fit(x_train,y_train)


def classificationNews(message):
    messageVector = vectorizer.transform([message])  # Transformar a mensagem para o formato adequado
    prediction = LRModel.predict(messageVector)  # Classificar a notícia
    label = le.inverse_transform(prediction)
    return label[0]


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