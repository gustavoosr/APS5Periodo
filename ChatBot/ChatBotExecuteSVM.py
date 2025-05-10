from flask import Flask, request, jsonify,g
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import re
import nltk
import wn
from nltk.corpus import stopwords
import random
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

#Processar informações Nltk
nltk.download('stopwords')
wn.download('own-pt')
pt = wn.Wordnet('own-pt')

stopWords = set(stopwords.words('portuguese'))

def preProcessing(text):
    text = re.sub(r'[a-zA-ZéóúâêôõçÀ-ÿs]', '', text.lower())
    words = text.split()
    wordsFiltered = [
            p for p in words
                if p not in stopWords       
    ]

    return ' '.join(wordsFiltered)

def getSynonyms(word):
    try:
        senses = pt.synsets(word, lang='pt')
        if not senses:
            return [word]
        synonyms = set()
        for sense in senses:
            for lemma in sense.lemmas(lang='pt'):
                if lemma.lower() != word.lower():
                    synonyms.add(lemma.lower())
                if len(synonyms) >= 2:
                    break
        return list(synonyms) or [word]
    except:
        return [word]


def augmentText(text, prob = 0.3):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < prob:
            sinonimos = getSynonyms(word)
            new_word = random.choice(sinonimos)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

url = "C:/Users/gusta/Documents/Aps5Cc/DataSet/dataSetMeioAmbiente.json"
df = pd.read_json(url)
#print('\nMostrando os 5 primeiros registros:')
pd.options.display.max_columns = None
print("DataSet Antes de formatado: " + df.head(5))

df ['noticia'] = df['noticia'].apply(preProcessing)

dfAug = df.copy()
dfAug['noticia'] = dfAug['noticia'].apply(lambda x: augmentText(x, prob=0.3))

dfTotal = pd.concat([df, dfAug]).reset_index(drop=True)

print ("DataSet Depois de formatado: " + dfTotal.head(10))
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(dfTotal['noticia'])
y_data = dfTotal['classificacão']

le = LabelEncoder()
y_data_encoded = le.fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(X, y_data_encoded, test_size=0.3, random_state=42)


LRModel = RandomForestClassifier()
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