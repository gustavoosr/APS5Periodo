import transformers
print("Transformers version:", transformers.__version__)
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModel, BertPreTrainedModel, AutoConfig,DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import WEIGHTS_NAME
import os


#Carregando o DataSet para o treinamento
df = pd.read_json("DataSet/dataSetMeioAmbiente.json")
df ["texto"] = df["noticia"]

#Codificando os rótutos
labelEncoderClassification = LabelEncoder()
labelEncoderSubject = LabelEncoder()
labelEncoderClassification.fit(df["classificacão"])
labelEncoderSubject.fit(df["assunto"])
df ["labelClassification"] = labelEncoderClassification.fit_transform(df["classificacão"])
df ["labelSubject"] = labelEncoderSubject.fit_transform(df["assunto"])

# Separando os dados para treino e testes
trainDf, testDf, = train_test_split(df[["texto", "labelClassification", "labelSubject"]], test_size = 0.2, random_state = 42)
trainDataSet = Dataset.from_pandas(trainDf.reset_index(drop = True))
testDataSet = Dataset.from_pandas(testDf.reset_index(drop = True))

# Realizando a tokenização do modelo
modelName = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelName)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
def tokenize(batch):
    tokenized = tokenizer(batch["texto"], padding = True, truncation = True, max_length = 512)
    tokenized["labelClassification"] = batch["labelClassification"]
    tokenized ["labelSubject"] = batch ["labelSubject"]
    return tokenized

trainDataSet = trainDataSet.map(tokenize, batched=True)
testDataSet = testDataSet.map(tokenize, batched= True)

# Criando o modelo de multitarefa
class BertMultiTask(BertPreTrainedModel):
    def __init__(self, config, numLabelsClassification, numLabelsSubject):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.3)
        self.classification = nn.Linear(self.bert.config.hidden_size, numLabelsClassification)
        self.subject = nn.Linear(self.bert.config.hidden_size, numLabelsSubject)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, input_ids, attention_mask, labelClassification = None, labelSubject = None):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logitsClassification = self.classification(pooled_output)
        logitsSubject = self.subject(pooled_output)

        if labelClassification is not None and labelSubject is not None:
            lossClassification = self.loss_fct(logitsClassification,labelClassification)
            lossSubject = self.loss_fct(logitsSubject, labelSubject)
            loss = lossClassification + lossSubject

            return {'loss': loss, 'logits': (logitsClassification, logitsSubject)}

        else:

            return {'logits': (logitsClassification, logitsSubject)}

# Criando um Trainer (treino) personalizado       
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_classification = inputs.pop("labelClassification")
        labels_subject = inputs.pop("labelSubject")

        outputs = model(**inputs, labelClassification = labels_classification, labelSubject = labels_subject)
        loss = outputs["loss"]
        logitsClassification,logitsSubjects = outputs["logits"]
        if return_outputs:
            return loss, {
                "logits": (logitsClassification, logitsSubjects),
                "labels": (labels_classification, labels_subject),
            }
        else:
            return loss

#Criando argumentos de treinamentos
trainingArgs = TrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size = 4,
    num_train_epochs = 3,
    learning_rate= 5e-5,                         # Taxa de aprendizado maior no começo
    weight_decay= 0.01,                          # Regularização
    warmup_steps=50,
    logging_dir = "./logs",
    logging_steps = 10,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "f1_classification",
    greater_is_better= True,
    save_total_limit= 1,
)

# função para computar as  méticras
def computeMetrics(pred):
    logitsClassification,logitsSubject = pred.predictions[0]
    labels_classification, labels_subject = pred.label_ids

    logitsClassification = torch.tensor(logitsClassification)
    logitsSubject = torch.tensor(logitsSubject)
    labels_classification = torch.tensor(labels_classification)
    labels_subject = torch.tensor(labels_subject)

    preds_classification = torch.argmax(logitsClassification,dim = 1)
    preds_subject = torch.argmax(logitsSubject, dim = 1)

    acc_classification = (preds_classification == labels_classification).float().mean().item()
    acc_subject = (preds_subject == labels_subject).float().mean().item()

    return{
        "accuracy_classification": accuracy_score(labels_classification, preds_classification),
        "f1_classification": f1_score(labels_classification, preds_classification, average="weighted"),
        "accuracy_subject": accuracy_score(labels_subject, preds_subject),
        "f1_subject": f1_score(labels_subject, preds_subject, average="weighted"),
    }


# Trainer
config = AutoConfig.from_pretrained(modelName)
model = BertMultiTask(config, numLabelsClassification=len(labelEncoderClassification.classes_), numLabelsSubject=len(labelEncoderSubject.classes_))


# Instanciando o modelo
trainer = MultiTaskTrainer(
    model = model,
    args = trainingArgs,
    train_dataset = trainDataSet,
    eval_dataset = testDataSet,
    compute_metrics = computeMetrics,
    data_collator=data_collator
)

trainer.train()
print ("Treino concluído com suscesso!")
trainer.evaluate()
model.save_pretrained("./modelNews", safe_serialization=False)
assert os.path.isfile(os.path.join("modelNews", WEIGHTS_NAME))

tokenizer.save_pretrained("./modelNewsTokenizer")
