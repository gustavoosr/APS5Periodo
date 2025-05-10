import transformers
print("Transformers version:", transformers.__version__)
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModel, BertPreTrainedModel, AutoConfig,DataCollatorWithPadding,EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import WEIGHTS_NAME
import os
import nlpaug.augmenter.word as naw
import random
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

#Carregando o DataSet para o treinamento
df = pd.read_json("DataSet/dataSetMeioAmbiente.json")
df ["texto"] = df["noticia"]
aug = naw.SynonymAug(aug_src = 'wordnet')

augmented_texts = []
augmented_labels = []

# Opcional: balancear classes específicas (exemplo: "ruim")
class_count = df["classificacão"].value_counts()
min_class = class_count.idxmin()

for _, row in df.iterrows():
    text = row["texto"]
    label = row["classificacão"]

    # Exemplo: só aumentar se for da classe minoritária
    if label == min_class:
        try:
            aug_text = aug.augment(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
        except:
            continue

# Criando novo DataFrame aumentado
df_aug = pd.DataFrame({
    "texto": augmented_texts,
    "classificacão": augmented_labels
})

# Concatenando original + aumentado
df = pd.concat([df, df_aug]).reset_index(drop=True)

#Codificando os rótutos
labelEncoderClassification = LabelEncoder()
labelEncoderClassification.fit(df["classificacão"])

df ["labelClassification"] = labelEncoderClassification.fit_transform(df["classificacão"])

# Separando os dados para treino e testes
trainDf, testDf, = train_test_split(df[["texto", "labelClassification"]], test_size = 0.2, random_state = 42)
trainDataSet = Dataset.from_pandas(trainDf.reset_index(drop = True))
testDataSet = Dataset.from_pandas(testDf.reset_index(drop = True))

# Realizando a tokenização do modelo
modelName = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelName)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
def tokenize(batch):
    tokenized = tokenizer(batch["texto"], padding = True, truncation = True, max_length = 512)
    tokenized["labelClassification"] = batch["labelClassification"]
    return tokenized

trainDataSet = trainDataSet.map(tokenize, batched=True)
testDataSet = testDataSet.map(tokenize, batched= True)

# Criando o modelo de multitarefa
class BertSingleTask(BertPreTrainedModel):
    def __init__(self, config, numLabelsClassification):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.3)
        self.classification = nn.Linear(self.bert.config.hidden_size, numLabelsClassification)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, input_ids, attention_mask, labelClassification = None):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logitsClassification = self.classification(pooled_output)


        if labelClassification is not None:
            lossClassification = self.loss_fct(logitsClassification,labelClassification)

            return {'loss': lossClassification, 'logits': (logitsClassification)}
        else:
            return {'logits': (logitsClassification)}

# Criando um Trainer (treino) personalizado       
class SingleTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_classification = inputs.pop("labelClassification")

        outputs = model(**inputs, labelClassification = labels_classification)
        loss = outputs["loss"]
        logitsClassification = outputs["logits"]
        if return_outputs:
            return loss, {
                "logits": (logitsClassification),
                "labels": (labels_classification),
            }
        else:
            return loss

#Criando argumentos de treinamentos
trainingArgs = TrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size = 4,
    num_train_epochs = 3,
    learning_rate= 3e-5,                         # Taxa de aprendizado maior no começo
    weight_decay= 0.1,                          # Regularização
    warmup_steps=200,
    logging_dir = "./logs",
    logging_steps = 10,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "f1_classification",
    greater_is_better= True,
    save_total_limit= 5,
    lr_scheduler_type="cosine"
)

# função para computar as  méticras
def computeMetrics(pred):
    logitsClassification = pred.predictions[0]
    labels_classification = pred.label_ids

    logitsClassification = torch.tensor(logitsClassification)
    labels_classification = torch.tensor(labels_classification)


    preds_classification = torch.argmax(logitsClassification,dim = 1)


    return{
        "accuracy_classification": accuracy_score(labels_classification, preds_classification),
        "f1_classification": f1_score(labels_classification, preds_classification, average="weighted"),
    }


# Trainer
config = AutoConfig.from_pretrained(modelName)
model = BertSingleTask(config, numLabelsClassification=len(labelEncoderClassification.classes_))

# Instanciando o modelo
trainer = SingleTaskTrainer(
    model = model,
    args = trainingArgs,
    train_dataset = trainDataSet,
    eval_dataset = testDataSet,
    compute_metrics = computeMetrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)]
)

trainer.train()
print ("Treino concluído com suscesso!")

model.save_pretrained("./modelNews", safe_serialization=False)
assert os.path.isfile(os.path.join("modelNews", WEIGHTS_NAME))

tokenizer.save_pretrained("./modelNewsTokenizer")
