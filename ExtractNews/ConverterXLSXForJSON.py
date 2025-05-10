import pandas as pd

arqExecl = 'DataSet/newsLastFiveDays.xlsx'

df = pd.read_excel(arqExecl)
df["noticia"] = df["titulo"] + ". " + df["conteudo"]
df['data'] = pd.to_datetime(df['data'], unit='ms').dt.strftime('%Y-%m-%d')

columsFromJson = ["noticia", "data", "classificacão", "assunto"]

dfColums = df[columsFromJson]

jsonOutput = dfColums.to_json(orient = "records", force_ascii = False, indent = 4)

with open("newsLastFiveDays.json", "w", encoding = 'utf-8') as f:
    f.write(jsonOutput)


print("Arquivo JSON criado com sucesso!")