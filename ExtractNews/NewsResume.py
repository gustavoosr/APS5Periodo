import pandas as pd
from ExtractNews.Verificanoticia import classificarAssunto, classificarManchete, resumirNoticia

# Ler o dataset
arquivo_entrada = 'DataSet/dataSetMeioAmbiente.xlsx'
df = pd.read_excel(arquivo_entrada)

df["conteudoR"] = df["conteudo"].apply(resumirNoticia)
df["classificação"] = df["conteudo"].apply(classificarManchete)
df["assunto"] = df["conteudo"].apply(classificarAssunto)

df.to_excel("noticiasClassificadas.xlsx", index = False, engine = "openpyxl")


print("Processamento concluído! O arquivo Excel foi salvo.")
