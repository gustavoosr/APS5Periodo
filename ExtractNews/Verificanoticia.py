import openai

openai.api_key = "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"

def classificarAssunto(conteudo):
    prompt =  f"""
Você é um assistente inteligente especializado em identificar o tema principal de notícias relacionadas ao meio ambiente.
Leia a notícia a seguir e responda com **apenas o assunto principal** abordado. Escolha entre as seguintes opções:
- Queimadas
- Alagamentos
- Aquecimento global
- Desmatamento
- Tempestades
- Poluição
- Secas
- Outros (caso o assunto não se encaixe nas opções acima)
Responda somente com AS OPÇÕES FORNECIDAS!! CASO O ASSUNTO NÃO SE ENCAIXE COM ASSUNTOS DO MEIO AMBIENTE PREENCHA COM A OPÇÃO "OUTROS"!!!! (sem frases ou explicações). Caso o assunto possui dois temas, classifique com o tema mais predominante na noticia!!!!
Assunto: {conteudo}
"""
    try:

        response = openai.chat.completions.create(
            model = "gpt-4",
            messages = [{"role":"user","content": prompt}],
            temperature = 0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro ao classificar: {conteudo} \n {e}")
        return "erro"


def classificarManchete(conteudo):
    prompt =  f"""
Você é uma assistente especializada em classificar notícias ambientais como 'boas', 'ruins' ou 'irrelevantes'.
Considere toda a notícia, do início ao fim, antes de classificar.
- Uma notícia é 'boa' se o conteúdo principal apresentar uma melhoria, redução de impactos, avanço positivo ou solução para um problema ambiental — mesmo que o início mencione aspectos negativos.
- Uma notícia é 'ruim' se o conteúdo principal abordar agravamento de problemas, aumento de desmatamento, queimadas, poluição ou retrocessos ambientais.
- Uma notícia é 'irrelevante' se o conteúdo não tiver impacto direto no meio ambiente ou tratar de temas apenas tangenciais.
Exemplos:
1. Se a notícia fala sobre treinamento de produtores para prevenir queimadas, mesmo que mencione o problema das queimadas ou combate a incendios, é 'boa'.
2. Se uma área devastada teve redução de 70% nas queimadas, é 'boa', mesmo que fale de destruição no passado.
3. Uma notícia sobre uma campanha publicitária de uma ONG sem resultados claros é 'irrelevante' ou qualquer assunto que não se trata do meio ambiente.
4. Uma notícia que aborda o aumento do desmatamento, poluição e semelhantes, incentivo ao desmatamento ou corta de verbas para o combate do desmatamento e queimada, é uma notícia RUIM!!
Responda apenas com 'Boa', 'Ruim' ou 'Irrelevante', SEM FRASES OU EXPLICAÇÕES!!!
Manchete: {conteudo}
"""
    try:

        response = openai.chat.completions.create(
            model = "gpt-4",
            messages = [{"role":"user","content": prompt}],
            temperature = 0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro ao classificar: {conteudo} \n {e}")
        return "erro"

def resumirNoticia(conteudo):
    prompt = f"""Leia a seguinte notícia e faça um resumo conciso com até 3 frases. O resumo deve destacar os principais pontos e eventos da notícia de forma clara e objetiva, sem incluir detalhes desnecessários. Caso a noticia já esteja resumida, não é necessário resumir!!
    Notícia:{conteudo}
    """
    try:

        response = openai.chat.completions.create(
            model = "gpt-4",
            messages = [{"role":"user","content": prompt}],
            temperature = 0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro ao classificar: {conteudo} \n {e}")
        return "erro"
