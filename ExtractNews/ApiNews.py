import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs, unquote
from ExtractNews.Verificanoticia import classificarManchete, classificarAssunto


# def gerar_id(url):
#     return hashlib.md5(url.encode()).hexdigest()

def extrair_conteudo(soup):
    seletor = "div.article__header__content--left h2"
    paragrafos = soup.select(seletor)
    if paragrafos:
        return [
            h2.get_text(strip=True)
            for h2 in paragrafos
            if len(h2.get_text(strip=True).split()) >= 15
        ]   

def extrair_links_reais(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href = True):
        href = a["href"]

        if "/busca/click" in href:
            href = "https:" + href if href.startswith("//") else href
            try:
                qs = parse_qs(urlparse(href).query)
                if "u" in qs:
                    url_real = unquote(qs["u"][0])
                if url_real.startswith("https://www.terra.com.br"):
                    links.append(url_real)
            except Exception as e:
                 print(f"[ERRO] Falha ao decodificar link: {href} ({e})")

    print("[DEBUG] Links extraídos:")
    for l in links:
        print(" -", l)

    return list(set(links))

def coletarLinksTerra(pages):
    headers = {"User-Agent": "Mozilla/5.0"}
    links = []

    for i in range(pages):
        start = i
        url = f"https://www.terra.com.br/busca/?q=meio+ambiente#gsc.tab=0&gsc.q=queimadas&gsc.page={start}"
        print(f"[INFO] Coletando página {i+1}: {url}")

        try:
            r = requests.get(url=url, headers=headers, timeout = 10)
            newLinks = extrair_links_reais(r.text)
            print(f"[INFO] {len(newLinks)} links encontrados na página {i+1}")
            links.extend(newLinks)
            time.sleep(1.5)
        except Exception as e:
            print(f"[ERRO] Falha ao acessar a página {url}: {e}")

    links = list(set(links))
    print(f"[INFO] Total de links únicos coletados: {len(links)}")
    return links

def extrair_dados_noticia(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"[ERRO] URL não acessível: {url} (status {r.status_code})")
            return None

        soup = BeautifulSoup(r.content, "html.parser")

        # Título
        titulo_elemento = soup.find("h1")
        titulo = titulo_elemento.get_text(strip=True) if titulo_elemento else ""

        # Conteúdo
        paragrafos_filtrados = extrair_conteudo(soup)
        conteudo = " ".join(paragrafos_filtrados).strip()

        # Data
        data = soup.find("div.date")
        data = soup.select_one("span")
        data_str = data.get_text(strip=True)if data else ""

        #Rotulo
        classificacao = classificarManchete(conteudo)

        #Assunto
        assunto = classificarAssunto(titulo)

        if conteudo:
            return {
                # "id": gerar_id(url),
                "titulo": titulo,
                "conteudo": conteudo,
                "data": data_str,
                "url": url,
                "classificacão": classificacao,
                "assunto": assunto
            }
        else:
            print(f"[INFO] Nenhum conteúdo extraído de: {url}")
            return None

    except Exception as e:
        print(f"[ERRO] ao extrair {url}: {e}")
        return None    

# ---------- EXECUÇÃO PRINCIPAL ----------
if __name__ == "__main__":
    pages = 3
    urls = coletarLinksTerra(pages=pages)
    dados = []

    for url in urls:
        news = extrair_dados_noticia(url)
        if news:
            dados.append(news)
        time.sleep(2.5)

df = pd.DataFrame(dados)
df.to_csv("noticiasPoluicao.csv", index=False, encoding="utf-8", sep=";", quoting= 1)
print(f"{len(dados)} notícias salvas com sucesso.")