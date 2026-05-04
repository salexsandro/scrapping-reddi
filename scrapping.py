import os
import json
import csv
import time
import logging
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

OUTPUT_DIR = "./" 
DEPTH_TO_SCRAPE = 2  # Ajuste para a profundidade (pasta) que deseja ler
COMMENTS_CSV_PATH = f"{OUTPUT_DIR}/all_comments_depth_{DEPTH_TO_SCRAPE}.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}
DELAY = 2
REQUEST_TIMEOUT = 15

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ─────────────────────────────────────────────
# FUNÇÃO DE REQUISIÇÃO (Com retry)
# ─────────────────────────────────────────────

def get_html(url, retries=3):
    """Faz o request e retorna o BeautifulSoup da página do post."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            elif resp.status_code == 429:
                wait = (attempt + 1) * 10
                logging.warning(f"[429] Rate limit atingido. Esperando {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"Erro HTTP {resp.status_code} na URL: {url}")
                return None
        except Exception as e:
            logging.error(f"Erro de conexão na tentativa {attempt + 1}: {e}")
            time.sleep(5)
    return None

# ─────────────────────────────────────────────
# EXTRAÇÃO DE COMENTÁRIOS DO HTML
# ─────────────────────────────────────────────

def parse_comments(soup, post_id, subreddit):
    """
    Recebe o HTML parseado do post e extrai todos os comentários.
    Achata a hierarquia (ignora quem respondeu quem) para focar no texto.
    """
    comments_data = []
    
    # Isola a área de comentários para não raspar a descrição do post original sem querer
    comment_area = soup.find("div", class_="commentarea")
    if not comment_area:
        return comments_data

    # Busca todas as divs que representam um comentário individual
    # O old.reddit usa as classes "thing" e "comment" juntas
    comments = comment_area.find_all("div", class_="comment")

    for comment in comments:
        # 1. Extrair ID do Comentário
        comment_id = comment.get("data-fullname")
        if not comment_id:
            continue

        # 2. Extrair Texto
        # O texto real fica sempre dentro de uma <div class="md">
        text_div = comment.find("div", class_="md")
        if not text_div:
            continue
        
        text = text_div.get_text(separator=" ", strip=True)
        # Ignora comentários deletados ou removidos pela moderação
        if text in ["[deleted]", "[excluído]", "[removed]"] or not text:
            continue

        # 3. Extrair Autor
        author_tag = comment.find("a", class_="author")
        author = author_tag.text if author_tag else "[deleted]"

        # 4. Extrair Pontuação (Score)
        score_tag = comment.find("span", class_="score unvoted")
        score = score_tag.get("title") if score_tag else "0"

        # 5. Extrair Timestamp
        time_tag = comment.find("time")
        timestamp = time_tag.get("datetime") if time_tag else None

        comments_data.append({
            "post_id": post_id,
            "comment_id": comment_id,
            "subreddit": subreddit,
            "author": author,
            "timestamp": timestamp,
            "score": score,
            "text": text
        })

    return comments_data

# ─────────────────────────────────────────────
# FLUXO PRINCIPAL
# ─────────────────────────────────────────────

def scrape_comments_from_jsons():
    # Prepara o arquivo CSV para salvar os comentários
    csv_fields = ["post_id", "comment_id", "subreddit", "author", "timestamp", "score", "text"]
    file_exists = os.path.exists(COMMENTS_CSV_PATH)
    
    with open(COMMENTS_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        if not file_exists:
            writer.writeheader()

        # Localiza os arquivos JSON salvos pelo script anterior
        subreddits_dir = f"{OUTPUT_DIR}/{DEPTH_TO_SCRAPE}/subreddits"
        if not os.path.exists(subreddits_dir):
            logging.error(f"Pasta não encontrada: {subreddits_dir}")
            return

        json_files = [f for f in os.listdir(subreddits_dir) if f.endswith(".json")]

        for filename in json_files:
            filepath = os.path.join(subreddits_dir, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                posts = json.load(f)
                
            logging.info(f"Processando {filename} ({len(posts)} posts)...")

            for post in posts:
                # Monta a URL completa baseada no permalink original ou ID
                # O formato do old.reddit usa o ID assim: old.reddit.com/by_id/t3_xxxx
                # Mas se você já salvou a URL relativa no JSON, use ela:
                
                post_url = post.get("url")
                post_id = post.get("id")
                subreddit = post.get("subreddit", filename.replace(".json", ""))

                if not post_url:
                    # Se não houver url salva, constrói via ID
                    post_url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id.replace('t3_', '')}/"
                elif post_url.startswith("/r/"):
                    post_url = f"https://old.reddit.com{post_url}"

                logging.info(f"  Acessando: {post_url}")
                
                soup = get_html(post_url)
                if soup:
                    extracted_comments = parse_comments(soup, post_id, subreddit)
                    
                    # Salva os comentários rasparos imediatamente no CSV
                    for comment in extracted_comments:
                        writer.writerow(comment)
                    
                    logging.info(f"    -> {len(extracted_comments)} comentários extraídos.")
                
                # Pausa para não ser bloqueado pelo Reddit
                time.sleep(DELAY)

if __name__ == "__main__":
    scrape_comments_from_jsons()