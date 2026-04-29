import requests
import json
import csv
from datetime import datetime, timezone
import time
import os
import logging
import urllib.parse
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# CONSTANTES
# Centralizadas no topo para facilitar ajustes sem mexer no código
# ─────────────────────────────────────────────

OUTPUT_DIR = "./"            # pasta raiz onde todos os dados serão salvos
MAX_SNOWBALL_DEPTH = 2          # número máximo de ciclos de expansão do snowball

MAX_POSTS_PER_SUBREDDIT = 200   # máximo de posts coletados por subreddit
MAX_USERS_PER_SUBREDDIT = 10    # amostra de usuários mais ativos por subreddit
SORT_BY = "hot"                 # ordenação dos posts: "hot", "new", "top", "rising"

MAX_POSTS_PER_USER = 20         # máximo de posts coletados por usuário

REQUEST_TIMEOUT = 20            # segundos antes de desistir de uma requisição
DELAY_BETWEEN_REQUESTS = 2      # segundos de espera entre requisições (evita rate limit)


# ─────────────────────────────────────────────
# SEED — Subreddits iniciais do snowball
# Selecionados manualmente com base na literatura sobre o manosphere
# (Farrell et al. 2019, Ribeiro et al. 2020)
# ─────────────────────────────────────────────

SEED_SUBREDDITS = [
    "depression",
    "Anxiety",
    "addiction",
    "teenagers",
    "nosurf",
    "selfimprovement",
    "selfhelp",
    "study",
    "studytips",
    "getdisciplined",
    "productivity",
    "digitalminimalism",
    "simpleliving"
]

SEARCH_QUERY = '(tiktok OR reels OR "youtube shorts" OR "short videos" OR "short-form" OR doomscrolling OR brainrot)'

# Cabeçalho HTTP enviado em cada requisição
# Simula um browser real para evitar bloqueios do Reddit
# Sem isso, o requests envia "python-requests/x.x" que é bloqueado
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}


# ─────────────────────────────────────────────
# LOGGING
# Registra todas as mensagens no terminal E em arquivo simultaneamente
# Permite monitorar a coleta em tempo real e revisar o histórico depois
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,                        # mostra INFO, WARNING e ERROR (ignora DEBUG)
    format="%(asctime)s %(message)s",          # formato: "2026-03-31 00:21:09 mensagem"
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/scraper.log"),  # salva em arquivo
        logging.StreamHandler()                             # mostra no terminal
    ]
)

# ─────────────────────────────────────────────
# CHECKPOINT
# Salva a profundidade atual do snowball em disco
# Se o código quebrar, retoma de onde parou ao invés de começar do zero
# ─────────────────────────────────────────────

def save_checkpoint(depth):
    with open(f"{OUTPUT_DIR}/checkpoint.json", "w") as f:
        json.dump({"depth": depth}, f)

def load_checkpoint():
    # Retorna a profundidade salva, ou 0 se não houver checkpoint
    path = f"{OUTPUT_DIR}/checkpoint.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["depth"]
    return 0

# ─────────────────────────────────────────────
# I/O — SUBREDDITS (JSON)
# Cada subreddit é salvo em um arquivo JSON separado por profundidade
# Estrutura: data2/{depth}/subreddits/{subreddit}.json
# Salvamento incremental: se o código quebrar, os dados já coletados são preservados
# ─────────────────────────────────────────────

def save_subreddit_data(all_data, depth, subreddit_name):
    dir_ = f"{OUTPUT_DIR}/{depth}/subreddits"
    os.makedirs(dir_, exist_ok=True)           # cria a pasta se não existir
    file = f"{dir_}/{subreddit_name}.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)  # ensure_ascii=False preserva acentos
    logging.info(f"Dados salvos em {file} ({len(all_data)} posts)")

# ─────────────────────────────────────────────
# I/O — USUÁRIOS (CSV por profundidade)
# Todos os usuários de uma profundidade são salvos em um único CSV
# Isso evita criar milhares de arquivos individuais
# Estrutura: data2/{depth}/users_depth_{depth}.csv
# ─────────────────────────────────────────────

def get_users_csv_path(depth):
    # Centraliza o padrão do nome do arquivo — mudança em um lugar só
    return f"{OUTPUT_DIR}/{depth}/users_depth_{depth}.csv"

def save_user_data(posts, user_name, depth):
    path = get_users_csv_path(depth)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # garante que a pasta existe
    file_exists = os.path.exists(path)

    # Colunas do CSV de usuários — define ordem e quais campos são salvos
    CSV_FIELDS = ["depth", "user_name", "id", "subreddit"]

    # "a" = append — adiciona no final sem apagar dados anteriores
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()  # escreve cabeçalho só na primeira vez
        for post in posts:
            # {**post, ...} expande o dict do post e adiciona depth e user_name
            writer.writerow({**post, "depth": depth, "user_name": user_name})

    logging.info(f"  {user_name}: {len(posts)} posts adicionados em users_depth_{depth}.csv")

def users_scraped(depth):
    """
    Retorna set com todos os usuários já coletados até a profundidade atual.
    Lê todos os CSVs de profundidades anteriores + atual para evitar reprocessar
    usuários já visitados mesmo que em profundidades diferentes.
    """
    users = set()
    for d in range(depth+1):
        path = get_users_csv_path(d)
        if not os.path.exists(path):
            print(f"Arquivo {path} não encontrado, pulando...")
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                users.add(row["user_name"])
    return users

def sample_users(posts, n):
    """
    Seleciona os N usuários mais ativos (maior número de posts) de uma lista de posts.
    Usuários deletados são ignorados.
    Metodologicamente: representa os membros mais engajados da comunidade.
    """
    from collections import Counter

    # Conta quantos posts cada autor tem na lista
    author_counts = Counter(
        post.get("author") for post in posts
        if post.get("author") and post.get("author") != "[deleted]"
    )

    # most_common(n) retorna os n mais frequentes em ordem decrescente
    return [author for author, _ in author_counts.most_common(n)]

# ─────────────────────────────────────────────
# REQUEST
# Faz requisições HTTP com retry automático e tratamento de erros
# Retorna um objeto BeautifulSoup (HTML parseado) ou None em caso de falha
#
# Códigos de erro comuns:
# 403 — Subreddit privado ou conteúdo adulto bloqueado
# 404 — Subreddit banido ou usuário deletado
# 429 — Rate limit: muitas requisições em pouco tempo
# 503 — Reddit sobrecarregado
# ─────────────────────────────────────────────

def make_request(url: str, retries: int = 10):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                # Transforma o HTML cru em objeto navegável
                # html.parser é o interpretador padrão do Python (sem dependências extras)
                return BeautifulSoup(resp.text, "html.parser")
            elif resp.status_code == 429:
                # Backoff linear: espera mais a cada tentativa falha
                # tentativa 0 → 10s, tentativa 1 → 20s, ..., tentativa 9 → 100s
                wait = (attempt + 1) * 10
                logging.info(f"  [429] Rate limited. Aguardando {wait}s...")
                time.sleep(wait)
                # Não retorna None — continua tentando após esperar
            else:
                # 403, 404, 503: erros que não melhoram com retry
                logging.info(f"  [HTTP {resp.status_code}] {url}")
                return None
        except requests.RequestException as e:
            # Erros de rede: timeout, sem conexão, DNS falhou
            # Espera 5s fixos e tenta novamente
            logging.error(f"  [Erro] {e} — tentativa {attempt + 1}/{retries}")
            time.sleep(5)
    # Esgotou todas as tentativas sem sucesso
    return None

# ─────────────────────────────────────────────
# PARSE
# Extrai dados de um elemento <div class="thing"> do old.reddit
# O parâmetro features controla quais campos são extraídos,
# permitindo coletas mais leves para usuários (só subreddit)
# vs. coletas completas para subreddits (todos os campos)
# ─────────────────────────────────────────────

def parse_post(post, features):
    try:
        infos = {}

        # Cada campo só é extraído se estiver na lista de features
        if "title" in features: infos['title'] = post.find("a", class_="search-title")
        if "author" in features: infos['author'] = post.get("data-author")
        if "subreddit" in features: infos['subreddit'] = post.get("data-subreddit")
        if "score" in features: infos['score'] = post.get("data-score")
        # data-fullname é o ID único do post no Reddit (ex: "t3_abc123")
        if "post_id" in features: infos['post_id'] = post.get("data-fullname")
        if "comments" in features: infos['comments'] = post.find("a", string=lambda x: x and "comment" in x.lower())
        if "timestamp_raw" in features:
            # data-timestamp vem em milissegundos — divide por 1000 para converter para segundos
            timestamp_raw = post.get("data-timestamp")
            infos['timestamp'] = datetime.fromtimestamp(int(timestamp_raw) / 1000, tz=timezone.utc).isoformat() if timestamp_raw else None

        return {
            "id": infos.get('post_id'),
            "title": infos.get('title').text if infos.get('title') else None,
            "author": infos.get('author'),
            "subreddit": infos.get('subreddit').lower() if infos.get('subreddit') else None,  # normaliza case
            "timestamp": infos.get('timestamp'),
            "score": infos.get('score'),
            "comments_text": infos.get('comments').text if infos.get('comments') else None,
            "url": infos.get('title')["href"] if infos.get('title') else None
        }
    except Exception as e:
        logging.error(f"Erro ao parsear post: {e}")
        return None

# ─────────────────────────────────────────────
# SCRAPING — SUBREDDITS
# Coleta posts de uma lista de subreddits com paginação
# Usa seen_ids para evitar duplicatas entre páginas
# Salva incrementalmente — cada subreddit salvo ao terminar
# ─────────────────────────────────────────────

def scrape_subreddits(subreddits, depth, max_posts=MAX_POSTS_PER_SUBREDDIT):
    # Coleta todos os campos para análise completa dos posts
    FEATURES_SUBREDDIT = ["title", "author", "subreddit", "score", "post_id", "comments", "timestamp_raw"]

    for subreddit in subreddits:
        all_data = []
        after = None      # cursor de paginação — data-fullname do último post da página
        seen_ids = set()  # IDs já coletados para evitar duplicatas entre páginas

        logging.info(f"Scraping subreddit: {subreddit}")

        while len(all_data) < max_posts:
            safe_query = urllib.parse.quote(SEARCH_QUERY)

            url = f"https://old.reddit.com/r/{subreddit}/search?q={safe_query}&restrict_sr=on&sort=/{SORT_BY}&t=all"
            if after:
                # Paginação: ?after=t3_abc passa o cursor para a próxima página
                url += f"&after={after}"

            try:
                soup = make_request(url)
                if soup is None:
                    break

                # Busca todos os elementos <div class="thing"> — cada um é um post
                posts = soup.find_all("div", class_="search-result")
                if not posts:
                    break

                new_posts = []
                for post in posts:
                    data = parse_post(post, FEATURES_SUBREDDIT)
                    if data and data["id"] not in seen_ids:
                        seen_ids.add(data["id"])
                        new_posts.append(data)

                # Se não há posts novos, chegou ao fim do conteúdo disponível
                if not new_posts:
                    logging.info(f"  {subreddit}: sem posts novos, encerrando")
                    break

                all_data.extend(new_posts)
                after = posts[-1].get("data-fullname")  # cursor para próxima página
                logging.info(f"  {len(all_data)} posts coletados...")
                time.sleep(DELAY_BETWEEN_REQUESTS)

            except Exception as e:
                logging.error(f"ERROR em {subreddit}: {e}")
                break

        save_subreddit_data(all_data, depth, subreddit)

# ─────────────────────────────────────────────
# SCRAPING — USUÁRIOS
# Para cada subreddit coletado, seleciona os N usuários mais ativos
# e coleta apenas o subreddit de cada post deles (para descobrir novas comunidades)
# ─────────────────────────────────────────────

def scrape_users(depth, max_posts=MAX_POSTS_PER_USER):
    users = users_scraped(depth)  # usuários já coletados (não reprocessa)
    # Para usuários, coleta apenas subreddit e id — suficiente para o snowball
    FEATURES_USER = ["subreddit", "post_id"]

    for filename in os.listdir(f"{OUTPUT_DIR}/{depth}/subreddits"):
        if filename.endswith(".json"):
            path = os.path.join(f"{OUTPUT_DIR}/{depth}/subreddits", filename)
            with open(path, "r", encoding="utf-8") as f:
                posts = json.load(f)
            logging.info(f"{filename}: {len(posts)} posts")

            # Amostra os N usuários mais ativos ao invés de todos
            # Controla a explosão exponencial do snowball
            sampled_users = sample_users(posts, MAX_USERS_PER_SUBREDDIT)
            logging.info(f"  Amostra: {len(sampled_users)} usuários mais ativos de {len(posts)} posts")

            for user_name in sampled_users: 
                if not user_name or user_name == "[deleted]":
                    continue

                if user_name in users:
                    continue  # já coletado em profundidade anterior ou atual

                users.add(user_name)
                all_data = []
                after = None
                seen_ids = set()

                while len(all_data) < max_posts:
                    # over18=yes bypassa o bloqueio de conteúdo adulto do old.reddit
                    url = f"https://old.reddit.com/user/{user_name}/submitted/?over18=yes"
                    if after:
                        url += f"&after={after}"  # & ao invés de ? porque já há parâmetros na URL

                    try:
                        soup = make_request(url)
                        if soup is None:
                            break

                        user_posts = soup.find_all("div", class_="thing")
                        if not user_posts:
                            break

                        new_posts = []
                        for user_post in user_posts:
                            data = parse_post(user_post, FEATURES_USER)
                            # Remove chaves com valor None para não salvar campos vazios
                            data = {k: v for k, v in data.items() if v is not None}
                            # data.get("id") garante que não quebra se "id" foi removido
                            if data and data.get("id") and data["id"] not in seen_ids:
                                seen_ids.add(data["id"])
                                new_posts.append(data)

                        if not new_posts:
                            logging.info(f"  {user_name}: sem posts novos, encerrando")
                            break

                        all_data.extend(new_posts)
                        after = user_posts[-1].get("data-fullname")
                        logging.info(f"  {user_name}: {len(all_data)} posts coletados...")
                        time.sleep(DELAY_BETWEEN_REQUESTS)

                    except Exception as e:
                        logging.error(f"ERROR em {user_name}: {e}")
                        break

                save_user_data(all_data, user_name, depth)

# ─────────────────────────────────────────────
# SNOWBALL
# Controla os ciclos de expansão da coleta:
# 1. Define quais subreddits coletar nesta profundidade
# 2. Coleta posts dos subreddits
# 3. Coleta histórico dos usuários mais ativos
# 4. Salva checkpoint e repete
# ─────────────────────────────────────────────

def subreddits_to_scrape(depth):
    """
    Retorna o set de subreddits a coletar na profundidade atual:
    - Seeds iniciais (sempre incluídas)
    - Subreddits descobertos via usuários de profundidades anteriores
    - Menos os subreddits já coletados (evita reprocessamento)
    """
    subreddits = set()

    # Seeds iniciais normalizadas para lowercase
    subreddits.update(s.lower() for s in SEED_SUBREDDITS)

    # Subreddits descobertos via usuários de profundidades anteriores
    # Lê os CSVs de usuários e extrai os subreddits onde eles postaram
    for d in range(depth):
        path = get_users_csv_path(d)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subreddit = row.get("subreddit")
                if subreddit:
                    subreddits.add(subreddit.lower())

    # Remove subreddits já coletados em qualquer profundidade anterior
    # Varre todas as pastas de subreddits para não reprocessar
    for d in range(depth + 1):
        dir_ = f"{OUTPUT_DIR}/{d}/subreddits"
        if not os.path.exists(dir_):
            continue
        for filename in os.listdir(dir_):
            if filename.endswith(".json"):
                subreddits.discard(filename.replace(".json", "").lower())

    return subreddits


def snow_ball():
    start_depth = load_checkpoint()
    if start_depth > 0:
        logging.info(f"Retomando do checkpoint: profundidade {start_depth}")

    for depth in range(start_depth, MAX_SNOWBALL_DEPTH + 1):
        subreddits = subreddits_to_scrape(depth)

        logging.info(f"\n{'='*50}")
        logging.info(f"SNOWBALL — Profundidade {depth}/{MAX_SNOWBALL_DEPTH}")
        logging.info(f"Subreddits nesta rodada: {subreddits}")
        logging.info(f"{'='*50}")

        # Critério de parada: saturação (nenhum subreddit novo encontrado)
        if not subreddits:
            logging.info("Nenhum subreddit novo. Encerrando snowball.")
            break

        scrape_subreddits(subreddits, depth)
        # scrape_users(depth)
        save_checkpoint(depth)  # salva progresso ao fim de cada profundidade completa

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

    try:
        snow_ball()
    except Exception as e:
        # Salva o checkpoint mesmo em caso de erro inesperado
        # para não perder o progresso da profundidade atual
        logging.error(f"Erro inesperado: {e}")
        save_checkpoint(load_checkpoint())

if __name__ == "__main__":
    main()
