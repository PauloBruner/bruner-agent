from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import tempfile
from collections import defaultdict
from openai import OpenAI
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO BÁSICA DO FLASK
# -----------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

# -----------------------------------------------------------------------------
# CONFIG OPENAI
# -----------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Memória por usuário (cada navegador tem um client_id diferente)
# conversation_histories["client-xyz"] = [ {"role": "user", "content": "..."}, ... ]
conversation_histories = defaultdict(list)

# -----------------------------------------------------------------------------
# FUNÇÕES DE IA
# -----------------------------------------------------------------------------

def gerar_resposta_agente(client_id: str) -> str:
    """
    Usa o histórico de conversa do client_id informado para gerar a próxima resposta.
    """
    if client is None:
        return (
            "Erro: a chave da OpenAI não está configurada.\n"
            "Defina a variável de ambiente OPENAI_API_KEY antes de rodar o servidor."
        )

    history = conversation_histories[client_id]

    if not history:
        return "Pode repetir? Ainda não recebi nenhuma mensagem na conversa."

    # Usa só as últimas 10 trocas pra não estourar tokens
    contexto = history[-10:]

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",  # ou outro modelo compatível da sua conta
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é uma assistente virtual chamada Bruner. "
                        "Você fala em Português Brasileiro, de forma clara, objetiva e amigável. "
                        "O usuário que está conversando com você se chama Paulo. "
                        "Sempre que for se dirigir diretamente a ele, chame-o de Paulo. "
                        "Responda normalmente quando ele disser 'Bruner' ou 'Oi Bruner'. "
                        "Quando o usuário mandar textos longos, você pode explicar, resumir "
                        "ou destacar os pontos principais."
                    ),
                },
                *contexto
            ],
            max_tokens=800,
            temperature=0.3,
        )
        texto = resposta.choices[0].message.content
        return texto.strip()
    except Exception as e:
        print("Erro ao chamar OpenAI (chat):", e)
        return f"Tive um problema ao falar com o modelo de IA.\nDetalhe técnico: {e}"


def resumir_texto(conteudo: str) -> str:
    """
    Resumo inteligente de um texto enviado em arquivo.
    """
    if client is None:
        return "Não consegui falar com a IA para resumir o texto (OpenAI não configurada)."

    trecho = conteudo[:8000]

    prompt = (
        "Resuma o texto abaixo em 5 a 8 linhas, em português claro, "
        "destacando ideias principais, tópicos importantes e conclusões.\n\n"
        f"Texto:\n{trecho}"
    )

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente que faz resumos claros e organizados "
                        "em português brasileiro."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.25,
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        print("Erro ao chamar OpenAI (resumo):", e)
        return f"Não consegui resumir o texto por um erro técnico: {e}"


def gerar_audio_openai(texto: str) -> str:
    """
    Gera um arquivo MP3 com a voz da OpenAI e devolve o caminho temporário do arquivo.
    """
    if client is None:
        raise RuntimeError("OpenAI não configurada")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=texto,
            instructions=(
                "Fale em português brasileiro, com sotaque natural do Brasil, "
                "pronúncia clara e ritmo de leitura natural."
            ),
        ) as response:
            response.stream_to_file(tmp_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    return tmp_path

# -----------------------------------------------------------------------------
# ROTAS FLASK
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    mensagem = data.get("message", "")
    client_id = data.get("client_id", "anonimo")  # <- pega o mesmo client_id do front

    if not mensagem.strip():
        return jsonify({"reply": "Pode repetir a pergunta? Não recebi nenhum texto."})

    # histórico específico desse client_id
    history = conversation_histories[client_id]

    # adiciona mensagem do usuário
    history.append({"role": "user", "content": mensagem})

    resposta = gerar_resposta_agente(client_id)

    # adiciona resposta do agente ao histórico
    history.append({"role": "assistant", "content": resposta})

    print(f"[DEBUG] chat: client_id={client_id}, mensagens_no_historico={len(history)}")

    return jsonify({"reply": resposta})

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Recebe um arquivo, lê o conteúdo (TXT ou PDF) e devolve prévia + resumo pela IA.
    Também armazena o texto completo na memória do client_id para permitir perguntas sobre ele.
    """
    arquivo = request.files.get("file")
    if not arquivo:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    # ✅ pega o client_id enviado pelo front via FormData
    client_id = request.form.get("client_id", "anonimo")

    nome = arquivo.filename or "arquivo"
    ext = os.path.splitext(nome)[1].lower()

    try:
        # Lê o conteúdo dependendo da extensão
        if ext in [".txt", ".md", ".csv", ".json", ".log"]:
            conteudo = arquivo.read().decode("utf-8", errors="ignore")

        elif ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                arquivo.save(tmp)
                tmp_path = tmp.name

            try:
                reader = PdfReader(tmp_path)
                partes = []
                for page in reader.pages:
                    texto_pagina = page.extract_text() or ""
                    partes.append(texto_pagina)
                conteudo = "\n\n".join(partes)
            finally:
                os.unlink(tmp_path)

        else:
            return jsonify({"error": "Tipo de arquivo não suportado. Use .txt ou .pdf."}), 400

    except Exception as e:
        return jsonify({"error": f"Erro ao processar arquivo: {e}"}), 500

    if not conteudo.strip():
        return jsonify({
            "filename": nome,
            "text": "",
            "preview": "",
            "summary": "Não foi possível extrair texto deste arquivo."
        })

    preview = conteudo[:1200]
    resumo = resumir_texto(conteudo)

    # 💾 Armazena o conteúdo completo na memória do usuário
    history = conversation_histories[client_id]
    history.clear()  # limpa histórico anterior
    history.append({
        "role": "system",
        "content": (
            f"O usuário enviou um arquivo chamado '{nome}'. "
            f"Abaixo está o conteúdo completo do arquivo:\n\n{conteudo}"
        )
    })

    print(f"[DEBUG] Conteúdo do arquivo armazenado na memória de {client_id} ({len(conteudo)} caracteres)")

    return jsonify({
        "filename": nome,
        "text": conteudo,
        "preview": preview,
        "summary": resumo,
    })

@app.route("/api/tts", methods=["POST"])
def tts():
    """
    Gera áudio MP3 com a voz da OpenAI a partir de um texto.
    """
    data = request.get_json() or {}
    texto = data.get("text", "")

    if not texto.strip():
        return jsonify({"error": "Texto vazio para TTS"}), 400

    try:
        mp3_path = gerar_audio_openai(texto)
        return send_file(mp3_path, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        print("Erro ao gerar TTS:", e)
        return jsonify({"error": f"Falha ao gerar áudio: {e}"}), 500

# -----------------------------------------------------------------------------
# NOVA ROTA: /api/stt (fala -> texto -> resposta)
# -----------------------------------------------------------------------------

@app.route("/api/stt", methods=["POST"])
def stt_conversa():
    """
    Recebe um áudio (voz do usuário), transcreve para texto,
    gera resposta no chat e devolve (texto do usuário + resposta).
    """
    if client is None:
        return jsonify({"error": "OpenAI não configurada"}), 500

    audio_file = request.files.get("audio")
    client_id = request.form.get("client_id", "anonimo")

    if not audio_file:
        return jsonify({"error": "Nenhum áudio enviado"}), 400

    try:
        # 1) Salva o áudio em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name

        try:
            # 2) Abre o arquivo como binário e manda para a OpenAI
            with open(tmp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    # se der erro de modelo, troque para "whisper-1"
                    model="gpt-4o-transcribe",
                    file=f,
                    response_format="text",
                )
        finally:
            # 3) Remove o arquivo temporário
            os.unlink(tmp_path)

        # Quando response_format="text", transcription costuma vir como string
        if isinstance(transcription, str):
            user_text = transcription.strip()
        else:
            user_text = str(transcription).strip()

    except Exception as e:
        print("Erro ao transcrever áudio:", e)
        return jsonify({"error": f"Falha ao transcrever áudio: {e}"}), 500

    # 4) Coloca a fala no histórico e gera resposta
    history = conversation_histories[client_id]
    history.append({"role": "user", "content": user_text})

    reply_text = gerar_resposta_agente(client_id)
    history.append({"role": "assistant", "content": reply_text})

    return jsonify({
        "user_text": user_text,
        "reply_text": reply_text,
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

