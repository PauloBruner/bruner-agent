from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import tempfile
from collections import defaultdict
from openai import OpenAI
from PyPDF2 import PdfReader

# ---------------------------------------------------------------------
# CONFIGURAÇÃO BÁSICA DO FLASK
# ---------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

# ---------------------------------------------------------------------
# CONFIG OPENAI
# ---------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Memória por usuário (cada navegador tem um client_id diferente)
conversation_histories = defaultdict(list)

# ---------------------------------------------------------------------
# FUNÇÕES DE IA
# ---------------------------------------------------------------------

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
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente virtual chamado Agente B. "
                        "Você fala em Português Brasileiro, de forma clara, objetiva e amigável. "
                        "O usuário que está conversando com você é simplesmente chamado de 'usuário'. "
                        "Responda sempre de forma educada, útil e focada na pergunta."
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
        print("Erro ao chamar OpenAI (chat):", repr(e))
        return f"Tive um problema ao falar com o modelo de IA.\nDetalhe técnico: {e}"


def resumir_texto(conteudo: str) -> str:
    """
    Resumo inteligente de um texto enviado em arquivo.
    """
    if client is None:
        return "Não consegui falar com a IA para resumir o texto (OpenAI não configurada)."

    trecho = conteudo[:8000]

    prompt = (
        "Resuma o texto completo abaixo, em português claro, "
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
                        "Você é um assistente que faz resumos completos, claros e organizados "
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
        print("Erro ao chamar OpenAI (resumo):", repr(e))
        return f"Não consegui resumir o texto por um erro técnico: {e}"

def gerar_audio_openai(texto: str) -> str:
    """
    Gera um arquivo MP3 com a voz da OpenAI e devolve o caminho temporário do arquivo.
    Versão usando streaming oficial (mais estável).
    """
    if client is None:
        raise RuntimeError("OpenAI não configurada")

    # cria arquivo temporário
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    try:
        # Versão recomendada pela OpenAI: with_streaming_response + stream_to_file
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

    except Exception as e:
        print("Erro ao gerar TTS:", repr(e))
        # apaga o arquivo se der erro
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        # repassa o erro para ser tratado na rota /api/tts
        raise

    return tmp_path

# ---------------------------------------------------------------------
# ROTAS FLASK
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    mensagem = data.get("message", "")
    client_id = data.get("client_id", "anonimo")

    if not mensagem.strip():
        return jsonify({"reply": "Pode repetir a pergunta? Não recebi nenhum texto."})

    history = conversation_histories[client_id]

    history.append({"role": "user", "content": mensagem})

    resposta = gerar_resposta_agente(client_id)

    history.append({"role": "assistant", "content": resposta})

    print(f"[DEBUG] /api/chat: client_id={client_id}, mensagens_no_historico={len(history)}")

    return jsonify({"reply": resposta})


@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Recebe um arquivo, lê o conteúdo (TXT ou PDF) e devolve prévia + resumo completo do PDF pela IA.
    Também armazena o texto completo na memória do client_id para permitir perguntas sobre ele.
    """
    arquivo = request.files.get("file")
    if not arquivo:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    client_id = request.form.get("client_id", "anonimo")

    nome = arquivo.filename or "arquivo"
    ext = os.path.splitext(nome)[1].lower()

    try:
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
        print("Erro ao processar arquivo:", repr(e))
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

    # NÃO limpamos o histórico inteiro; apenas acrescentamos a info do arquivo
    history = conversation_histories[client_id]
    history.append({
        "role": "system",
        "content": (
            f"O usuário enviou um arquivo chamado '{nome}'. "
            f"Abaixo está o conteúdo completo do arquivo:\n\n{conteudo}"
        )
    })

    print(f"[DEBUG] /api/upload: Conteúdo armazenado em {client_id} ({len(conteudo)} caracteres)")

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
        print("Erro ao gerar TTS /api/tts:", repr(e))
        return jsonify({"error": f"Falha ao gerar áudio: {e}"}), 500

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
        # 1) Salvar áudio em arquivo temporário
        suffix = ".webm"
        nome_arquivo = audio_file.filename or ""
        if nome_arquivo.lower().endswith((".mp4", ".m4a", ".aac")):
            suffix = ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name

        try:
            # 2) Abrir arquivo e mandar para Whisper
            with open(tmp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
        finally:
            os.unlink(tmp_path)

        if isinstance(transcription, str):
            user_text = transcription.strip()
        else:
            user_text = str(transcription).strip()

    except Exception as e:
        print("Erro ao transcrever áudio /api/stt:", repr(e))
        return jsonify({"error": f"Falha ao transcrever áudio: {e}"}), 500

    # Adiciona fala no histórico
    history = conversation_histories[client_id]
    history.append({"role": "user", "content": user_text})

    reply_text = gerar_resposta_agente(client_id)
    history.append({"role": "assistant", "content": reply_text})

    print(f"[DEBUG] /api/stt: client_id={client_id}, len_history={len(history)}")

    return jsonify({
        "user_text": user_text,
        "reply_text": reply_text,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
