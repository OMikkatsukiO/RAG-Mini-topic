import os
import io
import re
import json
import time
import uuid
import unicodedata
from typing import List, Optional

import chromadb
from pypdf import PdfReader
from docx import Document as DocxDocument
from google import genai
from google.genai import types
import gradio as gr
import discord
from telegram import Bot


# ============ 設定 ============
DATA_DIR = "/kaggle/working/data" if os.path.exists("/kaggle/working") else "/content/data" if os.path.exists("/content") else "./data"
CHROMA_DIR = f"{DATA_DIR}/chroma_db"
QA_LOG_PATH = f"{DATA_DIR}/qa_history.jsonl"
os.makedirs(DATA_DIR, exist_ok=True)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768  # gemini-embedding-001 預設 3072 維,縮小到 768 省儲存空間,RAG 用途足夠

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K_DOCS = 4
TOP_K_QA_HISTORY = 2
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = (
    "你是一個根據使用者上傳文件回答問題的助理。"
    "優先根據提供的文件內容與過去問答紀錄回答;"
    "如果內容中找不到答案,要誠實說不知道,不要編造。"
)


# ============ 文字清理:去除亂碼、控制字元、多餘的 Markdown 符號 ============
def clean_text(text: str) -> str:
    """清掉常見的亂碼/雜訊:控制字元、多餘空白、Markdown 符號,盡量還原成乾淨的純文字。
    刻意不處理單星號斜體(*文字*),因為『5 * 3』這種數學算式會被誤判、把內容吃掉,
    風險比留著沒清乾淨的符號更高。"""
    if not text:
        return text

    text = unicodedata.normalize("NFKC", text)  # 統一連字、組合字元,例如 ﬁ ﬂ 還原成 fi fl
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))  # 移除不可見控制字元

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # **粗體** → 粗體
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)     # # 標題 → 標題
    text = re.sub(r"`([^`]+)`", r"\1", text)                        # `code` → code
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)      # 行首項目符號統一成 •

    text = re.sub(r"\n{3,}", "\n\n", text)   # 三個以上換行收斂成兩個
    text = re.sub(r"[ \t]{2,}", " ", text)    # 同一行內多個空白收斂成一個

    return text.strip()


# ============ RAG:文件讀取、切塊、向量庫 ============
def load_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        raw = "\n".join(p.text for p in doc.paragraphs)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    else:
        raise ValueError(f"目前不支援的檔案格式:{ext}")
    return clean_text(raw)


def split_fixed_length(text, chunk_size, overlap=0):
    """1. 固定長度切分:純粹按字數切,不管語意邊界,速度快、實作簡單。"""
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def split_into_sentences(text):
    """把文字切成句子清單,句尾標點保留在句子尾端。"""
    pieces = re.split(r'(?<=[。!?;])|(?<=[.!?])(?=\s)', text)
    return [p.strip() for p in pieces if p.strip()]


def split_by_sentence(text, chunk_size):
    """2. 語義切分(簡化版):先切成完整句子,再把句子組合到接近 chunk_size,
    確保每個 chunk 都在句子邊界結束,不會切斷句子中間。"""
    text = text.strip()
    if not text:
        return []
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    chunks, current = [], ""
    for sent in sentences:
        if current and len(current) + len(sent) > chunk_size:
            chunks.append(current.strip())
            current = sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _merge_small_pieces(pieces, chunk_size):
    """把切出來但太小的相鄰片段合併,避免『文件裡有很多短段落』這種情況
    被切成一堆瑣碎的小 chunk,不利於之後的檢索品質。"""
    if not pieces:
        return []
    merged, current = [], pieces[0]
    for p in pieces[1:]:
        if len(current) + len(p) + 2 <= chunk_size:
            current = current + "\n\n" + p
        else:
            merged.append(current)
            current = p
    merged.append(current)
    return merged


def split_recursive(text, chunk_size, overlap=0, separators=None):
    """3. 遞歸切分:照『段落 -> 句子 -> 固定長度』優先順序,
    只有超過限制的區塊才會往下一層細分;切完後再把過小的相鄰片段合併一次。"""
    text = text.strip()
    if not text:
        return []
    if separators is None:
        separators = ["\n\n", "\n"]

    def _split(chunk, seps):
        chunk = chunk.strip()
        if not chunk:
            return []
        if len(chunk) <= chunk_size:
            return [chunk]
        if not seps:
            sentence_chunks = split_by_sentence(chunk, chunk_size)
            result = []
            for sc in sentence_chunks:
                if len(sc) <= chunk_size:
                    result.append(sc)
                else:
                    result.extend(split_fixed_length(sc, chunk_size, overlap=0))
            return result
        sep, rest = seps[0], seps[1:]
        pieces = [p for p in chunk.split(sep) if p.strip()]
        if len(pieces) <= 1:
            return _split(chunk, rest)
        result = []
        for p in pieces:
            result.extend(_split(p, rest))
        return result

    raw_pieces = _split(text, separators)
    return _merge_small_pieces(raw_pieces, chunk_size)


def split_sliding_window(text, chunk_size, overlap):
    """4. 滑動視窗切分:固定長度切分,但保留重疊區域,避免重要語境被切在邊界上。"""
    return split_fixed_length(text, chunk_size, overlap)


def split_hybrid(text, chunk_size, overlap):
    """5. 混合策略:先用遞歸切分抓自然邊界,區塊之間再補上重疊,
    兼顧語意完整跟上下文連續。"""
    chunks = split_recursive(text, chunk_size, overlap=0)
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
        overlapped.append((prev_tail + " " + chunks[i]).strip())
    return overlapped


# 統一介面:key 是介面上顯示的策略名稱,value 是對應的切分函式
CHUNK_STRATEGIES = {
    "固定長度": lambda text, chunk_size, overlap: split_fixed_length(text, chunk_size, overlap=0),
    "語義切分": lambda text, chunk_size, overlap: split_by_sentence(text, chunk_size),
    "遞歸切分": lambda text, chunk_size, overlap: split_recursive(text, chunk_size, overlap=0),
    "滑動視窗": lambda text, chunk_size, overlap: split_sliding_window(text, chunk_size, overlap),
    "混合策略": lambda text, chunk_size, overlap: split_hybrid(text, chunk_size, overlap),
}
DEFAULT_STRATEGY = "固定長度"


def embed_texts(api_key: str, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """呼叫 Gemini 的 embedding API 把文字轉成向量。刻意不在本地載入任何 embedding 模型
    (例如 sentence-transformers)——那會連帶拉入 PyTorch,在 512MB 記憶體的免費方案上會被 OOM 砍掉。
    task_type 分開 RETRIEVAL_DOCUMENT(存文件用)跟 RETRIEVAL_QUERY(查詢用),檢索品質比兩邊都用同一種更好。
    """
    if not texts:
        return []
    if not api_key:
        raise RuntimeError("請先在上方輸入你的 Gemini API Key。")
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION, task_type=task_type),
    )
    return [e.values for e in response.embeddings]


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        # 故意不指定 embedding_function——向量一律由呼叫端先用 embed_texts()(呼叫 Gemini API)
        # 算好再手動傳進來,chromadb 這裡只負責儲存跟比對,不會在本地載入任何模型。
        self.documents = self.client.get_or_create_collection("documents")
        self.qa_history = self.client.get_or_create_collection("qa_history")

    def add_text(self, api_key: str, text: str, source_name: str, strategy: str = DEFAULT_STRATEGY) -> int:
        """把『已經讀取好、清理過』的文字,依指定策略切塊後存進向量庫。
        跟讀檔案的步驟分開,讓上傳文件、選切分策略可以是兩個獨立動作。"""
        split_fn = CHUNK_STRATEGIES.get(strategy, CHUNK_STRATEGIES[DEFAULT_STRATEGY])
        chunks = split_fn(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return 0

        # 同一個來源重新套用策略時,先清掉舊片段,避免新舊策略的內容同時留著互相干擾。
        existing = self.documents.get(where={"source": source_name})
        if existing["ids"]:
            self.documents.delete(ids=existing["ids"])

        embeddings = embed_texts(api_key, chunks, task_type="RETRIEVAL_DOCUMENT")
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "chunk_index": i, "strategy": strategy} for i in range(len(chunks))]
        self.documents.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        return len(chunks)

    def list_sources(self) -> List[str]:
        result = self.documents.get()
        sources = {m.get("source") for m in result.get("metadatas", []) if m}
        return sorted(sources)

    def search_documents(self, api_key: str, query: str, top_k: int = TOP_K_DOCS) -> List[str]:
        if self.documents.count() == 0:
            return []
        query_embedding = embed_texts(api_key, [query], task_type="RETRIEVAL_QUERY")[0]
        result = self.documents.query(query_embeddings=[query_embedding], n_results=min(top_k, self.documents.count()))
        return result.get("documents", [[]])[0]


vector_store = VectorStore()


# ============ 模型層(Gemini API,API key 由使用者在介面輸入,不快取)============
def generate_answer(api_key: str, question: str, context_chunks, history, qa_history_chunks=None):
    if not api_key:
        raise RuntimeError("請先在上方輸入你的 Gemini API Key。")

    client = genai.Client(api_key=api_key)
    history = history[-MAX_HISTORY_MESSAGES:]

    context_text = "\n\n".join(context_chunks) if context_chunks else "(沒有檢索到相關文件片段)"
    history_text = "\n\n".join(qa_history_chunks) if qa_history_chunks else ""

    user_content = f"參考文件片段:\n{context_text}\n"
    if history_text:
        user_content += f"\n過去相關問答:\n{history_text}\n"
    user_content += f"\n使用者問題:{question}"

    contents = history + [{"role": "user", "parts": [{"text": user_content}]}]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3),
    )
    answer = response.text

    updated_history = contents + [{"role": "model", "parts": [{"text": answer}]}]
    return answer, updated_history


# ============ 問答記憶(檢索式記憶 + 完整歷史紀錄的匯出)============
class QAMemory:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def save(self, api_key: str, question, answer, source="web"):
        question = clean_text(question)
        answer = clean_text(answer)
        qa_id = str(uuid.uuid4())
        record = {"id": qa_id, "question": question, "answer": answer, "source": source, "timestamp": time.time()}
        doc_text = f"問題:{question}\n答案:{answer}"
        embedding = embed_texts(api_key, [doc_text], task_type="RETRIEVAL_DOCUMENT")[0]
        self.vector_store.qa_history.add(
            documents=[doc_text], embeddings=[embedding], ids=[qa_id],
            metadatas=[{"source": source, "timestamp": record["timestamp"]}],
        )
        with open(QA_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def search_similar(self, api_key: str, question, top_k=TOP_K_QA_HISTORY):
        collection = self.vector_store.qa_history
        if collection.count() == 0:
            return []
        query_embedding = embed_texts(api_key, [question], task_type="RETRIEVAL_QUERY")[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))
        return result.get("documents", [[]])[0]

    def load_all(self) -> List[dict]:
        """讀取『全部』問答紀錄,不像先前的 load_recent 只挑最近幾筆——匯出檔案要用這個。"""
        try:
            with open(QA_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def export_as_json_bytes(self) -> bytes:
        records = self.load_all()
        return json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")

    def export_as_txt_bytes(self) -> bytes:
        records = self.load_all()
        if not records:
            text = "目前還沒有問答紀錄。"
        else:
            lines = []
            for r in records:
                lines.append(f"[{r.get('source', '未知來源')}] Q: {r['question']}\nA: {r['answer']}\n")
            text = "\n".join(lines)
        return text.encode("utf-8")


qa_memory = QAMemory(vector_store)


# ============ 把完整問答紀錄存成檔案,推播到 Telegram / Discord ============
async def send_history_file(platform, file_format, telegram_token, telegram_chat_id, discord_webhook_url):
    if platform == "不傳送":
        return "目前選擇「不傳送」,先在上面選 Telegram 或 Discord。"

    records = qa_memory.load_all()
    if not records:
        return "目前還沒有任何問答紀錄可以匯出。"

    if file_format == "JSON":
        file_bytes = qa_memory.export_as_json_bytes()
        filename = "qa_history.json"
    else:
        file_bytes = qa_memory.export_as_txt_bytes()
        filename = "qa_history.txt"

    try:
        if platform == "Telegram":
            if not telegram_token or not telegram_chat_id:
                return "請先填寫 Telegram 的 Bot Token 跟 Chat ID。"
            bot = Bot(token=telegram_token)
            await bot.send_document(
                chat_id=telegram_chat_id,
                document=io.BytesIO(file_bytes),
                filename=filename,
            )
            return f"已把 {filename}({len(records)} 筆紀錄)傳送到 Telegram。"

        if platform == "Discord":
            if not discord_webhook_url:
                return "請先填寫 Discord 的 Webhook URL。"
            webhook = discord.SyncWebhook.from_url(discord_webhook_url)
            webhook.send(file=discord.File(io.BytesIO(file_bytes), filename=filename))
            return f"已把 {filename}({len(records)} 筆紀錄)傳送到 Discord。"
    except Exception as e:
        return f"傳送失敗:{e}"

    return "不支援的平台選項。"


# ============ Gradio 介面 ============
def stage_documents(files, staged_docs):
    """第一步(上傳):只讀取、清理文件內容,暫存起來,不切塊、不存進向量庫。
    切分策略要等第二步使用者選好之後才會用到。"""
    if not files:
        return staged_docs, "沒有選擇檔案。", gr.update(visible=False)

    staged_docs = dict(staged_docs or {})
    names = []
    for file in files:
        path = file.name if hasattr(file, "name") else file
        name = os.path.basename(path)
        staged_docs[name] = load_text_from_file(path)
        names.append(name)

    status = f"已上傳 {len(names)} 個檔案({', '.join(names)}),請在下面選擇切分策略,再按「套用切分策略」。"
    return staged_docs, status, gr.update(visible=True)


def process_staged_documents(staged_docs, strategy, api_key):
    """第二步(套用策略):使用者選好切分策略後,才真正把暫存的文字切塊、存進向量庫。
    這一步需要呼叫 Gemini 的 embedding API,所以也需要 api_key。"""
    if not staged_docs:
        return "還沒有上傳文件,請先在上面上傳。"
    if not api_key:
        return "請先在上方輸入你的 Gemini API Key,切塊需要呼叫 embedding API。"
    try:
        total_chunks, names = 0, []
        for name, text in staged_docs.items():
            total_chunks += vector_store.add_text(api_key, text, source_name=name, strategy=strategy)
            names.append(name)
        return f"已用「{strategy}」切分 {len(names)} 個檔案({', '.join(names)}),共存入 {total_chunks} 個片段。"
    except Exception as e:
        return f"處理失敗:{e}"


def list_sources_fn():
    sources = vector_store.list_sources()
    return "已收錄的文件:\n" + "\n".join(f"- {s}" for s in sources) if sources else "目前向量庫裡還沒有文件。"


def chat(message, chat_history, session_history, api_key):
    if not message.strip():
        return "", chat_history, session_history
    try:
        doc_chunks = vector_store.search_documents(api_key, message)
        qa_chunks = qa_memory.search_similar(api_key, message)
        answer, session_history = generate_answer(api_key, message, doc_chunks, session_history, qa_chunks)
        qa_memory.save(api_key, message, answer, source="web")
    except Exception as e:
        answer = f"發生錯誤,請稍後再試:{e}"
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", chat_history, session_history


def toggle_platform_fields(platform):
    return gr.update(visible=(platform == "Telegram")), gr.update(visible=(platform == "Discord"))


with gr.Blocks(title="RAG 文件問答小專題(Gemini 版)") as demo:
    gr.Markdown(
        "## RAG 文件問答小專題(Gemini 版)\n"
        "上傳文件後直接提問;問答會被記住,不用重新上傳文件。\n"
        "下面先填你自己的 Gemini API Key 才能開始問答。"
    )

    with gr.Accordion("設定(API Key / 傳送目的地)", open=True):
        api_key_input = gr.Textbox(label="Gemini API Key", type="password", placeholder="到 Google AI Studio 申請")
        platform_choice = gr.Radio(["不傳送", "Telegram", "Discord"], value="不傳送", label="問答紀錄要傳送到哪裡?")
        with gr.Group(visible=False) as telegram_group:
            telegram_token_input = gr.Textbox(label="Telegram Bot Token", type="password", placeholder="向 @BotFather 申請")
            telegram_chatid_input = gr.Textbox(label="Telegram Chat ID", placeholder="先跟你的 bot 對話,再用 @userinfobot 查詢")
        with gr.Group(visible=False) as discord_group:
            discord_webhook_input = gr.Textbox(label="Discord Webhook URL", type="password", placeholder="頻道設定 > 整合 > Webhook")

    platform_choice.change(toggle_platform_fields, inputs=platform_choice, outputs=[telegram_group, discord_group])

    with gr.Row():
        with gr.Column(scale=1):
            staged_docs_state = gr.State({})  # 暫存「已上傳但還沒切塊」的文件內容:{檔名: 清理過的文字}

            gr.Markdown("**步驟 1:上傳文件**")
            file_input = gr.File(file_count="multiple", label="上傳文件(PDF / DOCX / TXT / MD)")
            upload_btn = gr.Button("上傳")
            upload_status = gr.Textbox(label="上傳狀態", interactive=False)

            with gr.Group(visible=False) as strategy_group:
                gr.Markdown("**步驟 2:選擇切分策略並套用**")
                strategy_choice = gr.Radio(
                    list(CHUNK_STRATEGIES.keys()),
                    value=DEFAULT_STRATEGY,
                    label="文件切分策略",
                )
                process_btn = gr.Button("套用切分策略")
                process_status = gr.Textbox(label="處理狀態", interactive=False)
                gr.Markdown("*想試不同策略,改選項後直接再按一次「套用切分策略」就好,不用重新上傳。*")

            list_btn = gr.Button("查看已收錄的文件")
            source_list = gr.Textbox(label="文件清單", interactive=False)
        with gr.Column(scale=2):
            try:
                chatbot = gr.Chatbot(label="問答", height=450, type="messages")
            except TypeError:
                chatbot = gr.Chatbot(label="問答", height=450)
            msg = gr.Textbox(label="輸入問題", placeholder="針對上傳的文件提問…")
            session_state = gr.State([])

            with gr.Row():
                file_format_choice = gr.Radio(["JSON", "TXT"], value="JSON", label="匯出格式", scale=1)
                send_history_btn = gr.Button("把完整問答紀錄存成檔案並傳送", scale=2)
            send_status = gr.Textbox(label="傳送狀態", interactive=False)

    upload_btn.click(
        stage_documents,
        inputs=[file_input, staged_docs_state],
        outputs=[staged_docs_state, upload_status, strategy_group],
    )
    process_btn.click(
        process_staged_documents, inputs=[staged_docs_state, strategy_choice, api_key_input], outputs=process_status
    ).then(
        list_sources_fn, outputs=source_list
    )
    list_btn.click(list_sources_fn, outputs=source_list)
    msg.submit(chat, inputs=[msg, chatbot, session_state, api_key_input], outputs=[msg, chatbot, session_state])
    send_history_btn.click(
        send_history_file,
        inputs=[platform_choice, file_format_choice, telegram_token_input, telegram_chatid_input, discord_webhook_input],
        outputs=send_status,
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
