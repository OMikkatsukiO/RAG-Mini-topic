# RAG 文件問答小專題(Gemini 版,部署於 Render)

用 RAG + Gemini API 做的文件問答小專題。上傳文件、選擇切分策略後就能提問;問答會被記住,也能把完整問答紀錄存成 JSON 或 TXT 檔案,推播到自己的 Telegram 或 Discord。

## 使用方式

1. **填入你自己的 Gemini API Key**(到 [Google AI Studio](https://aistudio.google.com) 申請)。這個服務不提供共用金鑰,每個使用者用自己的額度,金鑰只在當次瀏覽器分頁有效,不會被保存。
2. **上傳文件**(PDF / DOCX / TXT / MD),按「上傳」。
3. **選擇切分策略**(五種可選,見下方說明),按「套用切分策略」。想換策略比較效果,不用重新上傳,直接改選項再按一次「套用切分策略」就好。
4. 在右側輸入問題開始問答。
5.(選用)想把問答紀錄傳到 Telegram 或 Discord:在上方「問答紀錄要傳送到哪裡?」選平台、填對應的 Token/Webhook,再按「把完整問答紀錄存成檔案並傳送」。

## 架構

- **RAG**:文件上傳後先清理文字(去除控制字元、多餘的 Markdown 符號),依選擇的策略切塊,存進本地向量庫(Chroma),問答時檢索相關片段當上下文。
- **五種文件切分策略**:
  - 固定長度:純按字數切,不管語意邊界。
  - 語義切分:按句子邊界組合,不會切斷句子中間。
  - 遞歸切分:優先照段落、句子等自然邊界切,過小的相鄰片段會合併。
  - 滑動視窗:固定長度 + 區塊間保留重疊。
  - 混合策略:遞歸切分的基礎上再補重疊。
- **模型**:Gemini API(`gemini-2.5-flash`),不自架模型,免費方案就能跑,不用等模型下載。
- **Embedding**:也是呼叫 Gemini API(`gemini-embedding-001`),不在本地載入任何 embedding 模型。這點是刻意的:本地跑 embedding(例如 sentence-transformers)會連帶載入 PyTorch,記憶體需求動輒 500MB 以上,免費方案(通常 512MB)撐不住,啟動時就會被系統判定 OOM 砍掉。
- **問答記憶**:每輪問答存進向量庫(未來類似問題可以被檢索到)+ 一份 `data/qa_history.jsonl`(完整紀錄)。這是檢索式記憶,不會拿這些資料重新訓練模型。
- **匯出推播**:把累積的完整問答紀錄打包成 JSON 或 TXT,透過 Telegram Bot API(`send_document`)或 Discord Webhook 推送出去。

## 取得 Telegram / Discord 憑證

- **Telegram**:跟 `@BotFather` 對話建立 bot,取得 Bot Token;先傳一則訊息給你的 bot,再用 `@userinfobot` 查詢你的 Chat ID。
- **Discord**:頻道設定 → 整合(Integrations)→ Webhook → 建立 Webhook → 複製網址。

## 部署到 Render

1. 把 `app.py`、`requirements.txt`、這份 `README.md` 推到一個 GitHub repo。
2. 到 [Render](https://render.com) 建立新的 **Web Service**,連接這個 repo(不需要信用卡)。
3. 設定這兩個欄位:
   - **Build Command**:`pip install -r requirements.txt`
   - **Start Command**:`python app.py`
4. 硬體方案選 **Free**。這個應用不需要額外設定任何環境變數(API Key、Telegram/Discord 憑證都是使用者在網頁介面自己輸入,不是伺服器端的密鑰)。
5. 按 **Create Web Service**,建置通常幾分鐘內完成(套件都有預編譯 wheel),完成後會拿到一個 `xxx.onrender.com` 的網址。

## 這個服務的限制(免費方案)

- **閒置會休眠**:免費方案的 Web Service 在 15 分鐘沒有請求後會休眠,下一次有人連進來要等 30–60 秒喚醒,這段時間第一個訪客會覺得沒反應,是正常現象。
- **每月 750 小時額度**:一般小專題用量不太可能超過,但仍值得留意。
- **重啟/休眠會清空記憶**:服務休眠喚醒或重新部署後,`data/` 目錄(向量庫、問答紀錄)會被清空,文件需要重新上傳。問答紀錄可以隨時匯出到 Telegram/Discord 當備份,降低這個限制的影響。
- **API Key/Token 不會被保存**:重新整理頁面要重新輸入。

## 檔案結構

```
.
├── app.py             # Gradio 應用(這個服務的進入點)
├── requirements.txt
├── README.md
└── data/               # 執行時自動建立,存放向量庫與問答紀錄
```
