from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from markov import ShakespeareMarkov

# ---------------------------------------------------------
# アプリケーションの起動時設定 (Lifespan)
# サーバー起動時に一度だけモデルを読み込み、メモリに保持します。
# ---------------------------------------------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時: モデルをロード
    print("Initializing Shakespeare Model...")
    model = ShakespeareMarkov()
    model.load_and_process() # テキスト読み込み・前処理
    model.build_matrix()     # 行列Mの作成
    ml_models["shakespeare"] = model
    print("Model loaded successfully.")
    yield
    # 停止時: メモリ解放 (今回は特に処理なし)
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) 設定
# フロントエンド (React) からのアクセスを許可します。
# ---------------------------------------------------------
origins = [
    "http://localhost:5173",
    "https://sentence-simulator.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# データ構造の定義
# ---------------------------------------------------------
class GenerateRequest(BaseModel):
    start_word: str = "romeo"

# ---------------------------------------------------------
# APIエンドポイント
# ---------------------------------------------------------

@app.get("/")
def read_root():
    """サーバー稼働確認用"""
    return {"status": "ok", "message": "Shakespeare API is running"}

@app.post("/generate")
def generate_text(req: GenerateRequest):
    """
    指定された単語から文章を生成して返すAPI
    """
    model = ml_models.get("shakespeare")
    if not model:
        raise HTTPException(status_code=503, detail="Model is not ready")
    
    # 文章生成を実行
    # 入力を小文字にして渡す
    result = model.generate_sentence(req.start_word.lower())
    
    # 結果がエラーメッセージ ("Error: ...") の場合、そのまま返すかエラーにするか
    # ここではそのまま返します
    return {"sentence": result}