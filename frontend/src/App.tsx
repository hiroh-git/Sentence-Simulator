import { useState } from 'react'
import './App.css'

function App() {
  const [startWord, setStartWord] = useState<string>("romeo")
  const [generatedText, setGeneratedText] = useState<string>("")
  const [loading, setLoading] = useState<boolean>(false)

  const handleGenerate = async () => {
    setLoading(true);
    setGeneratedText(""); // リセット

    try {
      // Pythonバックエンド (FastAPI) にリクエストを送信
      // ※ Vercelにデプロイする際はここを環境変数に切り替えます
      const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ start_word: startWord }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setGeneratedText(data.sentence);

    } catch (error) {
      console.error("Error:", error);
      setGeneratedText("エラーが発生しました。バックエンドが起動しているか確認してください。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>シェイクスピア風 文章生成器</h1>
      <p>MCMCモデルによる文章生成</p>
      
      <div className="card">
        <div className="input-group">
          <label>最初の単語 (例: romeo, i, thou): </label>
          <input 
            type="text" 
            value={startWord} 
            onChange={(e) => setStartWord(e.target.value)}
            placeholder="Type a word..."
          />
        </div>
        
        <button onClick={handleGenerate} disabled={loading}>
          {loading ? "生成中..." : "Generate Sentence"}
        </button>
      </div>

      {generatedText && (
        <div className="result-card">
          <h3>Result:</h3>
          <p>"{generatedText}"</p>
        </div>
      )}
    </div>
  )
}

export default App