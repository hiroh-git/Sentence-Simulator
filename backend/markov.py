import re
import numpy as np
from collections import Counter

class ShakespeareMarkov:
    def __init__(self, filepath="shakespeare.txt", mlag=4, top_k=1000):
        self.filepath = filepath
        self.mlag = mlag
        self.top_k = top_k
        self.vocab = []      # トップk単語のリスト
        self.word_to_id = {} # 単語 -> ID
        self.id_to_word = {} # ID -> 単語
        self.M = None        # 遷移行列
        self.tokens = []     # 全テキストのID列

    def load_and_process(self):
        """テキストを読み込み、前処理を行う"""
        print("Loading and processing text...")
        
        # 1. 読み込み (ヘッダー/フッター削除は簡易的に行数指定ではなく、内容でトリミングも可能ですが、
        # ここではRに合わせて行範囲を指定して読み込みます)
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # R: skip=83, nlines=196043-83
        # Pythonのリストスライスは0始まりなので調整
        start_line = 83
        end_line = 196043
        text_lines = lines[start_line:end_line]
        raw_text = " ".join([l.strip() for l in text_lines])

        # 2. Stage directions ([...]) の削除
        # Rではループを使っていますが、Pythonの正規表現なら一発です
        # 入れ子の括弧には対応していませんが、Practicalの要件ならこれで十分です
        text = re.sub(r'\[.*?\]', '', raw_text)

        # 3. 前処理 (大文字削除、数字削除、記号分離)
        # まず単語に分割して処理しやすくします
        # 句読点の前後にスペースを入れて分割しやすくする (Rの split_punct 相当)
        puncs = [",", ".", ";", "!", ":", "?"]
        for p in puncs:
            text = text.replace(p, f" {p} ")
        
        words = text.split()
        cleaned_words = []
        
        for w in words:
            # "_" と "-" を削除
            w = re.sub(r'[_-]', '', w)
            if not w: continue

            # Rロジック: 全大文字(名前)と数字を削除。ただし "I" と "A" は残す
            if w.isupper() and w not in ["I", "A"]:
                continue
            if w.isdigit():
                continue
                
            cleaned_words.append(w.lower()) # 小文字化

        # 4. 語彙の構築 (TOKENIZATION)
        # 頻出上位 k 個を取得 (Rの 'b')
        counts = Counter(cleaned_words)
        most_common = counts.most_common(self.top_k)
        
        self.vocab = [word for word, count in most_common]
        
        # IDマッピング作成 (1始まりではなく0始まりにします)
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_word = {i: w for i, w in enumerate(self.vocab)}

        # 全テキストをIDに変換 (Rの 'M1')
        # 語彙にない単語は -1 (Rの NA 相当) とする
        self.tokens = [self.word_to_id.get(w, -1) for w in cleaned_words]
        
        print(f"Preprocessing done. Total tokens: {len(self.tokens)}, Vocab size: {len(self.vocab)}")

    def build_matrix(self):
        """行列Mを作成する"""
        print("Building matrix M...")
        
        tokens_np = np.array(self.tokens)
        n = len(tokens_np)
        
        # mlag+1 個の列を持つ行列を作るための準備
        # numpyの sliding_window_view を使うと高速に作成できます
        # shape: (n - mlag, mlag + 1)
        from numpy.lib.stride_tricks import sliding_window_view
        window_size = self.mlag + 1
        M_view = sliding_window_view(tokens_np, window_size)
        
        # 学習データとして有効な行だけを抽出
        
        # 最後の列 (ターゲット) が有効か
        valid_target = M_view[:, -1] != -1
        # 文脈の最後が有効か
        valid_context = M_view[:, -2] != -1
        
        self.M = M_view[valid_target & valid_context].copy()
        
        print(f"Matrix M built. Shape: {self.M.shape}")

    def next_word(self, key_words):
        """
        次の単語を予測する
        key_words: ['romeo', 'is'] などの単語リスト
        """
        # 単語をIDに変換（未知語は無視して短くする）
        key_ids = [self.word_to_id.get(k) for k in key_words if k in self.word_to_id]
        
        # キーが長すぎる場合は切り詰める
        if len(key_ids) > self.mlag:
            key_ids = key_ids[-self.mlag:]
        
        next_word_candidates = []
        weights = []
        
        # R: for (i in length(key):1)
        # 最長一致から順に短くしてマッチを探す
        for i in range(len(key_ids), 0, -1):
            sub_key = key_ids[-i:] # 末尾からi個
            
            # マッチさせる列の範囲 (Rの mc:mlag に相当)
            # Pythonのインデックスは 0 ~ mlag (全 mlag+1列)
            # ターゲットは col index [mlag]
            # マッチ対象は col index [mlag-i : mlag]
            
            # コンテキスト部分の行列
            # M[:, mlag-i : mlag] が sub_key と一致する行を探す
            context_cols = self.M[:, self.mlag-i : self.mlag]
            
            # 行ごとに一致判定 (axis=1 で全要素一致を確認)
            matches = np.all(context_cols == sub_key, axis=1)
            
            if np.any(matches):
                # マッチした行のターゲット列 (最後の列) を取得
                matched_targets = self.M[matches, self.mlag]
                
                next_word_candidates.extend(matched_targets)
                
                # 重み計算 (R: w[length(key)-i+1] / length(next_word))
                # 簡易化のため重み w=1 と仮定すると、 1 / 件数
                w_val = 1.0 / len(matched_targets)
                weights.extend([w_val] * len(matched_targets))

        # 候補がない場合、ランダムに選ぶ (Fallback)
        if not next_word_candidates:
            # 頻度上位リストからランダム (NA(-1)以外)
            valid_tokens = [t for t in self.tokens if t != -1]
            return self.id_to_word[np.random.choice(valid_tokens)]
        
        # 重み付きランダムサンプリング
        # 確率の合計を1に正規化
        weights = np.array(weights)
        probs = weights / weights.sum()
        
        chosen_id = np.random.choice(next_word_candidates, p=probs)
        return self.id_to_word[chosen_id]

    def generate_sentence(self, start_word="romeo"):
        """指定した単語からピリオドまで生成する"""
        if start_word not in self.word_to_id:
            return "Error: Start word not in vocabulary."
            
        generated = [start_word]
        current_token = start_word
        
        # ピリオドが出るまでループ (無限ループ防止でmax 50単語制限)
        count = 0
        while current_token != "." and count < 50:
            next_w = self.next_word(generated)
            generated.append(next_w)
            current_token = next_w
            count += 1
            
        # 文章の整形 (句読点の前のスペース削除)
        text = " ".join(generated)
        puncs = [",", ".", ";", "!", ":", "?"]
        for p in puncs:
            text = text.replace(f" {p}", p)
            
        return text

# テスト実行用（このファイルを直接実行したときのみ動く）
if __name__ == "__main__":
    model = ShakespeareMarkov()
    model.load_and_process()
    model.build_matrix()
    print("--- Generated Text ---")
    print(model.generate_sentence("romeo"))