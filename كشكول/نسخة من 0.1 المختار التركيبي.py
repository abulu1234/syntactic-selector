# -*- coding: utf-8 -*-
import re, os, math
from collections import Counter, defaultdict
import pandas as pd

# ---------- أدوات المعالجة النصيّة ----------
class TextProcessor:
    DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    AR_LETTER_RE  = re.compile(r"[\u0600-\u06FF]+")
    @staticmethod
    def remove_diacritics(text): return TextProcessor.DIACRITICS_RE.sub("", text)
    @staticmethod
    def read_file(path):
        for enc in ("utf-8", "cp1256", "latin-1"):
            try: return open(path, encoding=enc).read()
            except UnicodeDecodeError: continue
        raise UnicodeDecodeError(f"تعذّر فك ترميز {path}")

class StopWords:
    def __init__(self, path):
        raw = TextProcessor.remove_diacritics(TextProcessor.read_file(path))
        self.stops = set(TextProcessor.AR_LETTER_RE.findall(raw))
    def is_stop(self, tok): return TextProcessor.remove_diacritics(tok) in self.stops

# ---------- استخراج التوكنات والمركبات ----------
class TokenExtractor:
    WORD_RE = re.compile(r"^[\u0600-\u06FF]+$")
    def __init__(self, folder, stopwords, match_mode):
        self.folder, self.sw, self.mode = folder, stopwords, match_mode
        if match_mode not in (1, 2, 3): raise ValueError("match_mode يجب أن يكون 1 أو 2 أو 3")
    def norm(self, w): return (TextProcessor.remove_diacritics(w)
                               if self.mode == 2 else w)

    def extract(self):
        uni = Counter(); bi = Counter()
        ctx = defaultdict(lambda: {'L':Counter(), 'R':Counter()})
        files = defaultdict(set)

        for fn in os.listdir(self.folder):
            if not fn.lower().endswith(".txt"): continue
            toks = TextProcessor.AR_LETTER_RE.findall(
                       TextProcessor.read_file(os.path.join(self.folder, fn)))
            toks = [self.norm(w) for w in toks
                    if self.WORD_RE.match(w) and not self.sw.is_stop(w)]
            for i, w in enumerate(toks): uni[w] += 1
            for i in range(len(toks)-1):
                w1, w2 = toks[i], toks[i+1]
                key = f"{w1} {w2}"
                bi[key] += 1; files[key].add(fn)
                if i > 0: ctx[key]['L'][toks[i-1]] += 1
                if i+2 < len(toks): ctx[key]['R'][toks[i+2]] += 1
        return uni, bi, files, ctx

# ---------- المقاييس الإحصائية ----------
class Stats:
    @staticmethod
    def pmi(f12, f1, f2, N): return math.log2((f12/N)/((f1/N)*(f2/N))) if f1 and f2 else 0
    @staticmethod
    def t_score(f12, f1, f2, N):
        exp = (f1 * f2) / N
        return (f12 - exp) / math.sqrt(f12) if f12 else 0
    @staticmethod
    def ll(f12, f1, f2, N):
        k11 = f12; k12 = f1 - f12; k21 = f2 - f12; k22 = N - (f1 + f2 - f12)
        def term(k, m): return k * math.log(k / m) if k and m else 0
        row1, row2 = k11 + k12, k21 + k22
        col1, col2 = k11 + k21, k12 + k22
        m11 = row1 * col1 / N; m12 = row1 * col2 / N
        m21 = row2 * col1 / N; m22 = row2 * col2 / N
        return 2 * sum(term(k, m) for k, m in ((k11, m11), (k12, m12), (k21, m21), (k22, m22)))
    @staticmethod
    def entropy(counter):
        tot = sum(counter.values())
        return -sum((c / tot) * math.log2(c / tot) for c in counter.values()) if tot else 0

# ---------- تصنيف الترشيح ----------
class Classify:
    @staticmethod
    def basic(pmi, t, ll):
        if pmi >= 7 and t >= 5 and ll >= 150: return "ترجيح قوي جدًّا"
        elif pmi >= 6 and t >= 4 and ll >= 100: return "ترجيح قوي"
        elif pmi >= 5 and t >= 3.5 and ll >= 70: return "ترجيح محتمل جدًّا"
        elif pmi >= 4 and t >= 3 and ll >= 50: return "ترجيح محتمل"
        elif pmi >= 3 and t >= 2.5 and ll >= 20: return "تركيب ضعيف"
        else: return "تركيب ضعيف جدًّا"

    @staticmethod
    def comprehensive(pmi, t, ll, spread, density, strong, pmi_ll, entropy):
        score = sum([
            pmi >= 6, t >= 4, ll >= 100,
            spread >= 3, density >= 2,
            strong == "نعم", pmi_ll < 0.15,
            entropy >= 1.0
        ])
        return ("ترجيح قوي جدًّا"     if score >= 8 else
                "ترجيح قوي"         if score >= 6 else
                "ترجيح محتمل جدًّا" if score >= 5 else
                "ترجيح محتمل"       if score >= 4 else
                "تركيب ضعيف"        if score >= 2 else
                "تركيب ضعيف جدًّا")

# ---------- التحليل النهائي ----------
class BigramAnalysis:
    MIN_ENTROPY = 1.0
    def __init__(self, folder, stop_file, mode, out_excel):
        self.sw = StopWords(stop_file)
        self.extr = TokenExtractor(folder, self.sw, mode)
        self.out = out_excel

    def run(self):
        uni, bi, files, ctx = self.extr.extract()
        N = sum(bi.values()); rows = []

        for comp, f12 in bi.items():
            parts = comp.split()
            if len(parts) != 2:
                continue   # تجاهل أي مفتاح ليس ثنائياً
            w1, w2 = parts
            f1, f2 = uni[w1], uni[w2]
            spread = len(files[comp])
            pmi = Stats.pmi(f12, f1, f2, N)
            t = Stats.t_score(f12, f1, f2, N)
            ll = Stats.ll(f12, f1, f2, N)
            density = round(f12 / spread, 3) if spread else 0
            pmi_ll = round(pmi / ll, 5) if ll else 0
            strong = "نعم" if f1 >= 100 and f2 >= 100 else "لا"
            entL = Stats.entropy(ctx[comp]['L'])
            entR = Stats.entropy(ctx[comp]['R'])
            entropy = round((entL + entR) / 2, 3)

            if pmi >= 3 and t >= 2 and ll >= 10 and entropy >= self.MIN_ENTROPY:
                rows.append({
                    "المركّب": comp, "التكرار": f12,
                    "الانتشار في الملفات": spread,
                    "الكثافة النسبية": density,
                    "التنوّع السياقي": entropy,
                    "شيوع جزئي المركب": strong,
                    "نسبة الربط النسبي": pmi_ll,
                    "PMI": round(pmi, 3), "t-score": round(t, 3),
                    "log-likelihood": round(ll, 3),
                    "ترجيح 3 معايير": Classify.basic(pmi, t, ll),
                    "ترجيح كل المعايير": Classify.comprehensive(
                        pmi, t, ll, spread, density, strong, pmi_ll, entropy)})

        if rows:
            df = pd.DataFrame(rows).sort_values(by="PMI", ascending=False)
            df.to_excel(self.out, index=False)
            print("تم إنشاء:", self.out)
        else:
            print("لا نتائج اجتازت الفلاتر")

# ---------- التشغيل ----------
if __name__ == "__main__":
    BigramAnalysis(folder="New folder",stop_file="stop_words.txt",
        mode = 2 ,out_excel="المركبات.xlsx").run()
#    1 → مطابقة تامة بالتشكيل (تمييز كامل بين الحركات).
#    2 → مطابقة جزئية بلا تشكيل (يُزال التشكيل كليًّا).
#    3 → تُنقل الكلمات كما كُتبت، دون تجريد ولا مطابقة.
