# -*- coding: utf-8 -*-
import re, os, math, json
from collections import Counter, defaultdict
import pandas as pd
from datetime import datetime

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

# ---------- استخراج التوكنات والمركبات مع السياقات ----------
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
        contexts = defaultdict(list)  # حفظ السياقات النصية
        file_stats = defaultdict(lambda: {'total_words': 0, 'bigrams': 0})

        for fn in os.listdir(self.folder):
            if not fn.lower().endswith(".txt"): continue
            file_path = os.path.join(self.folder, fn)
            text = TextProcessor.read_file(file_path)
            toks = TextProcessor.AR_LETTER_RE.findall(text)
            toks = [self.norm(w) for w in toks
                    if self.WORD_RE.match(w) and not self.sw.is_stop(w)]
            
            file_stats[fn]['total_words'] = len(toks)
            
            for i, w in enumerate(toks): uni[w] += 1
            for i in range(len(toks)-1):
                w1, w2 = toks[i], toks[i+1]
                key = f"{w1} {w2}"
                bi[key] += 1; files[key].add(fn)
                file_stats[fn]['bigrams'] += 1
                
                # حفظ السياق النصي
                start = max(0, i-2)
                end = min(len(toks), i+4)
                context = " ".join(toks[start:end])
                contexts[key].append({
                    'file': fn,
                    'context': context,
                    'position': i,
                    'highlight_start': len(" ".join(toks[start:i])) + (1 if start < i else 0),
                    'highlight_end': len(" ".join(toks[start:i+2]))
                })
                
                if i > 0: ctx[key]['L'][toks[i-1]] += 1
                if i+2 < len(toks): ctx[key]['R'][toks[i+2]] += 1
                
        return uni, bi, files, ctx, contexts, file_stats

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
    @staticmethod
    def z_score(value, mean, std): return (value - mean) / std if std else 0

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

# ---------- التحليل النهائي المحسن ----------
class BigramAnalysis:
    MIN_ENTROPY = 1.0
    def __init__(self, folder, stop_file, mode, out_excel, out_json):
        self.sw = StopWords(stop_file)
        self.extr = TokenExtractor(folder, self.sw, mode)
        self.out_excel = out_excel
        self.out_json = out_json

    def run(self):
        uni, bi, files, ctx, contexts, file_stats = self.extr.extract()
        N = sum(bi.values()); rows = []
        
        # حساب الإحصائيات الإجمالية
        pmi_values = []; t_values = []; ll_values = []
        
        for comp, f12 in bi.items():
            parts = comp.split()
            if len(parts) != 2: continue
            w1, w2 = parts
            f1, f2 = uni[w1], uni[w2]
            spread = len(files[comp])
            pmi = Stats.pmi(f12, f1, f2, N)
            t = Stats.t_score(f12, f1, f2, N)
            ll = Stats.ll(f12, f1, f2, N)
            
            pmi_values.append(pmi)
            t_values.append(t)
            ll_values.append(ll)
            
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

        # حساب Z-scores
        pmi_mean = sum(pmi_values) / len(pmi_values) if pmi_values else 0
        t_mean = sum(t_values) / len(t_values) if t_values else 0
        ll_mean = sum(ll_values) / len(ll_values) if ll_values else 0
        
        pmi_std = math.sqrt(sum((x - pmi_mean) ** 2 for x in pmi_values) / len(pmi_values)) if pmi_values else 1
        t_std = math.sqrt(sum((x - t_mean) ** 2 for x in t_values) / len(t_values)) if t_values else 1
        ll_std = math.sqrt(sum((x - ll_mean) ** 2 for x in ll_values) / len(ll_values)) if ll_values else 1

        # إضافة Z-scores للنتائج
        for row in rows:
            pmi_val = row["PMI"]
            t_val = row["t-score"]
            ll_val = row["log-likelihood"]
            row["PMI_Z"] = round(Stats.z_score(pmi_val, pmi_mean, pmi_std), 3)
            row["T_Z"] = round(Stats.z_score(t_val, t_mean, t_std), 3)
            row["LL_Z"] = round(Stats.z_score(ll_val, ll_mean, ll_std), 3)

        if rows:
            df = pd.DataFrame(rows).sort_values(by="PMI", ascending=False)
            df.to_excel(self.out_excel, index=False)
            print("تم إنشاء:", self.out_excel)
            
            # تصدير JSON للتفاعل
            self.export_json(rows, contexts, file_stats, uni, bi)
            print("تم إنشاء:", self.out_json)
        else:
            print("لا نتائج اجتازت الفلاتر")

    def export_json(self, rows, contexts, file_stats, uni, bi):
        # تحضير البيانات للتصدير
        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_bigrams": len(bi),
                "total_unigrams": len(uni),
                "total_files": len(file_stats)
            },
            "bigrams": [],
            "file_stats": file_stats,
            "contexts": contexts,
            "statistics": {
                "unigrams": dict(uni.most_common(100)),
                "bigrams": dict(bi.most_common(100))
            }
        }
        
        for row in rows:
            comp = row["المركّب"]
            bigram_data = {
                "compound": comp,
                "frequency": row["التكرار"],
                "spread": row["الانتشار في الملفات"],
                "density": row["الكثافة النسبية"],
                "entropy": row["التنوّع السياقي"],
                "strong_parts": row["شيوع جزئي المركب"],
                "pmi_ratio": row["نسبة الربط النسبي"],
                "pmi": row["PMI"],
                "t_score": row["t-score"],
                "log_likelihood": row["log-likelihood"],
                "pmi_z": row["PMI_Z"],
                "t_z": row["T_Z"],
                "ll_z": row["LL_Z"],
                "basic_classification": row["ترجيح 3 معايير"],
                "comprehensive_classification": row["ترجيح كل المعايير"],
                "contexts": contexts.get(comp, [])
            }
            export_data["bigrams"].append(bigram_data)
        
        with open(self.out_json, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

# ---------- التشغيل ----------
if __name__ == "__main__":
    BigramAnalysis(
        folder="المدونة",
        stop_file="stop_words.txt",
        mode=2,
        out_excel="المركبات_المحسنة.xlsx",
        out_json="البيانات_التفاعلية.json"
    ).run()
