import pandas as pd
import re
from glob import glob
from pathlib import Path


# 1️ 여러 파일 경로 지정 (현재 폴더 기준)
file_paths = [
    "./논문일별요약/논문월별요약/월별요약.xlsx",
]


# ---------- 유틸 ----------
def detect_columns(df: pd.DataFrame):
    """
    월 컬럼과 텍스트(월별 병합 요약) 컬럼 자동 탐지
    - 월 컬럼 후보: 'month'('YYYY-MM'), '월', '기간'
    - 텍스트 컬럼 후보: 'merged_daily_summaries', 'merged_monthly_summaries', 'summary', '요약'
    """
    cols_lower = {c.lower(): c for c in df.columns}

    month_candidates = ["month", "월", "기간"]
    text_candidates  = ["merged_monthly_summaries", "merged_daily_summaries", "summary", "요약", "내용"]

    def pick(cands):
        # 정확 일치
        for key in cols_lower:
            if key in cands:
                return cols_lower[key]
        # 포함 일치
        for key in cols_lower:
            for token in cands:
                if token in key:
                    return cols_lower[key]
        return None

    month_col = pick(month_candidates)
    text_col  = pick(text_candidates)

    # 최후 보루: 월 컬럼이 없으면 날짜형으로 해석 가능한 컬럼을 찾아 month로 변환
    if month_col is None:
        best, ratio = None, 0.0
        for c in df.columns:
            ser = pd.to_datetime(df[c], errors="coerce")
            r = ser.notna().mean()
            if r > ratio and r >= 0.5:
                best, ratio = c, r
        if best is None:
            raise ValueError("월(또는 날짜) 컬럼을 찾지 못했습니다. (예: 'month' = 'YYYY-MM')")
        month_col = best

    # 텍스트 컬럼 없으면 가장 텍스트가 긴 object 컬럼 사용
    if text_col is None:
        text_lengths = {
            c: df[c].astype(str).str.len().mean()
            for c in df.columns if df[c].dtype == object
        }
        if not text_lengths:
            raise ValueError("요약 텍스트로 사용할 컬럼을 찾지 못했습니다.")
        text_col = max(text_lengths, key=text_lengths.get)

    return month_col, text_col

def clean_text(x):
    s = "" if pd.isna(x) else str(x).strip()
    return re.sub(r"\s+", " ", s)

def bullet_join(texts):
    texts = [t for t in texts if t]
    return "\n".join(f"- {t}" for t in texts)

# ---------- 로드 & 병합 ----------
dfs = []
for p in file_paths:
    df = pd.read_excel(p)
    df["__source_file__"] = p
    dfs.append(df)

if not dfs:
    raise RuntimeError("입력 파일이 없습니다.")

data = pd.concat(dfs, ignore_index=True)

# ---------- 컬럼 탐지 ----------
month_col, text_col = detect_columns(data)

# ---------- 전처리 ----------
work = data.copy()

# month_col이 'YYYY-MM' 문자열이면 뒤에 '-01' 붙여서 날짜로 파싱
if work[month_col].dtype == object:
    # 'YYYY-MM' 형태 감지 → 안전하게 '-01' 붙여 파싱
    work["month_dt"] = pd.to_datetime(work[month_col].astype(str).str.strip() + "-01", errors="coerce")
else:
    # 날짜형이면 월 시작일로 정규화
    work["month_dt"] = pd.to_datetime(work[month_col], errors="coerce")
    work["month_dt"] = work["month_dt"].dt.to_period("M").dt.to_timestamp()

work = work.dropna(subset=["month_dt"])

# 텍스트 정리
work[text_col] = work[text_col].map(clean_text)

# ---------- 반년(H1/H2) 키 생성 ----------
# H1: 1~6월, H2: 7~12월
half_idx = (work["month_dt"].dt.month - 1) // 6 + 1  # 1 또는 2
work["half"] = work["month_dt"].dt.year.astype(str) + "-H" + half_idx.astype(str)

# ---------- 반기 집계 ----------
semi = (
    work.groupby("half", dropna=False)
        .agg(
            n_months=("month_dt", "nunique"),
            n_items=(text_col, "count"),
            merged_monthly_summaries=(text_col, bullet_join)
        )
        .reset_index()
        .sort_values("half")
)

# (옵션) LLM 입력용 콤팩트 텍스트
semi["merged_compact"] = semi["merged_monthly_summaries"].str.replace(r"\s+", " ", regex=True)

# ---------- 저장 (입력 첫 파일과 같은 폴더) ----------
out_dir = Path(file_paths[0]).resolve().parent
csv_path  = out_dir / "반기_요약_병합.csv"
xlsx_path = out_dir / "반기_요약_병합.xlsx"

semi.to_csv(csv_path, index=False, encoding="utf-8-sig")
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
    semi.to_excel(w, index=False, sheet_name="half_year")
    ws = w.sheets["half_year"]
    for i, c in enumerate(semi.columns):
        max_len = min(80, max(10, semi[c].astype(str).map(len).max()))
        ws.set_column(i, i, max_len + 2)

print(" 반기(6개월) 그룹화 완료")
print(f"→ CSV : {csv_path}")
print(f"→ XLSX: {xlsx_path}")
