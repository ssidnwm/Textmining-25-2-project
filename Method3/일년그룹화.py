import re
import pandas as pd
from pathlib import Path

# 1) 입력 파일 경로(여러 파일도 가능)
file_paths = [
    "./논문일별요약/논문월별요약/논문반년요약/반기요약.xlsx",   # ← 네 경로에 맞게 수정
    # "./다른경로/반기_요약_병합.xlsx",
]

# ---------- 유틸 ----------
def detect_columns(df: pd.DataFrame):
    """
    연도 산출에 필요한 반기/월/날짜 컬럼과 텍스트 컬럼을 자동 탐지
    - half 후보: 'half', '반기', 'half_year', 'halfyear', '기간'
    - month 후보: 'month'('YYYY-MM'), '월'
    - date 후보: 날짜형으로 파싱 가능한 컬럼
    - text 후보: 'merged_monthly_summaries', 'merged_half_summaries', 'summary', '요약', '내용'
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(cands):
        # 정확 일치
        for key in cols_lower:
            if key in cands:
                return cols_lower[key]
        # 부분 일치
        for key in cols_lower:
            for token in cands:
                if token in key:
                    return cols_lower[key]
        return None

    half_col  = pick(["half", "반기", "half_year", "halfyear", "기간"])
    month_col = pick(["month", "월"])
    text_col  = pick(["merged_half_summaries", "merged_monthly_summaries",
                      "merged_daily_summaries", "summary", "요약", "내용"])

    # 텍스트 컬럼 없으면 가장 텍스트가 긴 object 컬럼 사용
    if text_col is None:
        text_lengths = {
            c: df[c].astype(str).str.len().mean()
            for c in df.columns if df[c].dtype == object
        }
        if not text_lengths:
            raise ValueError("요약 텍스트로 사용할 컬럼을 찾지 못했습니다.")
        text_col = max(text_lengths, key=text_lengths.get)

    # 날짜형 컬럼 후보(연도 유도용) 확보
    date_like_cols = []
    for c in df.columns:
        ser = pd.to_datetime(df[c], errors="coerce")
        if ser.notna().mean() >= 0.5:
            date_like_cols.append(c)

    return half_col, month_col, date_like_cols, text_col

def extract_year_from_half(val: str):
    """'2025-H1' 같은 문자열에서 연도 추출"""
    if not isinstance(val, str):
        return None
    m = re.search(r"(\d{4})", val)
    return m.group(1) if m else None

def bullet_join(texts):
    texts = [str(t).strip() for t in texts if str(t).strip()]
    return "\n".join(f"- {t}" for t in texts)

# ---------- 로드 & 결합 ----------
dfs = []
for p in file_paths:
    df = pd.read_excel(p)
    df["__source_file__"] = p
    dfs.append(df)

if not dfs:
    raise RuntimeError("입력 파일이 없습니다.")

data = pd.concat(dfs, ignore_index=True)

# ---------- 컬럼 탐지 ----------
half_col, month_col, date_like_cols, text_col = detect_columns(data)

work = data.copy()

# ---------- 연도 컬럼(year) 생성 ----------
year = None

if half_col is not None:
    # 1) 반기 문자열에서 연도 뽑기
    work["__year_from_half__"] = work[half_col].apply(extract_year_from_half)
    if work["__year_from_half__"].notna().any():
        year = work["__year_from_half__"]

if year is None and month_col is not None:
    # 2) month('YYYY-MM')에서 연도 뽑기
    #    문자열이면 안전하게 '-01' 붙여 파싱
    if work[month_col].dtype == object:
        month_dt = pd.to_datetime(work[month_col].astype(str).str.strip() + "-01", errors="coerce")
    else:
        month_dt = pd.to_datetime(work[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work["__year_from_month__"] = month_dt.dt.year.astype("Int64")
    if work["__year_from_month__"].notna().any():
        year = work["__year_from_month__"].astype("Int64").astype("string")

if year is None and date_like_cols:
    # 3) 날짜형으로 파싱 가능한 컬럼 중 하나에서 연도 뽑기
    best_col, best_ratio = None, 0.0
    for c in date_like_cols:
        ser = pd.to_datetime(work[c], errors="coerce")
        r = ser.notna().mean()
        if r > best_ratio:
            best_col, best_ratio = c, r
    ser = pd.to_datetime(work[best_col], errors="coerce")
    work["__year_from_date__"] = ser.dt.year.astype("Int64").astype("string")
    year = work["__year_from_date__"]

if year is None:
    raise ValueError("연도를 유도할 수 없습니다. (반기/월/날짜 컬럼을 확인하세요)")

work["year"] = year

# ---------- 집계(연도별 병합) ----------
yearly = (
    work.groupby("year", dropna=False)
        .agg(
            n_rows=("__source_file__", "count"),   # 원본 행 수
            n_halves=(half_col, "nunique") if half_col in work.columns else ("year", "count"),
            merged_half_summaries=(text_col, bullet_join)
        )
        .reset_index()
        .sort_values("year")
)

# (옵션) LLM 입력용 콤팩트 텍스트
yearly["merged_compact"] = yearly["merged_half_summaries"].str.replace(r"\s+", " ", regex=True)

# ---------- 저장 (입력 첫 파일과 같은 폴더) ----------
out_dir = Path(file_paths[0]).resolve().parent
csv_path  = out_dir / "연도_요약_병합.csv"
xlsx_path = out_dir / "연도_요약_병합.xlsx"

yearly.to_csv(csv_path, index=False, encoding="utf-8-sig")

# xlsxwriter 미설치 환경 대비: openpyxl로 fallback
try:
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
        yearly.to_excel(w, index=False, sheet_name="yearly")
        ws = w.sheets["yearly"]
        for i, c in enumerate(yearly.columns):
            max_len = min(80, max(10, yearly[c].astype(str).map(len).max()))
            ws.set_column(i, i, max_len + 2)
except ModuleNotFoundError:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        yearly.to_excel(w, index=False, sheet_name="yearly")

print(" 연도별 병합 완료")
print(f"→ CSV : {csv_path}")
print(f"→ XLSX: {xlsx_path}")
