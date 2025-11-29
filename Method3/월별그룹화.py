import pandas as pd
import re
from glob import glob

# 1️ 여러 파일 경로 지정 (현재 폴더 기준)
file_paths = [
    "./논문일별요약/일별요약.xlsx",
]
# 2️ 여러 파일 불러와 병합 (현재는 1개지만 확장 가능)
df_list = []
for path in file_paths:
    temp = pd.read_excel(path)
    temp["source_file"] = path
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print(f" 총 {len(df)}개 행 불러옴")

# 3️ 날짜 및 요약 컬럼 이름 지정
date_col = "date"   # 예: "날짜", "date", "발행일" 등으로 수정 가능
summary_col = "요약"  # 예: "요약", "summary" 등으로 수정 가능

# 4️ 데이터 정리
df = df.dropna(subset=[date_col, summary_col])
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

def clean_text(x):
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

df[summary_col] = df[summary_col].apply(clean_text)

# 5️ 월별 그룹화
def merge_summaries(texts):
    texts = [t for t in texts if t]
    return "\n".join(f"- {t}" for t in texts)

df["month"] = df[date_col].dt.to_period("M").astype(str)

monthly_df = (
    df.groupby("month")
      .agg(
          n_days=(date_col, lambda s: s.dt.date.nunique()),
          n_items=(summary_col, "count"),
          merged_daily_summaries=(summary_col, merge_summaries)
      )
      .reset_index()
      .sort_values("month")
)

# 6️ 저장
monthly_df.to_csv("./논문일별요약/월별_요약_병합.csv", index=False, encoding="utf-8-sig")
monthly_df.to_excel("./논문일별요약/월별_요약_병합.xlsx", index=False)

print(" 월별 요약 병합 완료!")
print("→ ./논문일별요약/월별_요약_병합.csv")
print("→ ./논문일별요약/월별_요약_병합.xlsx")
