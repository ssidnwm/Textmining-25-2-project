import pandas as pd
import re
from glob import glob

# 1️ 여러 파일 경로 지정 (현재 폴더 기준)
file_paths = [
    "./논문초록요약/1618논문별요약.xlsx",
    "./논문초록요약/1921논문별요약.xlsx",
    "./논문초록요약/2224논문별요약.xlsx"
]

# 2️ 모든 파일 불러와 하나의 DataFrame으로 병합
df_list = []
for path in file_paths:
    temp = pd.read_excel(path)
    temp["source_file"] = path  # 원본 파일명 기록
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print(f" 총 {len(df)}개의 논문 데이터가 병합되었습니다.")

# 3️ 컬럼명 확인
print("컬럼 목록:", df.columns.tolist())

# 4️ 날짜 및 요약 컬럼 지정
date_col = "date"    # 예: date, 발행일 등
summary_col = "요약"  # 예: 요약, abstract_summary 등

# 5️ 결측치 제거 및 날짜 정제
df = df.dropna(subset=[date_col, summary_col])
df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
df = df.dropna(subset=[date_col])

# 6️ 텍스트 정리
def clean_text(x):
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

df[summary_col] = df[summary_col].apply(clean_text)

# 7️ 날짜별 병합
def merge_summaries(texts):
    texts = [t for t in texts if t]
    return "\n".join(f"- {t}" for t in texts)

merged_df = (
    df.groupby(date_col)
      .agg(
          n_papers=(summary_col, "count"),
          merged_summaries=(summary_col, merge_summaries)
      )
      .reset_index()
      .sort_values(date_col)
)

# 8️ 결과 확인
print(merged_df.head())

# 9️ CSV / Excel 저장
merged_df.to_csv("날짜별_요약_통합.csv", index=False, encoding='utf-8-sig')
merged_df.to_excel("날짜별_요약_통합.xlsx", index=False)

print(" 모든 파일을 병합하고 날짜별 요약문을 통합했습니다.")
print("→ 결과 파일: 날짜별_요약_통합.csv / 날짜별_요약_통합.xlsx")
