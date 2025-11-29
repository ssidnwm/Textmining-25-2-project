import os
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from datetime import datetime

# 1. 설정
# 파일 경로와 파일 이름
file_path = "C:/Users/82109/Desktop/벼리/7학기/텍스트마이닝/GPT기반방법론/논문일별요약/월별_요약_병합.csv" # 경로 변경하기

# 결과물을 저장할 폴더 설정 (원본 파일 위치에 '샘플 이벤트' 폴더 생성)
folder_path = os.path.dirname(file_path)
save_dir = os.path.join(folder_path, "논문월별요약")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path_csv = os.path.join(save_dir, f"월별요약.csv") # 파일명 변경하기
save_path_xlsx = save_path_csv.replace(".csv", ".xlsx")

# CSV 파일 불러오기
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig').reset_index(drop=True)
    print("데이터를 성공적으로 불러왔습니다.")
    
    # 불러온 데이터의 일부 확인 (처음 5줄)
    print(df.head())

except FileNotFoundError:
    print(f"오류: 지정된 경로에 파일이 없습니다. 경로를 다시 확인해주세요: {file_path}")
    exit() # 파일이 없으면 스크립트 종료

# --- 중요: 여기에 OpenAI API 키를 입력하세요 ---
os.environ["OPENAI_API_KEY"] = "api key"

chat_model = ChatOpenAI(model="gpt-4o-mini")

# 2. 프롬프트 정의
prompt_header = """
You are given a collection of paper titles and abstracts that were published within the same month.
Please analyze these papers collectively to identify the major research trends for that month.

**Output Requirements**
1. Identify the dominant research themes or focus areas that repeatedly appear across the papers.
2. Highlight any emerging or novel topics that gained attention during the month.
3. Summarize in 4–6 sentences the overall research direction and methodological trends for that month.
4. Be strictly objective and factual; do not speculate beyond what is supported by the abstracts.
5. Emphasize recurring keywords, research methods, and application domains (e.g., motion analysis, performance optimization, injury prevention, player monitoring).
6. Output format:
   - **Month:** (e.g., 2025-10)
   - **Main Research Themes:** (concise paragraph summary)
   - **Emerging Topics:** (list or short paragraph)
   - **Representative Keywords:** (comma-separated list)
"""

# 3. 저장 디렉토리 생성
os.makedirs(save_dir, exist_ok=True)
print(f"결과물은 다음 위치에 저장됩니다: {save_dir}")

# 4. 전체 파일 처리 (한 줄씩 LLM 호출)
print(f"파일 처리 중: {os.path.basename(file_path)}")
try:
    if not all(col in df.columns for col in ['month', 'n_items', 'merged_daily_summaries']):
        print(f"'{os.path.basename(file_path)}' 파일에 필수 컬럼(merged_daily_summaries, n_items, month)이 없어 건너뜁니다.")
        exit()
    
    valid_rows = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        print(f"처리 중... ({idx+1}/{total_rows})")

        article_text = row.get('merged_daily_summaries', 'N/A') 
        article_date = row.get('month', 'N/A')
        
        # 각 행에 대한 개별 프롬프트 생성
        news_block = f"merged_summaries: {article_text}\nDate: {article_date}\n"
        full_prompt = prompt_header + "\n" + news_block

        try:
            response = chat_model.invoke([HumanMessage(content=full_prompt)])
            event = response.content.strip()
            
                
        except Exception as e:
            print(f"LLM 호출 중 오류 발생: {e}. 다음 행으로 넘어갑니다.")
            continue

        
        new_row = {
                "month": row.get("month", ""),
                "merged_daily_summaries": row.get("merged_daily_summaries", ""),
                "요약": event
            }
        valid_rows.append(new_row)

    if valid_rows:
        df_to_save = pd.DataFrame(valid_rows)
        df_to_save.to_csv(
            save_path_csv,
            index=False,
            encoding='utf-8-sig'
        )
        print(f"{len(valid_rows)}개의 유효한 데이터를 CSV에 저장했습니다.")
    else:
        print("유효한 데이터가 없어 CSV 파일에 저장할 내용이 없습니다.")

except Exception as e:
    print(f"파일 '{os.path.basename(file_path)}' 처리 중 오류 발생: {e}")

# 5. 최종 CSV 파일을 Excel로 저장
if os.path.exists(save_path_csv) and os.path.getsize(save_path_csv) > 0:
    print(f"\nCSV 파일을 Excel로 변환 중: {os.path.basename(save_path_xlsx)}...")
    try:
        df_final = pd.read_csv(save_path_csv, on_bad_lines='skip')
        df_final.to_excel(save_path_xlsx, index=False, engine='openpyxl')
        print("모든 작업 완료! CSV와 Excel 파일이 모두 저장되었습니다.")
    except Exception as e:
        print(f"Excel 저장 중 오류 발생: {e}")
else:
    print("\n저장할 유효한 데이터가 없습니다. 출력 파일이 생성되지 않았습니다.")
    
# 총 소요시간만 출력
elapsed = time.perf_counter() - t0
h, rem = divmod(int(elapsed), 3600)
m, s = divmod(rem, 60)
ms = int((elapsed - int(elapsed)) * 1000)
print(f"\n총 소요시간: {h:02d}:{m:02d}:{s:02d}.{ms:03d}")