import os
import json
import re  # [로컬 최적화] 정규표현식 지원 추가
import time
import pandas as pd
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_from_directory
import google.generativeai as genai
import requests

app = Flask(__name__)

# [로컬 최적화] 데이터 영구 저장 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
CURRICULUM_CACHE = os.path.join(DATA_DIR, 'curriculum_cache.json')

# [로컬 최적화] LLM 응답에서 순수 JSON만 안전하게 추출하는 유틸리티
def extract_json_safe(text):
    if not text:
        return {}
    try:
        # 1. 마크다운 백틱 제거 시도
        json_match = re.search(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # 2. 텍스트 내 가장 바깥쪽 { } 또는 [ ] 추출 시도
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
            
        # 3. 최후의 수단: 전체 텍스트 파싱
        return json.loads(text.strip())
    except Exception as e:
        print(f"JSON Parsing Error: {e}\nOriginal Text: {text}")
        # 파싱 실패 시 원본 텍스트를 담은 객체 반환 (프론트엔드 대응용)
        return {"result": text.strip()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    task_type = data.get('taskType')
    payload = data.get('payload')
    engine = data.get('engine', 'local')
    api_key = data.get('apiKey', '')
    ollama_model = data.get('ollamaModel', 'gemma3:4b') # 검증된 gemma3:4b로 변경

    persona = "당신은 한국의 15년 차 수석 교사이자 백워드 설계(UbD) 전문가입니다. "
    prompt = build_prompt(task_type, payload, persona)
    schema = get_schema(task_type)

    if engine == 'cloud':
        try:
            print(f"DEBUG: Attempting Cloud AI (Gemini) with task={task_type}")
            if not api_key:
                return jsonify({"error": "Gemini API 키가 설정되지 않았습니다."}), 400
            
            genai.configure(api_key=api_key)
            
            # [지능형 폴백] 단일 최적화 모델 적용 (무한 로딩/타임아웃 방지)
            models_to_try = [
                'gemini-flash-latest'
            ]
            
            last_error = ""
            for model_name in models_to_try:
                try:
                    print(f"DEBUG: Trying Gemini Model -> {model_name}")
                    model = genai.GenerativeModel(model_name)
                    full_prompt = f"{prompt}\nPayload Data: {json.dumps(payload, ensure_ascii=False)}"
                    
                    # 우선 스키마 포함 시도
                    try:
                        response = model.generate_content(
                            full_prompt,
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                response_schema=schema
                            )
                        )
                    except Exception as schema_e:
                        print(f"DEBUG: Schema request failed, trying simple request for {model_name}: {str(schema_e)}")
                        # 스키마 없이 일반 텍스트 모드로 재시도 (백엔드에서 JSON 추출할 것임)
                        response = model.generate_content(full_prompt)

                    print(f"DEBUG: Gemini ({model_name}) successful")
                    return jsonify(extract_json_safe(response.text))

                except Exception as inner_e:
                    print(f"DEBUG: Gemini ({model_name}) failed: {str(inner_e)}")
                    last_error = str(inner_e)
                    continue

            return jsonify({"error": f"Gemini API 호출 최종 실패: {last_error}"}), 500
        except Exception as e:
            print(f"DEBUG: Gemini Config Error -> {str(e)}")
            return jsonify({"error": f"Gemini 설정 오류: {str(e)}"}), 500
            
    else: # local (Ollama)
        try:
            print(f"DEBUG: Attempting Local AI (Ollama) with model={ollama_model}, task={task_type}")
            url = "http://localhost:11434/api/generate"
            body = {
                "model": ollama_model,
                "prompt": f"{prompt}\nPayload: {json.dumps(payload, ensure_ascii=False)}",
                "stream": False
            }
            res = requests.post(url, json=body, timeout=60)
            res.raise_for_status()
            res_data = res.json()
            print(f"DEBUG: Ollama response received")
            parsed_result = extract_json_safe(res_data.get('response', ''))
            return jsonify(parsed_result)
        except requests.exceptions.ConnectionError:
            print(f"DEBUG: Ollama Connection Refused")
            return jsonify({"error": "Ollama 서버가 실행 중인지 확인하세요 (localhost:11434)."}), 500
        except Exception as e:
            print(f"DEBUG: Ollama Error -> {str(e)}")
            return jsonify({"error": f"로컬 AI 요청 실패: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 전송되지 않았습니다."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400
    
    filename = file.filename
    ext = filename.split('.')[-1].lower()
    
    try:
        data_list = []
        if ext == 'csv':
            # [로컬 최적화] 다양한 인코딩 및 구분자 대응
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
            df = None
            raw_content = file.read()
            
            for enc in encodings:
                try:
                    text = raw_content.decode(enc)
                    lines = text.splitlines()
                    
                    # 1. 헤더 행 탐색 (Pre-scan)
                    header_keywords = ['성취기준', '내용', '코드', '교과', '과목', 'Subject', 'Content', 'Code']
                    best_line_idx = 0
                    max_matches = 0
                    
                    for i, line in enumerate(lines[:15]): # 상단 15줄 검사
                        matches = sum(1 for kw in header_keywords if kw in line)
                        if matches > max_matches:
                            max_matches = matches
                            best_line_idx = i
                    
                    # 2. 헤더 행부터 다시 읽기
                    import io
                    valid_text = "\n".join(lines[best_line_idx:])
                    # sep=None, engine='python'은 구분자를 자동 감지함
                    df = pd.read_csv(io.StringIO(valid_text), sep=None, engine='python', on_bad_lines='warn')
                    
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    print(f"Encoding {enc} trial failed: {e}")
                    continue
            
            if df is None or df.empty:
                return jsonify({"error": "CSV 파일분석 실패. 인코딩이나 형식을 확인하세요."}), 400
            data_list = process_df(df)
            
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
            data_list = process_df(df)
            
        elif ext == 'pdf':
            with pdfplumber.open(file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() or ""
                    full_text += "\n"
                data_list = process_text(full_text)
                
        elif ext == 'txt':
            text = file.read().decode('utf-8', errors='ignore')
            data_list = process_text(text)
        else:
            return jsonify({"error": f"지원하지 않는 파일 형식입니다: {ext}"}), 400
            
        # [NEW] 영구 저장 (캐싱)
        with open(CURRICULUM_CACHE, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
            
        return jsonify({"filename": filename, "data": data_list})
    except Exception as e:
        import traceback
        traceback.print_exc() # 상세 서버 에러 출력
        return jsonify({"error": f"파일 분석 오류: {str(e)}"}), 500

def process_df(df):
    # [로컬 최적화] 다이나믹 헤더 탐지 보강
    header_keywords = ['성취기준', '내용', '코드', '교과', '과목', 'Subject', 'Content', 'Code']
    
    best_row_idx = -1
    max_keywords = 0
    
    # 0. 컬럼명 자체가 유효한지 먼저 확인
    cols = [str(c) for c in df.columns]
    matches = sum(1 for col in cols if any(kw in col for kw in header_keywords))
    if matches >= 2: 
        max_keywords = matches
        best_row_idx = -1 
        
    # 1. 상단 10개 행 탐색
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str).tolist()
        matches = sum(1 for val in row_vals if any(kw in val for kw in header_keywords))
        if matches > max_keywords:
            max_keywords = matches
            best_row_idx = i
            
    if best_row_idx != -1:
        new_header = df.iloc[best_row_idx].tolist()
        df = df.iloc[best_row_idx + 1:].copy()
        df.columns = new_header
        
    cols = df.columns.tolist()
    actual_map = {}
    
    # [로컬 최적화] 매핑 우선순위: 완전 일치 > 부분 일치
    header_map = {
        'subject': ['교과', '과목', 'Subject', '분류'],
        'domain': ['영역', '분야', 'Domain', '주제', '내용 체계'],
        'code': ['성취기준코드', '성취기준 코드', '코드', '번호', 'Code', 'ID'],
        'content': ['성취기준', '내용', 'Content', '상세', '설명']
    }
    
    for key, aliases in header_map.items():
        # 1순위: 완전 일치
        for col in cols:
            c_clean = str(col).strip()
            if c_clean in aliases:
                actual_map[key] = col
                break
        if key in actual_map: continue
        
        # 2순위: 부분 일치
        for col in cols:
            c_clean = str(col).replace(" ", "")
            if key == 'content' and '코드' in c_clean: continue
            if any(alias in c_clean for alias in aliases):
                actual_map[key] = col
                break

    # [로컬 최적화] 폴백: 가장 긴 텍스트 컬럼
    if 'content' not in actual_map:
        max_avg = 0
        sample = df.head(10).fillna("")
        for col in cols:
            avg_len = sample[col].astype(str).str.len().mean()
            if avg_len > max_avg:
                max_avg = avg_len
                actual_map['content'] = col

    print(f"DEBUG: Mapped Columns -> {actual_map}")

    results = []
    df = df.dropna(how='all')
    
    for idx, row in df.iterrows():
        c_col = actual_map.get('content')
        if not c_col: continue
        
        val = row.get(c_col)
        content = str(val).strip() if pd.notnull(val) else ""
        if len(content) < 5 or content in cols: continue
        
        results.append({
            "id": idx + 1,
            "subject": str(row.get(actual_map.get('subject'), '미분류')).strip(),
            "domain": str(row.get(actual_map.get('domain'), '기본')).strip(),
            "code": str(row.get(actual_map.get('code'), f"SC-{idx+1}")).strip(),
            "content": content
        })
    return results

def process_text(text):
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10] # 너무 짧은 줄 제외
    results = []
    for i, line in enumerate(lines):
        results.append({
            "id": i + 1,
            "subject": "자동분석",
            "domain": "텍스트",
            "code": f"L{i+1}",
            "content": line
        })
    return results

def build_prompt(task_type, payload, persona):
    stds = payload.get('stds', '')
    topic = payload.get('topic', '')
    context = f"\n[선택된 성취기준]: {stds}\n[프로젝트 주제]: {topic}"
    
    # [UX 개선] 더욱 명확한 JSON 유도 프롬프트
    json_instr = "\n반드시 다른 설명 없이 순수 JSON 형식으로만 응답해라."

    prompts = {
        "SHORT_GEN": f"{persona}주어진 텍스트를 기반으로 학생의 학습 도달점이나 본질적 질문을 짧게 제시해 줘.\n{payload.get('text','')}",
        "RECOMMEND_EU": f"{persona}다음 맥락을 분석하여, 이번 단원을 관통하는 철학적인 '영속적 이해'를 1문장으로 도출해 줘.{json_instr}\n{context}",
        "RECOMMEND_EQ": f"{persona}위 성취기준과 주제에 어울리는 본질적 질문 3가지를 JSON 배열로 만들어 줘.{json_instr}\n{context}",
        "RECOMMEND_KSA": f"{persona}지식과 기능을 추출하여 JSON(knowledge, skill 배열)으로 반환해.{json_instr}\n{context}",
        "RECOMMEND_GRASPS": f"{persona}다음 성취기준과 주제를 바탕으로 백워드 설계의 GRASPS 수행과제 시나리오를 작성해 줘. JSON 형식(goal, role, audience, situation, product 필드)으로 반환해.{json_instr}\n{context}",
        "COMPLEX_PLANNING": f"{persona}다음 맥락을 바탕으로 체계적인 수업 흐름을 설계해 주십시오.\n\n[핵심 지침]\n- 사용자가 요청한 총 차시는 {payload.get('lessonCount', '6~8')}차시입니다. \n- 반드시 1차시부터 {payload.get('lessonCount', '6~8')}차시까지의 내용만 생성하세요.\n- 시수가 부족하거나 넘치지 않게 전체 교육과정의 호흡을 조절하여 배분하세요.\n- 백워드 설계의 WHERETO 원칙을 각 차시에 적절히 녹여내야 합니다.\n\n[상세 기술 지침]\n1. 각 차시는 단계(phase), 제목(title), 상세설명(desc), WHERETO 요소 배열(whereto), 연관 루브릭 항목(rubricLinks)을 포함해야 합니다.\n2. WHERETO 요소는 ['W','H','E','R','E2','T','O'] 중 해당 차시의 성격과 맞는 기호를 선택하십시오.\n3. rubricLinks는 제공된 평가 루브릭의 항목명들을 참고하여 연관된 것을 나열하십시오.\n4. 반드시 JSON 배열 형식으로만 응답해 주십시오.{json_instr}\n\n[주제]: {payload.get('topic','')}\n[성취기준]: {payload.get('stds','')}\n[영속적 이해]: {payload.get('eu','')}\n[GRASPS 시나리오]: {json.dumps(payload.get('grasps',{}), ensure_ascii=False)}\n[평가 루브릭]: {json.dumps(payload.get('rubric',[]), ensure_ascii=False)}",
        "RUBRIC": f"{persona}다음 GRASPS 시나리오와 맥락을 분석하여, 학생의 성취도를 정밀하게 판별할 수 있는 루브릭(평가 기준)을 3~4개 항목으로 제안해 줘.\n\n[작성 지침]\n1. 반드시 JSON 배열로만 반환해.\n2. 배열의 각 객체는 '항목명', '매우잘함', '잘함', '보통', '노력요함' 5개 필드를 반드시 포함해야 함.\n3. '항목명'은 '내용의 타당성', '창의적 표현' 등 구체적인 평가 요소를 사용해.\n4. '매우잘함', '잘함', '보통', '노력요함' 각 필드의 값은 해당 수준의 학생 수행 특징을 1~2문장으로 구체적으로 기술해.\n\n출력 예시:\n[\n  {{\"항목명\": \"내용의 타당성\", \"매우잘함\": \"학생이...\", \"잘함\": \"...\", \"보통\": \"...\", \"노력요함\": \"...\"}},\n  {{\"항목명\": \"창의적 표현\", \"매우잘함\": \"...\", ...}}\n]\n\n시나리오 데이터: {json.dumps(payload.get('grasps',{}), ensure_ascii=False)}\n{context}",
        "SUMMARIZE_TITLE": f"{persona}다음 프로젝트 주제를 분석하여, 문서의 제목으로 사용하기에 적합한 10자 내외의 짧고 매력적인 제목을 단 하나만 생성해 줘.\n\n[작성 지침]\n- 반드시 10자 내외(최대 15자)로 작성할 것.\n- 명사형으로 종결할 것 (예: ~의 탐구, ~ 프로젝트).\n- 불필요한 수식어는 제거하고 핵심 키워드 중심으로 구성할 것.\n- 반드시 JSON 형식(`{{\"title\": \"요약된 제목\"}}`)으로만 응답할 것.\n\n[프로젝트 주제]: {payload.get('topic','')}"
    }
    return prompts.get(task_type, f"{persona}요청을 처리해 줘.")

# [NEW] 저장된 성취기준 로드
@app.route('/api/load_curriculum', methods=['GET'])
def load_curriculum():
    if os.path.exists(CURRICULUM_CACHE):
        try:
            with open(CURRICULUM_CACHE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({"success": True, "data": data})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "message": "No cache found"}), 404

# [NEW] 캐시 초기화
@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    if os.path.exists(CURRICULUM_CACHE):
        os.remove(CURRICULUM_CACHE)
        return jsonify({"success": True})
    return jsonify({"success": True, "message": "Already empty"})

def get_schema(task_type):
    if task_type == 'RECOMMEND_EU':
        return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}
    if task_type == 'RECOMMEND_EQ':
        return {"type": "array", "items": {"type": "string"}}
    if task_type == 'RECOMMEND_KSA':
        return {"type": "object", "properties": {"knowledge": {"type": "array", "items": {"type": "string"}}, "skill": {"type": "array", "items": {"type": "string"}}}, "required": ["knowledge", "skill"]}
    if task_type == 'COMPLEX_PLANNING':
        return {
            "type": "array", 
            "items": {
                "type": "object", 
                "properties": {
                    "id": {"type": "integer"}, 
                    "phase": {"type": "string"}, 
                    "title": {"type": "string"}, 
                    "desc": {"type": "string"}, 
                    "whereto": {"type": "array", "items": {"type": "string"}}, 
                    "rubricLinks": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["id", "phase", "title", "desc"]
            }
        }
    if task_type == 'RECOMMEND_GRASPS':
        return {"type": "object", "properties": {"goal": {"type": "string"}, "role": {"type": "string"}, "audience": {"type": "string"}, "situation": {"type": "string"}, "product": {"type": "string"}}, "required": ["goal", "role", "audience", "situation", "product"]}
    if task_type == 'RUBRIC':
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "항목명": {"type": "string"},
                    "매우잘함": {"type": "string"},
                    "잘함": {"type": "string"},
                    "보통": {"type": "string"},
                    "노력요함": {"type": "string"}
                },
                "required": ["항목명", "매우잘함", "잘함", "보통", "노력요함"]
            }
        }
    if task_type == 'SUMMARIZE_TITLE':
        return {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    return {"type": "object"}

if __name__ == '__main__':
    # 로컬용이므로 포트 충돌 방지를 위해 기본 5001 사용
    app.run(debug=True, port=5001, host='0.0.0.0')
