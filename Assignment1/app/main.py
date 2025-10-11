# 1. 기본 설정
import os
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

def main():
    # 2. ollama 서버 연동
    print("챗봇 초기화 중... 로컬 sLLM 서버에 연결 중")
    for service_name, service_url in [("Ollama", OLLAMA_URL)]:
        while True:
            try:
                import requests
                response = requests.get(service_url)
                if response.status_code == 200:
                    print(f"{service_name} 서버가 준비됨")
                    break
            except requests.exceptions.ConnectionError:
                print(f"{service_name} 서버를 기다리는 중...")
                time.sleep(5)

    import ollama
    
    # 3. sLLM, 임베딩 모델 다운로드
    def ensure_model_pulled(model_name: str, ollama_url: str):
        
        try:
            # http://ollama:11434 → 'ollama'만 추출
            host = ollama_url.split('//')[1].split(':')[0]
            
            # ollama 서버와 통신하기 위한 클라이언트 객체 생성
            client = ollama.Client(host=host)
            
            # ollama 서버에 모델 정보 요청
            client.show(model_name)
            print(f"모델 '{model_name}'이 이미 존재함")
            
        except ollama.ResponseError as e:
            # 모델 없음 코드: 404
            if e.status_code == 404:
                print(f"모델 '{model_name}'을(를) 찾을 수 없음")
                print(f"모델 '{model_name}' 다운로드 중...")
                
                # ollama 서버에 모델 다운로드 요청
                stream = client.pull(model_name, stream=True)
                
                # 진행률 출력
                for progress in stream:
                    if 'total' in progress and 'completed' in progress:
                        total = progress['total']
                        completed = progress['completed']
                        percentage = (completed / total) * 100
                        print(f'{progress["status"]}: {percentage:.2f}% 완료', end='\r')
                    else:
                        print(' ' * 80, end='\r')
                        print(progress.get('status'))
                print()
                print(f"모델 '{model_name}' 다운로드 완료")
            else:
                raise e

    ensure_model_pulled("mistral:7b", OLLAMA_URL)
    ensure_model_pulled("nomic-embed-text", OLLAMA_URL)

    # 4. sLLM, 임베딩 모델 로드
    llm = Ollama(model="mistral:7b", base_url=OLLAMA_URL)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
    print("채팅 및 임베딩 모델 로드 완료")

    print("\n--- 로컬 sLLM 기반 챗봇 ---")
    print("질문해보세요. (종료: 'exit' 또는 'quit')")

    # 5. 챗봇 대화 루프
    while True:
        try:
            # 질문 쿼리 생성
            query = input("\n> ")
            
            # query를 소문자로 변환
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            
            # 응답 쿼리 생성
            # 랭체인을 쓰지 않으므로 llm 자체 응답 처리
            # Assignment2에는 랭체인을 써서 응답 쿼리를 생성해야 함
            result = llm.invoke(query)
            print(f"\n답변: {result}")

        except (EOFError, KeyboardInterrupt):
            break
    print("\n챗봇 종료 중")

if __name__ == "__main__":
    main()
