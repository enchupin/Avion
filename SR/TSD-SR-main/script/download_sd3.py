import os
from huggingface_hub import snapshot_download

# 다운로드 경로 설정
download_path = os.path.join(os.getcwd(), "checkpoint", "tsdsr")
repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

print(f"=== Stable Diffusion 3 모델 다운로더 ===")
print(f"다운로드 대상: {repo_id}")
print(f"저장 위치: {download_path}")
print("-" * 50)
print("주의: 이 모델은 Hugging Face에서 사용 권한 동의(License Agreement)가 필요합니다.")
print(f"동의하러 가기: https://huggingface.co/{repo_id}")
print("-" * 50)

# 토큰 입력 받기
token = input("Hugging Face Access Token을 입력하세요 (입력 없이 엔터키 누르면 익명 시도): ").strip()

if not token:
    token = None

try:
    print("다운로드를 시작합니다... (용량이 크니 시간이 걸릴 수 있습니다)")
    snapshot_download(
        repo_id=repo_id,
        local_dir=download_path,
        token=token,
        local_dir_use_symlinks=False # 윈도우 호환성을 위해 False 권장
    )
    print("\n[성공] 모든 파일이 다운로드되었습니다!")
except Exception as e:
    print(f"\n[오류 발생] 다운로드 중 문제가 생겼습니다:\n{e}")
    if "401" in str(e) or "403" in str(e):
        print("\n[힌트] 권한 문제일 가능성이 높습니다.")
        print("1. Hugging Face 웹사이트에서 모델 사용 동의(Agree)를 눌렀는지 확인해주세요.")
        print("2. 입력한 토큰이 'Read' 권한을 가지고 있는지 확인해주세요.")
