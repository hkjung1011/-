# 로컬 검사 앱 API · R01

기본 주소: `http://127.0.0.1:8893`. `Start-Platform.cmd`로 실행하거나 프로젝트 Python 환경에서 `python -m uvicorn app:app --host 127.0.0.1 --port 8893`로 실행한다. `/docs`에서 요청·응답 스키마를 확인할 수 있다. 외부 서비스로 이미지를 전송하지 않는다.

`Setup.ps1`은 프로젝트 `.venv`에 CPU용 환경을 설치한다. 모델 가중치는 별도로 준비하며, 공개 데이터 추가 학습에는 감사된 데이터와 CUDA 환경이 필요하다. 모델·데이터 준비 범위는 [설치와 실행](README.md#설치와-실행)을 참고한다.

| 요청 | 동작 |
|---|---|
| `GET /` | 한국어 검사 UI |
| `GET /api/status` | 모델 준비 상태, 기본 채택 모델, 보고서, 학습 프로세스 상태 |
| `GET /api/samples` | test → val → train 순으로 최대 72개 샘플 |
| `GET /api/sample-image/{sample_id}` | 서버가 발급한 샘플 ID의 이미지 |
| `POST /api/infer` | multipart: `file` 또는 `sample_id` 중 하나, `model=baseline/finetuned/field_finetuned`, `confidence=0.01~0.95`, `tiled=true/false` |
| `GET /api/results/{id}` | 검출 박스가 그려진 JPG |
| `GET /api/inputs/{id}` | 검사에 사용한 원본 이미지의 JPG 저장본 |
| `GET /api/history?limit=30` | 최근 이력 1~200개 |
| `POST /api/training/start` | JSON `{"epochs":3}`. 3, 5, 10 epoch 중 선택 |
| `GET /api/training/status` | 진행 중인 작업과 `training_status.json` |
| `GET /api/training/log/{job_id}` | 해당 UI 학습 실행 로그의 마지막 12 KB |
| `POST /api/camera/capture` | JSON `{"camera_index":0,"model":"baseline","confidence":0.25,"tiled":false}`. OpenCV로 한 장 촬영 후 추론 |
| `GET /hardware/design.html` | 카메라·조명·이송부 설계 문서 |
| `GET /reports/summary.html` | 로컬 HTML 리포트가 생성된 경우에 제공 |

추론 응답에는 ID, 시각, 모델, CPU 장치, 입력 크기, 추론 시간, 박스 수, 클래스·신뢰도·전역 xyxy 좌표, 이미지 URL이 포함된다. 앱 추론은 GPU 학습과의 경합을 줄이기 위해 CPU에서 수행한다. 전체 프레임의 추론 크기는 1024이다.

타일 모드는 입력 이미지를 원래 픽셀 크기의 1024 패치로 나누고 약 20%를 겹친다. 패치별 배치 1 CPU 추론 후 전역 좌표를 계산하고 클래스별 IoU 0.45 NMS로 중복 후보를 줄인다. 최대 16개 패치만 처리한다. 타일 모드의 실제 검출 정확도는 검증되지 않았으며 패치 경계에 걸친 결함의 검출이 영향을 받을 수 있다.

업로드는 최대 20 MB, 4,000만 화소, 한 변 12,000픽셀이다. 실제 파일명을 저장 경로로 사용하지 않고 UUID로 저장한다. 해석할 수 없는 이미지, 과도한 크기, 잘못된 ID, 미준비 모델은 오류 응답을 반환한다. 저장 위치는 `runtime/inputs`, `runtime/results`, `runtime/history.jsonl`이다. 원본 JPEG/PNG 바이트가 필요한 데이터 수집 목적이라면 별도 원본 보존을 해야 한다. 앱 원본 보기에는 디코딩·EXIF 방향 보정 후 저장한 JPG가 표시된다.

UI 학습은 `sys.executable scripts/finetune.py --epochs N`을 `shell=False`로 실행한다. UI 자체의 원자적 작업 파일/PID/poll과 CLI의 학습 상태 보고서 PID를 확인해서 중복 실행을 막는다. 실제 CLI 스크립트의 전역 학습 잠금도 별도로 적용된다. 앱을 종료해도 이미 시작한 자식 학습 프로세스는 계속 진행될 수 있다. 진행 결과는 프로젝트 보고서와 모델 파일에 남는다.

실제 USB 카메라 촬영 버튼은 사용자가 누를 때만 동작한다. 이 API에는 모터·조명·스테이지 이동 명령이 없다. 카메라 및 실제 광학·이송 장치의 성능 상태는 `NOT VERIFIED`이다.

검증: `python -m unittest discover -s tests -p test_app.py`. 실제 모델·데이터를 이용한 CPU 전체 프레임 및 타일 추론까지 확인하려면 Windows PowerShell에서 `$env:WAFER_REAL_INFERENCE='1'`을 설정한 뒤 같은 명령을 실행한다. 테스트는 실제 추론 실행 기록을 `runtime`에 남긴다. 하드웨어 카메라는 테스트에서 접근하지 않는다.
