# TeamProject 회의록

1. 주제 : OpenCV(OpenVINO)를 이용한 텍스트 출력 및 번역 프로그램(간단한 회화. 영어)
   
1.1 이유 : 해외로 가서 사용할 때 그 때마다 쓰는 말을 계속 적어야 하는 번거로움이 있음.
   히스토리 기능을 추가.
1.2 이유2 : 메뉴판, 설명서, 표지판 등을 카메라로 비추었을 때 바로 출력해주면 좋음. 
   즉시 번역해주도록 하는 것을 원함
1.3 이유3 : 한국 자막은 없고 다른 언어의 자막은 있을 때 그 텍스트를 받아서 번역해주면 좋을 것 같음

1.4 부가 기능1 : 잘못된 텍스트(오타) 수정 프로그램
1.5 부가 기능2 : 출력된 내용과 관련된 링크 출력시키는 프로그램
1.6 부가 기능3 : 언어 추가
1.7 부가 기능4 : 히스토리 기능 또는 장소에 맞는 회화문 즐겨찾기(저장소)
1.8 부가 기능5 : 욕설, 차별과 관련된 내용 추가
1.9 부가 기능6 : 출력된 텍스트의 언어를 번역해서 음성으로 출력
1.10 부가 기능7 : 악필교정 및 폰트 변경, 자신의 필체를 원하는 폰트로 변경

1.11 학습 모델 : 텍스트 출력에 Faster R-CNN사용. 출력된 텍스트에 대한 번역은 LLM모델 사용.
1.12 데이터셋 : 간단한 회화 100문장으로 데이터셋 구성. 깔끔한 이미지부터 블러처리된 이미지, 회전된 이미지 등 다양하게 구성.
1.13 언어 : 파이썬
1.14 필요한 기술 : OpenCV(OpenVINO), DeepLearning(LLM, Faster R-CNN etc), Vscode, 데이터셋을 다양하게 구성하기 위한 프로그램 Ex) imgaug는 이미지 증강을 위한 파이썬 라이브러리, Augmentor는 이미지 데이터셋을 쉽게 증강할 수 있는 파이썬 라이브러리
TextAttack는 텍스트 데이터셋을 증강하고 공격하는 데 사용되는 파이썬 라이브러리

1.15 과정 : 
1) 요구 사항 분석:
 먼저 프로그램이 무엇을 수행해야 하는지 명확히 이해해야 합니다. 예를 들어, 어떤 종류의 텍스트를 추출하고 어떤 언어로 번역해야 하는지 등을 결정해야 합니다.

2) 데이터 수집:
 프로그램에 필요한 데이터를 수집합니다. 이는 번역을 위한 병렬 코퍼스(예: 영어-한국어 번역에는 영어 문장과 그에 대응하는 한국어 문장의 집합)를 포함할 수 있습니다.

3) LLM 모델 선택 또는 학습:
 적절한 LLM 모델을 선택하거나 학습합니다. GPT, BERT, 또는 트랜스포머 등의 사전 학습된 모델을 사용할 수 있습니다. 필요에 따라 모델을 미세 조정하여 번역 작업에 맞게 조정할 수도 있습니다.

4) 텍스트 추출 기능 구현:
 선택한 LLM 모델을 사용하여 텍스트 추출 기능을 구현합니다. 이는 모델을 텍스트 입력에 적용하고, 결과를 처리하여 추출된 텍스트를 반환하는 것을 포함합니다.

5) 번역 기능 구현:
 선택한 LLM 모델을 사용하여 번역 기능을 구현합니다. 이는 번역하려는 텍스트를 모델에 입력하고, 모델이 해당 텍스트를 번역한 결과를 반환하는 것을 포함합니다.
=============================================================================================================
6) 사용자 인터페이스 설계 및 구현: (추후 시간 및 비용이 충분하다면 가능하다. 안되면 외주한다. 비용을 달라)
 사용자가 프로그램을 쉽게 사용할 수 있도록 사용자 인터페이스(UI)를 설계하고 구현합니다. 이는 텍스트 입력란, 번역 결과 출력란, 버튼 등을 포함할 수 있습니다.
=============================================================================================================
8) 프로그램 테스트 및 디버깅:
 프로그램을 테스트하여 예상대로 작동하는지 확인하고, 필요한 경우 버그를 수정합니다.

9) 배포 및 유지 보수:
 프로그램을 배포하고 사용자들이 사용할 수 있도록 만듭니다. 또한 프로그램을 지속적으로 관리하고 유지 보수하여 성능을 개선하고 새로운 기능을 추가할 수 있도록 합니다.

=============================================================================================================

2. 주제2 : 음성으로 입력된 언어를 번역해주는 프로그램
   
2.1 이유 : 적는거 귀찮으니까 음성으로 바로 입력받아서 텍스트로 출력해주면 편함
2.2 이유2 : 자막이 없어서 영화 및 드라마를 보기 힘들 때 바로바로 음성을 받아서 우리말로 번역해주면 좋을 것 같음
   
2.3 부가 기능1 : 음성으로 받은 언어를 번역해서 음성으로 출력
2.4 부가 기능2 : 언어 추가
2.5 부가 기능3 : 욕설, 차별과 관련된 내용 추가

2.6 장비 : 마이크, 스피커, 데스크탑

=============================================================================================================
3. 주제3 : 논문 번역 및 관련 링크 출력 프로그램
3.1 이유 : 논문은 전부 영어임. 우리말이 필요함. 난 영어가 싫어
3.2 이유2 : 유사한 논문을 찾을려면 인력, 시간, 비용이 많이 듬. 비용적인 측면에서 고려할만함

3.3 부가 기능1 : 언어 추가
3.4 부가 기능2 : 주제와 관련된 논문 추천

=============================================================================================================
4. 주제4 : 내 질문에 대한 답을 찾아주는 프로그램(건강 및 식단 주제)
4.1 이유 : 세상에는 결정장애들이 생각보다 많음. 그들의 시간낭비를 줄여주기 위한 프로그램이 필요
4.2 이유2 : 운동하는 사람들이 많음. 당신들이 먹는 음식에 대한 칼로리 계산 및 추천 음식이 필요
4.3 이유3 : 당뇨병 환자들을 위한 건강관리 프로그램 등이 부족함. 그들을 위한 당 관리, 식단 관리 필요

4.4 주 타겟 : 결정장애 / 식단관리가 필요한 사람 / 20~30대 건강관리, 다이어트, 운동에 관심이 많은 일반인 / 당뇨병 환자
4.5 기능 : 음식이름을 기입(텍스트) 또는 카메라로 이미지를 찍거나 비추었을 때, 그에 해당하는 칼로리 및 구성성분 등을 출력
4.6 기능2 : 오늘 뭐 먹지? 를 입력했을 때 랜덤으로 음식추천

4.7 부가 기능 : 현재 나의 위치를 반영하여 주변 반경 1~3Km내의 음식점 추천
4.8 부가 기능2 : 주식 / 디저트 / 음료 카테고리 생성

칼로리 계산을 위해서는 음식 종류도 중요하지만 중량도 중요함
면접관을 설득할 시나리오가 필요함(정확한 근거 데이터가 필요함, 헬스케어 쪽에 대한 솔루션에 대한 기반지식이 많이 필요. 논리적인 설득 반드시 필요)
어려운 주제 : 팀웍이 잘 맞으면 빌드업 하기 좋음. 범위를 좁게 하면 재미가 없고 넓히면 구현이 힘들 수 있음

앞의 3개의 프로젝트는 기존의 있는 앱이나 프로그램임 => 기존과 차별된 점이 필요함

집구석에 있는 사람들 위한 프로그램 => 유튜브, 넷플릭스, 디즈니 등에 부족한 자막 및 통역

기술 스택 리스트업 이후 프로젝트를 구현하기 위한 몇 개의 시스템을 구현.

