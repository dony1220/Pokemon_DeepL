# Deep Learning
## 1204 - 1208 딥러닝 프로젝트


<h3 align="left"><b>🛠 Tech Stack 🛠</b></h3>
</br>
<p align="left">
<img src="https://img.shields.io/badge/Python-blue?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp 
<img src="https://img.shields.io/badge/Tensorflow-1572B6?style=flat-square&logo=Tensorflow&logoColor=white"/></a> &nbsp


1. 포켓몬 FeatureExtractor, main, ImageProcessing(이미지 open 경로, path는 custom)
   
   ![image](https://github.com/dony1220/dl/assets/96913965/a4cc58e4-9cc1-4320-b0c6-1d9123cd3bcd)

   유사 이미지 분류를 위한 코드를 작성
   
   WorkFlow : FeatureExtractor라는 클래스를 생성하고 ->
   ImageProcessing이라는 파일에서 FeatureExtractor 클래스 객체를 생성하여 ->
   Main 파일에서 ImageProcessing이라는 함수에 Path 파라미터를 주어 돌림


3. movie_len은 main과 define_m으로만 나눠서 돌리기(데이터 셋 open 경로는 custom)

   ![image](https://github.com/dony1220/dl/assets/96913965/4619e3b5-0b01-4642-bd94-d8b50c621f7d)

   영화 관람 predict
   
   WorkFlow : define_m이라는 함수에 모델 생성을 진행했고 ->
   main 함수에서 csv파일로 저장하는 객체만을 분기하여 진행하였음
---------------------------------------------------------------------------


