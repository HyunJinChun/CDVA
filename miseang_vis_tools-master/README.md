## Miseang video instance segmentation
Mieseang video instance segmentation tools.


## sample_frames.py

sample 5 frames

## miseang_coco2input_format.py

main 함수의 coco_root(input dir), vis_root(output dir) 변수 수정

coco_root는 cvat에서 생성된 json 파일들이 들어있음(1001.0001.0001.0001.0006.json, ...)

코드 실행 시 모든 json 파일들을 합해서 video instance segmentation annotation으로 변환(train.json)

## analysis_dataset.py

구축된 데이터셋에 대한 분석 코드
