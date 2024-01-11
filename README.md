
**Project package requirements:**
   ++ Python 3.10.12
   ++ accelerate 0.26.0
   ++ ffmpeg 4.4.2
   ++ huggingface-hub 0.20.2
+ To run the training:
1. pip install -r requirements.txt
2. python3 download.py
3. Remenber download: it have the structure:
   
   -folder
   
   --model
   
   ---vocoder
   
   ---processor
   
   ---model
   
   --dataset
   
5. MODEL_DIR="<folder when running download.py>"python3 main.py --lr 1e-5 --eps 100 --batch_size 16 --output_dir "./SpeechT5"
