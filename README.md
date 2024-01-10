+ To run the training:
1. pip install -r requirements.txt
2. python3 download.py
3. Remenber download: it have the structure:
   &nbsp;|<folder>
   &nbsp;&nbsp;|-model
   <space>*<space>|     <space>*<space>|-vocoder
   <space>*<space>|     <space>*<space>|-processor
   <space>*<space>|     <space>*<space>|-model
   <space>*<space>|-dataset
5. MODEL_DIR="<folder when running download.py>"python3 main.py --lr --eps --batch_size --output_dir
