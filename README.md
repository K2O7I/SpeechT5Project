
**Project package requirements:**
   * Python 3.10.12
   * accelerate 0.26.0 >>> pip install accelerate
   * ffmpeg 4.4.2 >>> pip install python-ffmpeg==4.4.2
   * huggingface-hub 0.20.2 >>> pip install huggingface-hub
   * CUDA > 12
---
+ **To run the training**:
1. run "huggingface-cli" login and input "<HF_TOKEN>"
2. run "pip install -r requirements.txt"
3. run "python3 download.py"
4. Remember download path: it has the structure:
   
   > folder
   
   >> model
   
   >>> vocoder
 
   >>> processor
   
   >>> model

   >> dataset
   
5. python3 main.py --lr 1e-5 --eps 100 --batch_size 16 --output_dir "./SpeechT5"
