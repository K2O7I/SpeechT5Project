
**Project package requirements:**
   * Python 3.10.12
   * accelerate 0.26.0 >>> pip install accelerate
   * ffmpeg 4.4.2 >>> pip install python-ffmpeg==4.4.2
   * huggingface-hub 0.20.2 >>> pip install huggingface-hub
   * CUDA > 12
---
+ **To run the training**:
1. run "huggingface-cli" login and input "<HF_TOKEN>" then "y"
2. run "git clone https://github.com/K2O7I/SpeechT5Project.git"
3. run "cd SpeechT5Project"
4. run "pip install -r requirements.txt"
5. run "python3 download.py"
6. Remember download path: it has the structure:
   
   > folder
   
   >> model
   
   >>> vocoder
 
   >>> processor
   
   >>> model

   >> dataset
   
7. run python3 main.py --lr 1e-5 --eps 100 --batch_size 16 --output_dir "./SpeechT5"
