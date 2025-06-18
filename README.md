### 1. Create an enviroment.:
```
conda create -n tracking_env python=3.8
```
```
conda activate tracking_env
```

### 2. Install Dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git@536dc9d527074e3b15df5f6677ffe1f4e104a4ab

### 3. Downloads folder resources and move it in folder deep_sort, you can ignore folder detection and only download networks folder:
https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp
### 4. Run below cmd line to detection human and create folder frames:
python detection.py --input_video [Link to your video] --output_folder [Link to your output folder]
### 5. Run below cmd line to tracking people and identify groups:
python tracking_group.py 
--input_video [Link to your video] 
--frames_dir [Link to your output folder, same above output folder] 
--epsilon [DBSCAN epsilon sush as 50, 75, 100] --threshold_overlap [threshold overlap with old group (0.5 -> 1.0)]