# MobileYoloNet
Object detection neural network combining `MobileNetV3` and `YOLOv3`. (Work In Progress)


## Environment
Including but not limited to: Ubuntu 18.04.4 LTS, Windows Subsystem for Linux (WSL).

|  | version |
| ------ | ------: |
| Python | 3.6.9 |
| Tensorflow | 2.1.0 |


## Usage
1. To run the code, you need to create two folders.
   ```bash
   mkdir data
   mkdir log
   ```

2. Edit and run `tool/voc_ann.py` to create `train_file` for Pascal VOC dataset

   **or**

   prepare *images* and *train_file* with your own data.

   `train_file` is put in `./data` and looks like:
   ```
   /path/to/images/1.jpg 58,63,430,301,1
   /path/to/images/2.jpg 37,139,200,366,0 170,175,295,375,0
   /path/to/images/3.png 34,18,484,397,4
   /path/to/images/4.png 229,166,422,375,0 286,152,322,182,9 233,157,268,181,9 181,157,216,184,9 328,136,432,205,9
   /path/to/images/5.other 1,22,458,375,18 112,78,376,328,4
   ```
   - Every 5 numbers represent 1 ground truth box.
   - Each number in the 5 numbers represents: xmin, ymin, xmax, ymax, category

3. Config your training process in `config.py`.

4. Start training with:
   ```bash
   python3 train.py
   ```

5. To use the trained model for predicting, prepare `train_file` like:
   ```
   /path/to/images/1.jpg
   /path/to/images/2.jpg
   /path/to/images/3.png
   /path/to/images/4.png
   /path/to/images/5.other
   ```
   Start predicting with:
   ```bash
   python3 predict.py
   ```

6. To predict a video, edit `predict_video.py` to point to your video and run:
   ```shell
   python3 predict_video.py
   ```
