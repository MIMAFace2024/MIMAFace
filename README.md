# MIMAFace: Face Animation via Motion-Identity Modulated Appearance Feature Learning

<a href=""><img src="https://img.shields.io/badge/arXiv-2307.10797-b31b1b.svg" height=22.5></a>
<a href='https://mimaface2024.github.io/mimaface.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://huggingface.co/MIMAFace/MIMAFace'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>


<!-- 

<p align="center">
<img src="assets/self.gif" style="height: 150px"/>
<img src="assets/cross.gif" style="height: 150px"/>
</p> -->



# Installation
```
pip install -r requirements.txt
```

## Download Models

You can download both code and models of MIMAFace directly from [here](https://huggingface.co/MIMAFace/MIMAFace/tree/main):
```python
# Download the whole project containing all code and weight files 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="MIMAFace/MIMAFace", local_dir="./MIMAFace")
```


To run the demo, you should also download the pre-trained SD models below:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)


# Inference 
Reenacted by a single image:
```
python infer_image.py \
        --source_path ./examples/source/bengio.jpg \
		--target_path ./examples/target/0000025.jpg \
		--output_path ./examples/result
```
Reenacted by a video:
```
python infer_video.py \
        --source_path ./examples/source/bengio.jpg \
		--target_path ./examples/target/id10291#TMCTm7GxiDE#000181#000465.mp4 \
		--output_dir ./examples/result
```


# Training

### Preparing Dataset 
We convert the training datasets (Voxceleb2/VFHQ) into tfrecords files and put it to `datasets/voxceleb2`.


The meta file `voxceleb2_tfrecord_train_list.txt` contains items like
```
/path/to/tfrecord             offset       image_id 
tfrecord/voxceleb2_2.tfrecord 102797351015 train_id04128#5wFKqF1MVos#004810#005018
```
We extract the face landmarks and parsing masks (not used) of the images in datasets in advance and save it to tfrecord file, so an tfrecord file contains `image, mask, pts`:
```
image, mask, pts = get_tfrecord_item(tffile, offset)
```

### Start Training 
```
# Stage 1: Train on images (4 * A100 * ~1day)
sh start_train_image.sh

# Stage 2: Train on Videos (4 * A100 * ~1day)
sh start_train_video.sh
```



