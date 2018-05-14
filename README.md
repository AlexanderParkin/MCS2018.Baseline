
# MCS2018: Adversarial Attacks on Black-box Face Recognition

It's a baseline code for [Adversarial Attacks on Black-box Face Recognition](https://competitions.codalab.org/competitions/19090)
This code trains the white box on data from the black box model, makes iteration [FGSM attacks](https://arxiv.org/pdf/1412.6572.pdf) on the white box and makes submit zip archive.

## How to reproduce
1. download black-box model and move to baseline director
|OS|python 2.7|python 3.5|python 3.6|
|:-------:|:-------:|:-------:|:-------:|
|Ubuntu|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/ubuntu/MCS2018_CPU.so) [GPU](http://mcs2018-competition.visionlabs.ru/distribs/ubuntu/MCS2018.so)|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/ubuntu/MCS2018_CPU.cpython-35m-x86_64-linux-gnu.so) [GPU](http://mcs2018-competition.visionlabs.ru/distribs/ubuntu/MCS2018.cpython-35m-x86_64-linux-gnu.so)|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/ubuntu/MCS2018_CPU.cpython-36m-x86_64-linux-gnu.so) [GPU](http://mcs2018-competition.visionlabs.ru/distribs/ubuntu/MCS2018.cpython-36m-x86_64-linux-gnu.so)|
|CentOS|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/centos/MCS2018_CPU.so) [GPU](http://mcs2018-competition.visionlabs.ru/distribs/centos/MCS2018.so)|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/centos/MCS2018_CPU.cpython-35m-x86_64-linux-gnu.so) [GPU](http://mcs2018-competition.visionlabs.ru/distribs/centos/MCS2018.cpython-35m-x86_64-linux-gnu.so)|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/centos/MCS2018_CPU.cpython-36m-x86_64-linux-gnu.so) [GPU](http://mcs2018-competition.visionlabs.ru/distribs/centos/MCS2018.cpython-36m-x86_64-linux-gnu.so)|
|Windows|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/windows/MCS2018_CPU.pyd)|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/windows/MCS2018_CPU.cp35-win_amd64.pyd)|[CPU](http://mcs2018-competition.visionlabs.ru/distribs/windows/MCS2018_CPU.cp35-win_amd64.pyd)|

2. download [pair's data](http://mcs2018-competition.visionlabs.ru/imgs.zip), [student_model_imgs](http://mcs2018-competition.visionlabs.ru/student_model_imgs.zip), [submit list](http://mcs2018-competition.visionlabs.ru/submit_list.csv), [pairs list](http://mcs2018-competition.visionlabs.ru/pairs_list.csv) and move to data
```
python downloader.py --root ./data --main_imgs --student_model_imgs --submit_list --pairs_list
```
3. prepare data for student model
```
python prepare_data.py --root data/student_model_imgs/ --datalist_path data/datalist/ --datalist_type train --gpu_id 1;
python prepare_data.py --root data/imgs/ --datalist_path data/datalist/ --datalist_type val --gpu_id 1
```
5. train black-box student model (For example, ResNet18) on not annotated data
```
cd student_net_learning;
CUDA_VISIBLE_DEVICES=0 python main.py --name Baseline1 --epochs 20 --cuda --batch_size 32 --datalist ../data/datalist_small/ --root ../data/
```
6. make FGSM-attack on student model
```
cd ..;
CUDA_VISIBLE_DEVICES=0 python attacker.py --root ./data/imgs/ --save_root ./baseline1/ --datalist ./data/pairs_list.csv --model_name ResNet18 --checkpoint_path student_net_learning/checkpoint/Baseline1/best_model_ckpt.t7 --cuda
```

8. check ssim for submission, archive all files and make submission
```
python evaluate.py --attack_root ./baseline1/ --target_dscr ./data/val_descriptors.npy --submit_name Baseline1 --gpu_id 1
```

## How to improve
- add classification loss in student model training
- try other attacker types. [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/pdf/1801.00553.pdf)
- use teacher (black-box model) to get actual descriptors for augmentated images
