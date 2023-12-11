##

## Installation.
1) python 3.9
2) you can install libraries through requirements.txt
pip install -r requirements.txt

## data organiztion.

    """ 
    [Auto Generate Test Dataset]
    DATASET Folder
        |   
        |----category1 -- img1.jpg
        |              -- img2.jpg 
        |              -- etc      
        |   
        |----category2 -- img1.jpg
        |              -- etc
        |         
        |----etc
    """

## Generating data.
python generate_dataset_auto_create_testset.py --cfg config.yaml <data_folder>

Example:
python generate_dataset_auto_create_testset.py --cfg config.yaml ./game_data

Result:
train/val/test: 0.5/0.27/0.23

## Method of testing/ prediction
We use 2 methods:
1) test.py output results with confusion matrixes and wrong match log (it's saved in check_wrong_match &
wrong_match_images, the folders need to be generated mannualy).

python test.py --cfg config.yaml --model <models_path>

example: python test.py --cfg config.yaml --model ./experiments/legenddata/2023-04-11/mobilenetv2_035
/2023-04-11_22-09-52_default/train/model_best_2023-04-11_22-09-52_default.pth.tar

2) predict.py outputs results with your requirements.

python predict.py --cfg config.yaml --model <models_path>

python predict.py --cfg ./config.yaml --model ./experiments/legenddata/2023-04-11/mobilenetv2_035/
2023-04-11_22-09-52_default/train/model_best_2023-04-11_22-09-52_default.pth.tar

You can check my outputs in results with trained models (efficientnet_b1_pruned/ mobilenetv2_035/ resnet18) that include the results of test.py (check_wrong_match) and predict.py (output.txt)

## Link for data and experiment result. ([link]([https://drive.google.com/drive/folders/1lYmy7-diSegug3Yi--bETYJZyC3OYGni?usp=sharing])
