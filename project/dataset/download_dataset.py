# download the VQA questions and answers (annotations)
import os
from os import path as osp
dir_raw = './testing/' # enter directory here
dir_zip = osp.join(dir_raw, 'zip')
os.system('mkdir -p '+dir_zip)
dir_ann = osp.join(dir_raw, 'Research') # enter here
os.system('mkdir -p '+dir_ann)

os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+dir_zip)
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+dir_zip)
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '+dir_zip)
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+dir_zip)
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+dir_zip)
os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Train_mscoco.zip')+' -d '+dir_ann)
os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Val_mscoco.zip')+' -d '+dir_ann)
os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Test_mscoco.zip')+' -d '+dir_ann)
os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Train_mscoco.zip')+' -d '+dir_ann)
os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Val_mscoco.zip')+' -d '+dir_ann)

os.system('mv '+osp.join(dir_ann, 'v2_mscoco_train2014_annotations.json')+' '
                +osp.join(dir_ann, 'mscoco_train2014_annotations.json'))
os.system('mv '+osp.join(dir_ann, 'v2_mscoco_val2014_annotations.json')+' '
                +osp.join(dir_ann, 'mscoco_val2014_annotations.json'))
os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_train2014_questions.json')+' '
                +osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json'))
os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_val2014_questions.json')+' '
                +osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json'))
os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test2015_questions.json')+' '
                +osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json'))
os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')+' '
                +osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json'))