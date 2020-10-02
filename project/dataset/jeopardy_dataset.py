import json
from PIL import Image
from torch.utils.data import Dataset
import os
import string
from tqdm import tqdm

class JeopardyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, questions_file, answers_file, images_dir, transform):
        """
        Args:
            questions_file (string): Path to the json file with questions.
            answers_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        q_f = open(questions_file, 'r')
        a_f = open(answers_file, 'r')
        
        self.questions = json.load(q_f)["questions"]
        self.answers = json.load(a_f)["annotations"]
        self.images_dir = images_dir
        self.image_id_to_filename = self._find_images()
        self.transform = transform
        self.question_to_answer = self._find_answers()
        self.mega_dict = {} # this will store our indices
        self.numerical = {}

        answer_text = ' . '.join(list(self.question_to_answer.values()))
        question_text = ' . '.join([q['question'] for q in self.questions])
        question_text = question_text.replace("?", " END")
        # strip all punctuation
        question_text = question_text.translate(str.maketrans('', '', string.punctuation))
        tokens = question_text.split(' ')
        tokens.extend(answer_text.split(' '))
        vocab = set(tokens)
        vocab_length = len(vocab)
        print(vocab_length)
        print("vocab length")
        vocab = list(vocab)
        word2idx = {w:i for i,w in enumerate(vocab)}
        word2idx["DUMMY"] = len(vocab)
        
        index = 0
        for question in tqdm(self.questions):
          arr = question['question']
          arr = arr.replace("?", " END").translate(str.maketrans('', '', string.punctuation)).split()
          # dataset reduced by only 50%, and RNN is easier, don't have to worry about padding now
          if len(arr) < 6 or len(arr) > 7:
            continue
          # add padding to arr, and separate the question mark
          if len(arr) == 6:
            arr.append("DUMMY")
          # len is always 7
          if question["image_id"] in self.image_id_to_filename and question['question_id'] in self.question_to_answer:
              self.numerical[question["question_id"]] = [word2idx[word] for word in arr]
              self.mega_dict[index] = question, self.image_id_to_filename[question["image_id"]], word2idx[self.question_to_answer[question["question_id"]]]
              if index < 10:
                  print("index is " + str(index))
                  print(self.mega_dict[index])
              index += 1

        self.index = index
        print("length is " + str(index))

    def _find_answers(self):
        # TODO: deal with the complications - there several answers for each question, 
        # here's entry 0: {"answer": "net", "answer_confidence": "maybe", "answer_id": 1}
        question_to_answer = {}
        for ann in self.answers:
            for answer in ann['answers']:
              actual_ans = answer['answer']
              if answer['answer_confidence'] == 'yes' and len(actual_ans.split()) == 1:
                # whatever the first yes is, scrappy model; assuming every field has at least one yes answer
                question_to_answer[ann["question_id"]] = actual_ans
        return question_to_answer

    def _find_images(self):
        id_to_filename = {}
        
        for filename in os.listdir(self.images_dir):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            image_id = int(id_and_extension.split('.')[0])
            id_to_filename[image_id] = filename
            
        return id_to_filename


    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        question, image, answer = self.mega_dict[idx]
        path = os.path.join(self.images_dir, self.image_id_to_filename[question["image_id"]])
        img = self.transform(Image.open(path).convert('RGB'))
        if len(self.numerical[question['question_id']]) != 7:
          print("something wrong")
        # convert question to numerical representation here?
        return self.numerical[question['question_id']], img, answer

# example call
# test_ds = JeopardyDataset("v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json", "images")

