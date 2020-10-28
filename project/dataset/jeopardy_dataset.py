import json
from PIL import Image
from torch.utils.data import Dataset
import os
import string
from tqdm import tqdm
import nltk

START_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class JeopardyDataset(Dataset):
    def __init__(self, questions_file, answers_file, images_dir, transform, word2idx=None, train=True, q_len=8, ans_len=2, test_split=0.2):
        """
        Args:
            questions_file (string): Path to the json file with questions.
            answers_file (string): Path to the json file with annotations.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            word2idx: word2idx[word] = index, constructed on the train set only, must be passed in for the test set.  
                
        Example call:
            test_ds = JeopardyDataset("v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json", "images")

        """

        # initializing the lengths of our questions and answers
        self.q_len = q_len
        self.ans_len = ans_len

        # return test or train set?
        self.train = train

        q_f = open(questions_file, 'r')
        a_f = open(answers_file, 'r')
        
        # we want only 80% of the question/answer text to build the vocabulary
        self.questions = json.load(q_f)["questions"]
        self.answers = json.load(a_f)["annotations"]
        self.images_dir = images_dir
        self.image_id_to_filename = self._find_images()
        self.transform = transform
        self.question_to_answer = self._find_answers()
        self.mega_dict = {} # this will store our indices
        self.numerical = {}
        if train:
            # vocab made only from the train set
            self.vocab = self._make_vocab(test_split)
            self.word2idx = self._build_word2idx(self.vocab)
        else:
            self.word2idx = word2idx

        index = 0

        # the starting index of the test split might not be as simple as len(dataset) * (1-test_split)
        # since we are only considering questions with an answer length of ans_len
        # if this constraint was removed, life would probably be that easy and we wouldn't need to think 
        # about this subtlety 
        self.test_split_start = 0
        
        split_index = int(len(self.questions) * (1-test_split))
        for i, question in tqdm(enumerate(self.questions)):
            sentence_array = self._preprocess_sentence(question['question'])
            
            if not sentence_array:
                continue
            
            # not all questions have 1-word answers, and for some reason, the dataset does not seem to have a corresponding image for each question
            if question['question_id'] in self.question_to_answer and question["image_id"] in self.image_id_to_filename:
                self.numerical[question["question_id"]] = [self.word2idx[word] if word in self.word2idx else -1 for word in sentence_array]
                answer_word = self.question_to_answer[question["question_id"]]
                if answer_word not in self.word2idx:
                    answer_word = UNK_TOKEN
                answer_word = self.word2idx[answer_word]
                self.mega_dict[index] = question, self.image_id_to_filename[question["image_id"]], answer_word
                index += 1
                if not self.test_split_start and i >= split_index:
                    self.test_split_start = index

        self.index = index
        

    def _build_word2idx(self, vocab):
        '''
        word2idx[PAD_TOKEN] = 0
        '''
        word2idx = {w:i for i, w in enumerate(vocab)}
        return word2idx

    def _make_vocab(self, split):
        '''
        Makes the vocabulary out of the questions and corresponding answers in our 
        train set. Does not touch the test set.
        '''
        split_index = int(len(self.questions) * (1-split))
        question_text = ' '.join([q['question'] for q in self.questions[:split_index]])
        # only include answers for questions in our train set
        answer_set = [self.question_to_answer[q['question_id']] for q in self.questions[:split_index] if q['question_id'] in self.question_to_answer]
        answer_text = ' '.join(answer_set)
        vocab = nltk.word_tokenize(answer_text)
        vocab.extend(nltk.word_tokenize(question_text))
        # adding special tokens to the vocab
        vocab = [PAD_TOKEN, END_TOKEN, START_TOKEN, UNK_TOKEN] + vocab
        vocab = set(vocab)
        self.vocab_length = len(vocab)
        return vocab
        
    def _preprocess_sentence(self, sentence):
        arr = nltk.word_tokenize(sentence)
        if len(arr) > self.q_len:
            return []
        self._pad_arr(arr, self.q_len)
        return [START_TOKEN] + arr + [END_TOKEN]

    def _pad_arr(self, arr, length):
        while len(arr) < length:
            arr.append(PAD_TOKEN)
        
    def _find_answers(self):
        """
        
        There several answers for each question in the dataset, for example, 
        here's entry 0 of the answer vector for a random question:
         {"answer": "net", "answer_confidence": "maybe", "answer_id": 1}
        
        For the sake of simplicity, we only consider answers that have answer_confidence: yes,
        and are exactly one word long. We will apply a similar padding paradigm as we do with the question
        texts later on, by using the _pad_arr function. 

        """
        
        question_to_answer = {}
        for ann in self.answers:
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if answer['answer_confidence'] == 'yes' and len(actual_ans.split()) == 1:
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

    def vocabulary_length(self):
        # train has same vocab length as test set
        return self.vocab_length

    def get_split_index(self):
        '''
        Returns the start index of the test split for the dataloader to use.
        '''
        return self.test_split_start
    
    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        if self.train and idx >= self.test_split_start:
            # invalid index
            return 
        elif not self.train and idx + self.test_split_start < self.__len__():
            idx = idx + self.test_split_start
        question, image, answer = self.mega_dict[idx]
        path = os.path.join(self.images_dir, self.image_id_to_filename[question["image_id"]])
        img = self.transform(Image.open(path).convert('RGB'))
        return self.numerical[question['question_id']], img, answer
