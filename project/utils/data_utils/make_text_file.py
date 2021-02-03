# load jeopardy dataset
# pick 100 random indices
# print question, answer pairs
import torch
saved_train_file = "test.pt"
dataset = torch.load(saved_train_file)

idx2word = {v: k for k, v in dataset.word2idx.items()}
random_idx = torch.randint(0, dataset.i, (100,))

def write_to_disk(stuff):
    with open('qa_text', 'a') as f:
        f.write(stuff)
        print(stuff)
        f.write('\n')

def get_words(indices):
    final_sent = ""
    for i in indices:
        final_sent += idx2word[i] + " "
    return final_sent

# write to file
for i in random_idx:
    question, _, answer = dataset[i]
    question = get_words(question)
    answer = get_words([answer])
    write_to_disk(question + " : " + answer)