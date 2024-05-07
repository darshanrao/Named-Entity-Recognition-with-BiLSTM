def read_data(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r') as file:
        for line in file:
            
            line = line.strip()
            # print(line)
            if line:
                parts = line.split()
                # print(parts)
                current_sentence.append((parts[1], parts[2]))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

        if current_sentence:
            sentences.append(current_sentence)

    return sentences


def read_test_data(file_path):
    sentences = []
    current_sentence = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    current_sentence.append(parts[1])
                    # current_sentence += parts[1] + " "
                else:
                    print(f"Skipping invalid line: {line}")
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        # Add the last sentence if it exists
        if current_sentence:
            sentences.append(current_sentence)
    return sentences

# Specify the file paths
train_file_path = 'data/train'
dev_file_path = 'data/dev'
test_file_path = 'data/test'

# Fetch data from files
train_data = read_data(train_file_path)
dev_data = read_data(dev_file_path)
test_data_raw_sentences = read_test_data(test_file_path)

# Display a sample to verify the data
print("Train Data:")
print(train_data[:1])
print("\nDev Data:")
print(dev_data[:1])
print("\nTest Data - Raw Sentences:")
print(test_data_raw_sentences[:1])



from collections import Counter

def create_vocab(data):
    # Initialize a list to collect all words
    all_words = []
    # Loop over each sentence in the data
    for sentence in data:
        # Loop over each word/tag pair in the sentence
        for word, tag in sentence:
            # Add the word to the list of all words
            all_words.append(word)
    
    # Count the frequency of each word
    vocab_counter = Counter(all_words)
    # Create a list of words starting with '<UNK>' for unknown words
    vocab = ['<PAD>','<S>','</S>','<UNK>']
    # Extend the vocab list with words from the vocabulary counter
    vocab.extend(vocab_counter.keys())
    



    # Create a dictionary that maps words to unique indices
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return word_to_ix



tag_to_ix = {'I-ORG': 3,
 'B-LOC': 4,
 'O': 5,
 'B-ORG': 6,
 'B-MISC': 7,
 'I-LOC': 8,
 'B-PER': 9,
 'I-MISC': 10,
 'I-PER': 11,
 '<PAD>': 0,
 '<S>': 1,
 '</S>': 2}


def convert_to_indices(data, word_to_ix, tag_to_ix):
    """
    Convert words and tags in the dataset to their corresponding indices.

    :param data: List of sentences with word-tag pairs, e.g., [[('Word1', 'Tag1'), ...], ...]
    :param word_to_ix: Dictionary mapping words to their corresponding indices.
    :param tag_to_ix: Dictionary mapping tags to their corresponding indices.
    :return: List of converted sentences with word and tag indices.
    """
    indexed_data = []

    for sentence in data:
        # Separate words and tags, and convert them to their respective indices
        word_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word, _ in sentence]
        tag_indices = [tag_to_ix[tag] for _, tag in sentence]
        word_indices=[word_to_ix['<S>']]+word_indices+[word_to_ix['</S>']]
        tag_indices=[tag_to_ix['<S>']]+tag_indices+[tag_to_ix['</S>']]
        indexed_sentence = list(zip(word_indices, tag_indices))
        indexed_data.append(indexed_sentence)

    return indexed_data

word_to_ix = create_vocab(train_data)

indexed_train_data = convert_to_indices(train_data, word_to_ix, tag_to_ix)

indexed_dev_data=convert_to_indices(dev_data, word_to_ix, tag_to_ix)


import torch
from torch.utils.data import Dataset

class WordLabelDataset(Dataset):
    def __init__(self, data):
        """
        Initialization method for the dataset.

        Parameters:
        data (list of lists of tuples): The dataset, where each tuple represents a word and its label.
        """
        self.data = data

    def __len__(self):
        """
        Returns the total number of sentences (lists) in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Generates one sample of data, i.e., one sentence and its labels.
        """
        # Retrieve the sentence data at the specified index
        sentence_data = self.data[index]
        
        # Separate words and labels
        words, labels = zip(*sentence_data)

        # Convert words and labels to tensors if necessary
        # For example, if words are indices and labels are numerical tags
        words_tensor = torch.tensor(words, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return words_tensor, labels_tensor


# Create an instance of the dataset
train_dataset = WordLabelDataset(indexed_train_data)
dev_dataset = WordLabelDataset(indexed_dev_data)

# Use the dataset with a DataLoader
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define a collate function that can handle padding if necessary
def custom_collate_fn(batch):
    """
    Custom collate function for padding variable length sequences.

    Args:
        batch: List of tuples with (sequence, label).

    Returns:
        A tuple of two tensors: padded sequences and corresponding labels.
    """
    # Separate the sequences and labels in the batch
    sequences, labels = zip(*batch)

    # Pad the sequences to have the same length
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Stack labels into a tensor
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return sequences_padded, labels_padded

# Define batch size for DataLoader
batch_size = 32  # You can adjust this size as needed

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)  # Usually, we don't shuffle the test set


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the BiLSTM model according to the provided hyperparameters
class BiLSTMForNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(BiLSTMForNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, 128)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.33)
        self.hidden2tag2 = nn.Linear(128, tagset_size)

    def forward(self, sentence):
        x = self.embedding(sentence)
        x, _ = self.bilstm(x)
        x = self.hidden2tag(x)
        x = self.elu(x)
        x=self.dropout(x)
        tag_scores = self.hidden2tag2(x)
        return tag_scores

# Initialize the model with the specified hyperparameters
vocab_size = len(word_to_ix)  # Your vocabulary size
embedding_dim = 100
hidden_dim = 256  # LSTM hidden dimensions
tagset_size = len(tag_to_ix) # Number of tags in your tagset



model = BiLSTMForNER(vocab_size, embedding_dim, hidden_dim, tagset_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define the loss function and optimizer with the specified settings

# weights = torch.tensor([0,0.5,0.5,1,1,0.1,1,1,1,1,1,1]) 
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Using SGD as specified with a learning rate of 0.1

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.1, verbose=True)


# # Training loop with early stopping 
#Commented because it is already trained
# num_epochs = 100  # Number of epochs
# patience = 10 # Number of epochs to wait for improvement in validation loss

# best_valid_loss = float('inf')
# no_improvement_counter = 0

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for sentences, tags in train_loader:  # Assuming you have a train_loader
#         sentences, tags = sentences.to(device), tags.to(device)
#         model.zero_grad()
#         tag_scores = model(sentences)
#         loss = loss_function(tag_scores.view(-1, tagset_size), tags.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         valid_loss = 0
#         for sentences, tags in dev_loader:  # Assuming you have a valid_loader
#             sentences, tags = sentences.to(device), tags.to(device)
#             tag_scores = model(sentences)
#             valid_loss += loss_function(tag_scores.view(-1, tagset_size), tags.view(-1)).item()

#     avg_valid_loss = valid_loss / len(dev_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {avg_valid_loss}")

#     scheduler.step(avg_valid_loss)

#     # Check for early stopping
#     if avg_valid_loss < best_valid_loss:
#         best_valid_loss = avg_valid_loss
#         no_improvement_counter = 0
#     else:
#         no_improvement_counter += 1

#     if no_improvement_counter >= patience:
#         print(f'Early stopping at epoch {epoch+1} as there is no improvement in validation loss for {patience} consecutive epochs.')
#         break


# Load the saved state dictionary into the model
model.load_state_dict(torch.load('blstm1.pt'))


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Variables to accumulate metrics
sentence_level_true_tags = []
sentence_level_pred_tags = []

model.eval()
with torch.no_grad():
    for sentences, tags in dev_loader:
        sentences, tags = sentences.to(device), tags.to(device)
        tag_scores = model(sentences)

        # Get the predicted tags by taking the argmax of the log probabilities
        _, predicted_tags = torch.max(tag_scores, dim=2)
        
        # Process predictions and true tags sentence-wise
        for i in range(sentences.size(0)):  # Loop through each sentence in the batch
            sentence_length = (tags[i] != 0).sum().item()  # Calculate the true sentence length
            sentence_true_tags = tags[i][:sentence_length].cpu().tolist()
            sentence_pred_tags = predicted_tags[i][:sentence_length].cpu().tolist()
            
            sentence_level_true_tags.append(sentence_true_tags)
            sentence_level_pred_tags.append(sentence_pred_tags)



def remove_start_end(lst):
    new_lst = []
    for sublst in lst:
        # Check if the first element is 1 or 2 and remove it
        if sublst and sublst[0] in [1, 2]:
            sublst = sublst[1:]
        
        # Check if the last element is 1 or 2 and remove it
        if sublst and sublst[-1] in [1, 2]:
            sublst = sublst[:-1]

        new_lst.append(sublst)
    return new_lst


sentence_level_true_tags = remove_start_end(sentence_level_true_tags)
sentence_level_pred_tags = remove_start_end(sentence_level_pred_tags)




# Create the reverse mapping from indices to tags
ix_to_tag = {index: tag for tag, index in tag_to_ix.items()}




word_pred_tag_pairs = []

for sentence, preds in zip(dev_data, sentence_level_pred_tags):
    # Ensure that the number of words and predictions match
    
    if len(sentence) != len(preds):
        print(sentence)
        print(preds)
        print(len(sentence),len(preds))
        print("Length mismatch between words and predicted tags")
        break
    
    # Pair each word with its predicted tag
    paired_sentence = [(word, ix_to_tag.get(pred, 'O')) for (word, _), pred in zip(sentence, preds)]
    word_pred_tag_pairs.append(paired_sentence)


def convert_and_write_to_file(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        sentence_index = 1
        for sentence in data:
            for index, (token, label) in enumerate(sentence, start=1):
                output_file.write(f"{index} {token} {label}\n")
            output_file.write("\n")
            sentence_index += 1

output_file_path = 'dev1.out'
convert_and_write_to_file(word_pred_tag_pairs, output_file_path)





#TEST DATA

def transform_data(data):
    transformed_data = []
    for sublist in data:
        transformed_sublist = [(item, 'O') for item in sublist]
        transformed_data.append(transformed_sublist)
    return transformed_data


test_data = transform_data(test_data_raw_sentences)

indexed_test_data=convert_to_indices(test_data, word_to_ix, tag_to_ix)


test_dataset = WordLabelDataset(indexed_test_data)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


# Variables to accumulate metrics
sentence_level_true_tags = []
sentence_level_pred_tags = []

model.eval()
with torch.no_grad():
    for sentences, tags in test_loader:
        sentences, tags = sentences.to(device), tags.to(device)
        tag_scores = model(sentences)

        # Get the predicted tags by taking the argmax of the log probabilities
        _, predicted_tags = torch.max(tag_scores, dim=2)
        
        # Process predictions and true tags sentence-wise
        for i in range(sentences.size(0)):  # Loop through each sentence in the batch
            sentence_length = (tags[i] != 0).sum().item()  # Calculate the true sentence length
            sentence_true_tags = tags[i][:sentence_length].cpu().tolist()
            sentence_pred_tags = predicted_tags[i][:sentence_length].cpu().tolist()
            
            sentence_level_true_tags.append(sentence_true_tags)
            sentence_level_pred_tags.append(sentence_pred_tags)


sentence_level_true_tags = remove_start_end(sentence_level_true_tags)
sentence_level_pred_tags = remove_start_end(sentence_level_pred_tags)



word_pred_tag_pairs = []

for sentence, preds in zip(test_data, sentence_level_pred_tags):
    # Ensure that the number of words and predictions match
    
    if len(sentence) != len(preds):
        print(sentence)
        print(preds)
        print(len(sentence),len(preds))
        print("Length mismatch between words and predicted tags")
        preds = preds[:len(sentence)]
    
    # Pair each word with its predicted tag
    paired_sentence = [(word, ix_to_tag.get(pred, 'O')) for (word, _), pred in zip(sentence, preds)]
    word_pred_tag_pairs.append(paired_sentence)


def convert_and_write_to_file(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        sentence_index = 1
        for sentence in data:
            for index, (token, label) in enumerate(sentence, start=1):
                output_file.write(f"{index} {token} {label}\n")
            output_file.write("\n")
            sentence_index += 1

output_file_path = 'test1.out'
convert_and_write_to_file(word_pred_tag_pairs, output_file_path)







