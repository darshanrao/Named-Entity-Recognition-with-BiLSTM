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


import pickle
glove_vectors = {}
# Path to your saved pickle file
file_path = 'glove_vectors.pkl'

# Load the GloVe vectors from the pickle file
with open(file_path, 'rb') as f:
    glove_vectors = pickle.load(f)



# Assuming `word_vector` is a 100-dimensional GloVe vector for a word
def extend_vector_with_case(word, word_vector):
    case_vector = [0, 0, 0, 0]  # Initialize the case vector
    if word.isupper():
        case_vector = [1, 0, 0, 0]  # is_upper
    elif word.islower():
        case_vector = [0, 1, 0, 0]  # is_lower
    elif word.istitle():
        case_vector = [0, 0, 1, 0]  # is_title
    elif not word.isalpha() or (word.lower() != word and word.upper() != word):
        case_vector = [0, 0, 0, 1]  # is_mixed

    # Concatenate the original vector with the case vector
    extended_vector = word_vector + case_vector
    return extended_vector




import numpy as np

# Function to get the word vector for a word
def get_word_vector(word):
    # Convert to lowercase to handle case insensitivity of GloVe
    word_lower = word.lower()
    word_vector = glove_vectors.get(word_lower,glove_vectors['<UNK>'])
    return np.array(word_vector)

# Function to create case features
def get_case_features(word):
    case_features = np.zeros(4)
    if word.isupper():
        case_features[0] = 1  # is_upper
    elif word.islower():
        case_features[1] = 1  # is_lower
    elif word.istitle():
        case_features[2] = 1  # is_title
    else:  # Assume any other case is mixed
        case_features[3] = 1  # is_mixed
    return case_features

# Function to create an extended word vector with case features
def create_extended_word_vector(word):
    word_vector = get_word_vector(word)
    case_features = get_case_features(word)
    # print(word_vector)
    # print(word)
    extended_vector = np.concatenate((word_vector, case_features))
    return extended_vector


# Create a list to hold all word vectors with case features
word_vectors_with_case = []

for sentence in train_data:
    for word, tag in sentence:
        extended_vector = create_extended_word_vector(word)
        word_vectors_with_case.append(extended_vector)


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

indexed_train_data = convert_to_indices(train_data, word_to_ix, tag_to_ix)

indexed_dev_data=convert_to_indices(dev_data, word_to_ix, tag_to_ix)


def convert_to_glove_vectors(data,tag_to_ix):
    """
    Convert words in the dataset to their corresponding GloVe vectors and tags to their corresponding indices.

    :param data: List of sentences with word-tag pairs, e.g., [[('Word1', 'Tag1'), ...], ...]
    :param word_vectors: Dictionary mapping words to their GloVe vectors.
    :param tag_to_ix: Dictionary mapping tags to their corresponding indices.
    :return: List of converted sentences with GloVe vectors and tag indices.
    """
    indexed_data = []

    for sentence in data:
        # Extract words and tags from the sentence
        words = [word for word, _ in sentence]
        tags = [tag for _, tag in sentence]

        # Convert words to their GloVe vectors and handle unknown words
        glove_vectors = [create_extended_word_vector(word) for word in words]
        glove_vectors = [create_extended_word_vector('<S>')]+glove_vectors+[create_extended_word_vector('</S>')]
        # Convert tags to their corresponding indices
        tag_indices = [tag_to_ix[tag] for tag in tags]

        # Append start and end tags
        tag_indices = [tag_to_ix['<S>']] + tag_indices + [tag_to_ix['</S>']]

        # Combine GloVe vectors with tag indices into tuples and append to indexed data
        indexed_sentence = list(zip(glove_vectors, tag_indices))
        indexed_data.append(indexed_sentence)

    return indexed_data

# Example usage with train_data and glove_vectors_with_case
indexed_train_data = convert_to_glove_vectors(train_data, tag_to_ix)
indexed_dev_data = convert_to_glove_vectors(dev_data, tag_to_ix)



import torch
from torch.utils.data import Dataset


class WordLabelDataset2(Dataset):
    def __init__(self, data):
        """
        Initialization method for the dataset.
        Parameters:
        data (list of lists of tuples): The dataset, where each tuple contains a word embedding and its label.
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
        Each word is now represented by a 104-dimensional embedding.
        """
        # Retrieve the sentence data at the specified index
        sentence_data = self.data[index]
        
        # Separate embeddings and labels
        embeddings, labels = zip(*sentence_data)

        # Convert list of NumPy arrays to list of tensors
        embeddings_tensor = [torch.tensor(e, dtype=torch.float32) for e in embeddings]

        # Stack the tensors to create a single tensor for the embeddings
        embeddings_tensor = torch.stack(embeddings_tensor)

        # Convert labels to a tensor from a list
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return embeddings_tensor, labels_tensor

# Create an instance of the dataset
train_dataset = WordLabelDataset2(indexed_train_data)
dev_dataset = WordLabelDataset2(indexed_dev_data)



# Use the dataset with a DataLoader
from torch.utils.data import DataLoader

# pad_embedding=glove_vectors['<PAD>']
from torch.nn.utils.rnn import pad_sequence
# Define a collate function that can handle padding if necessary
def custom_collate_fn2(batch, pad_embedding=create_extended_word_vector('<PAD>')):
    """
    Custom collate function for padding variable length sequences of embeddings
    and their corresponding label sequences using a specific glove embedding for padding.

    Args:
        batch: List of tuples with (embedding_sequence, label_sequence).
        pad_embedding: The embedding vector to use for padding.

    Returns:
        A tuple of two tensors: padded embedding sequences and corresponding padded labels.
    """
    # Separate the embedding sequences and label sequences in the batch
    sequences, label_sequences = zip(*batch)
    
    # Get the length of the longest sequence
    max_seq_len = max(len(seq) for seq in sequences)

    # Pad the embedding sequences manually
    padded_sequences = []
    for seq in sequences:
        # Create a tensor for each sequence
        seq_tensor = torch.stack([torch.tensor(e, dtype=torch.float32) for e in seq])
        # Calculate how much padding is needed
        pad_size = max_seq_len - len(seq)
        # Create a tensor of pad embeddings of appropriate size
        # print(seq_tensor.shape)
        # print(pad_size)
        # print(pad_embedding.shape)
        # print(type(pad_embedding))
        pad_embedding = torch.tensor(pad_embedding, dtype=torch.float32)
        pad_tensor = pad_embedding.unsqueeze(0).repeat(pad_size, 1)
        
        # Concatenate the sequence and pad tensors
        padded_seq = torch.cat((seq_tensor, pad_tensor), dim=0)
        # print(padded_seq.shape)
        padded_sequences.append(padded_seq)

    # Convert the labels to tensors and pad them
    label_sequences_tensor = [torch.tensor(labels, dtype=torch.long) for labels in label_sequences]
    labels_padded = pad_sequence(label_sequences_tensor, batch_first=True, padding_value=0)  # Assuming -100 is not a valid label

    # Stack the padded sequences into a single tensor
    sequences_padded = torch.stack(padded_sequences)

    return sequences_padded, labels_padded

# Define batch size for DataLoader
batch_size = 32  # You can adjust this size as needed

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn2)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn2)  # Usually, we don't shuffle the test set


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BiLSTMForNER(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(BiLSTMForNER, self).__init__()
        # Remove the embedding layer if using precomputed embeddings
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, 128)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.33)
        self.hidden2tag2 = nn.Linear(128, tagset_size)

    def forward(self, embeddings):
        # If using precomputed embeddings, no need to look up embeddings
        # x = self.embedding(sentence)
        x, _ = self.bilstm(embeddings)
        x = self.hidden2tag(x)
        x = self.elu(x)
        x = self.dropout(x)
        tag_scores = self.hidden2tag2(x)
        return tag_scores

# Since you're not using an embedding layer, the vocab_size parameter is not needed
embedding_dim = 104  # Adjusted to match the actual size of your precomputed embeddings
hidden_dim = 256  # LSTM hidden dimensions
tagset_size = len(tag_to_ix)  # Number of tags in your tagset

model = BiLSTMForNER(embedding_dim, hidden_dim, tagset_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Define the loss function and optimizer with the specified settings

# weights = torch.tensor([0,0.5,0.5,1,1,0.1,1,1,1,1,1,1]) 
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Using SGD as specified with a learning rate of 0.1

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.1, verbose=True)


model.load_state_dict(torch.load('blstm2.pt'))


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

# Now you have the true and predicted tags sentence-wise, 
            



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

output_file_path = 'dev2.out'
convert_and_write_to_file(word_pred_tag_pairs, output_file_path)

##TEST DATA   


def transform_data(data):
    transformed_data = []
    for sublist in data:
        transformed_sublist = [(item, 'O') for item in sublist]
        transformed_data.append(transformed_sublist)
    return transformed_data


test_data = transform_data(test_data_raw_sentences)

indexed_test_data=convert_to_indices(test_data, word_to_ix, tag_to_ix)

indexed_test_data = convert_to_glove_vectors(test_data, tag_to_ix)


test_dataset = WordLabelDataset2(indexed_test_data)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn2)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

for sentence, preds in zip(test_data, sentence_level_pred_tags):
    # Ensure that the number of words and predictions match
    
    if len(sentence) != len(preds):
        print(sentence)
        print(preds)
        print(len(sentence),len(preds))
        print("Length mismatch between words and predicted tags")
        
    
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

output_file_path = 'test2.out'
convert_and_write_to_file(word_pred_tag_pairs, output_file_path)







