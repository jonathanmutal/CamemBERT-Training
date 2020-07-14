#!/usr/bin/env python
# coding: utf-8

# In[1]:
#domains = ['abdominal', 'anus', 'checkup', 'chest', 'accueil', 'colique', 'covid', 'dermato', 'examen', 'headache', 'orl', 'suivi', 'traumatologie', 'urines']
domains = ['abdominal', 'anus', 'checkup', 'chest', 'accueil', 'covid', 'dermato', 'examen', 'headache', 'orl', 'suivi', 'traumatologie', 'urines']
# In[2]:


PATH_compact = lambda domain: 'babeldr_652eeeff_{0}_plain_2.txt'.format(domain)
PATH_ellipsis = lambda domain: 'babeldr_652eeeff_{0}_plain_2.txt'.format(domain)


# # creating corpus

# In[3]:


import pandas as pd
import re


# In[4]:


def prepare_training(domain):
    #### ELLIPTICAL TRAINING DATA ####
#    ellipsisDf = pd.read_csv(PATH_ellipsis(domain), sep=';', names=['canonical', 'variation'])
#    ellipsisDf = ellipsisDf.fillna('')
    # new_variations = []
    # new_canonicals = []
    # first case we don't have any context
 #   ellipsis = list(ellipsisDf.iterrows())
 #   index, row = ellipsis[0]
 #   new_variations.append(row['variation'])
 #   new_canonicals.append(row['canonical'])
    # index, row.
#    for index, row in ellipsis[1:]:
#        elliptical_variation = re.sub(' \?', '', ellipsis[index-1][1]['canonical']) + ' ' + row['variation']
#        new_variations.append(elliptical_variation)
#        new_canonicals.append(row['canonical'])

   # assert(len(new_variations) == len(new_canonicals))

   # ellipticalTraining = pd.DataFrame({'canonical': new_canonicals, 'variation': new_variations})

    # normal training data
    trainingDf = pd.read_csv(PATH_compact(domain), sep=';', names=['canonical', 'variation'])
    trainingDf = trainingDf[~trainingDf.variation.isna() & ~trainingDf.canonical.isna()]
    trainingDf = trainingDf.drop_duplicates()
    
    #return trainingDf.append(ellipticalTraining, ignore_index = True)
    return trainingDf


# In[5]:


trainingData = dict()
for domain in domains:
    print('loading training data for', domain)
    trainingData[domain] = prepare_training(domain)
    trainingData[domain].to_csv('{0}_all.csv'.format(domain), index=False)


# In[6]:


print("ready training data!")


# In[7]:


print([len(trainingData[dom]) for dom in trainingData])


# ## Training models

# In[8]:

# In[9]:


import tensorflow as tf
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification, CamembertModel
# We'll borrow the `pad_sequences` utility function to do this.
from keras.preprocessing.sequence import pad_sequences
# Use train_test_split to split our data into train and validation sets for
# training
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# adam optimizer
from transformers import AdamW
# decay
from transformers import get_linear_schedule_with_warmup
# to get seed
import random
## to graphic the perfomance.
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


if torch.cuda.is_available():
    device = torch.device("cuda", 0)

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
    print(device)


# In[11]:


import os


# In[12]:


for domain in domains:
    if not os.path.isdir(domain):
        os.mkdir(domain)
print('path created! :)')


# In[13]:


import re
preprocess = lambda sent: re.sub(r'\s+', ' ', re.sub('-', ' ', re.sub("'", "' ",  sent.strip())))
preprocess_canonical = lambda sent: re.sub(r'\s+', ' ', sent.strip())


# In[14]:


import pickle
def save_model(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# In[15]:


import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[16]:


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[17]:


for domain in domains:
    variations = trainingData[domain].variation.apply(preprocess).tolist()
    canonicals = trainingData[domain].canonical.apply(preprocess_canonical).tolist()
    unique_canonicals = list(set(canonicals))
    canonicals += unique_canonicals
    variations += [preprocess(re.sub('\?', ' ', can)) for can in unique_canonicals]
    assert(len(canonicals) == len(variations))
    ## making index2canonical and canonical2index
    index2canonical = canonicals
    canonical2index = dict([(canonical, i) for (i, canonical) in enumerate(index2canonical)])
    # save model for index2canonical and canonical2index
    save_model(index2canonical, '{0}/index2canonical'.format(domain))
    save_model(canonical2index, '{0}/canonical2index'.format(domain))
    print('index2canonical and canonical2index saved in {0}'.format(domain))
    ## map categorical variable to numbers
    labels = [canonical2index[sent] for sent in canonicals]
    print('training data ready!!!....')
    X = variations
    y = labels
    assert(len(X) == len(y) )
    print('begin to pre-processing (tokenization)')
    cambertTokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    print('length X:', len(X))
    input_ids = []

    # For every sentence...
    for sent in X:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = cambertTokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # Print sentence 0, now as a list of IDs.
    print(domain)
    print('Original: ', X[-1])
    print('Token IDs:', input_ids[-1])
    MAX_LEN = max([len(sen) for sen in input_ids])
    print('Max sentence length: ', MAX_LEN)

    ##### PADDING SENTENCES/TRUNCATING
    print('\nPadding/truncating all sentences to {} values...'.format(MAX_LEN))

    print('\nPadding token: "{:}", ID: {:}'.format(cambertTokenizer.pad_token, cambertTokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")

    print('\nDone.')
    
    
    print('creating attention mask')
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, y, 
                                                            random_state=2018, test_size=0.1)

    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, y,
                                             random_state=2018, test_size=0.1)
    
    
    ### train to torch tensor
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    
    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    
    ##### loading the model and beging the training
    print('loading the model...')
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = len(index2canonical), # The number of output labels for multi-class classification.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    # train the model on GPU
    model.cuda()
    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
             lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
             eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    )
    
    # Number of training epochs 
    epochs = 30 

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)


    # seed value to make it reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    print('begin the actual training!!!!!')
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        if (avg_train_loss <= 0.01):
            print('I learnt before {} epochs :)'.format(epochs))
            break
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

    print("")
    print("Running Validation...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")
    
    
    
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o')

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig('{0}/fig_training'.format(domain))
    
    print('saving model! yaaayyyyy', domain)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = '{0}/CamemBERT/'.format(domain)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    cambertTokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    # let's free the cpu and gpu
        
    del cambertTokenizer
    del model
    torch.cuda.empty_cache()


# In[ ]:


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
cambertTokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
# let's free the cpu and gpu

del cambertTokenizer
del model
torch.cuda.empty_cache()


# In[ ]:


# import pandas as pd

# # Report the number of sentences.
# print('Number of test sentences: {:,}\n'.format(len(sentences)))

# # Tokenize all of the sentences and map the tokens to thier word IDs.
# input_ids = []

# # For every sentence...
# for sent in sentences:
#     # `encode` will:
#     #   (1) Tokenize the sentence.
#     #   (2) Prepend the `[CLS]` token to the start.
#     #   (3) Append the `[SEP]` token to the end.
#     #   (4) Map tokens to their IDs.
#     encoded_sent = cambertTokenizer.encode(
#                         sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                    )
    
#     input_ids.append(encoded_sent)

# # Pad our input tokens
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
#                           dtype="long", truncating="post", padding="post")

# # Create attention masks
# attention_masks = []

# # Create a mask of 1s for each token followed by 0s for padding
# for seq in input_ids:
#     seq_mask = [float(i>0) for i in seq]
#     attention_masks.append(seq_mask)

# # Convert to tensors.
# prediction_inputs = torch.tensor(input_ids)
# prediction_masks = torch.tensor(attention_masks)
# prediction_labels = torch.tensor(labels)

# # Set the batch size.  
# batch_size = 32  

# # Create the DataLoader.
# prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
# prediction_sampler = SequentialSampler(prediction_data)
# prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# In[ ]:


# # Prediction on test set

# print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# # Put model in evaluation mode
# model.eval()

# # Tracking variables 
# predictions , true_labels = [], []

# # Predict 
# for batch in prediction_dataloader:
#   # Add batch to GPU
#     batch = tuple(t.to(device) for t in batch)
  
#   # Unpack the inputs from our dataloader
#     b_input_ids, b_input_mask, b_labels = batch
  
#   # Telling the model not to compute or store gradients, saving memory and 
#   # speeding up prediction
#     with torch.no_grad():
#       # Forward pass, calculate logit predictions
#         outputs = model(b_input_ids, token_type_ids=None, 
#                       attention_mask=b_input_mask)

#     logits = outputs[0]

#   # Move logits and labels to CPU
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()
  
#   # Store predictions and true labels
#     predictions.append(logits)
#     true_labels.append(label_ids)

# print('    DONE.')


# In[ ]:


# from sklearn.metrics import matthews_corrcoef, classification_report

# matthews_set = []

# # Evaluate each test batch using Matthew's correlation coefficient
# print('Calculating Matthews Corr. Coef. for each batch...')

# # For each input batch...
# for i in range(len(true_labels)):
  
#   # The predictions for this batch are a 2-column ndarray (one column for "0" 
#   # and one column for "1"). Pick the label with the highest value and turn this
#   # in to a list of 0s and 1s.
#     pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  
#   # Calculate and store the coef for this batch.  
#     matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
#     matthews_set.append(matthews)


# In[ ]:


# a = list(filter(lambda prediction: prediction[0] > 0, zip(flat_true_labels, flat_predictions)))


# In[ ]:


# true_labels, predictions = list(zip(*a))


# In[ ]:


# print(classification_report(true_labels, predictions))


# In[ ]:


# # Combine the predictions for each batch into a single list of 0s and 1s.
# flat_predictions = [item for sublist in predictions for item in sublist]
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# # Combine the correct labels for each batch into a single list.
# flat_true_labels = [item for sublist in true_labels for item in sublist]

# # Calculate the MCC
# mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

# print('MCC: %.3f' % mcc)
f.close()
