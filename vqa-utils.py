import os 
import numpy as np
import random 
import json
from datetime import datetime

import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# HYPER PARAMETERS
ROOT = '../input/ann-and-dl-vqa/dataset_vqa'
IMG_DIR = os.path.join(ROOT, 'train')
TST_DIR = os.path.join(ROOT, 'test')

MAX_NUM_WORDS = 71
SENTENCE_LENGTH = 42
NUM_ANS = 13
N_SAMPLES = 259492
N_TEST = 3000
N_TRAIN = int(np.ceil(N_SAMPLES*0.8))
IMG_WIDTH = 480
IMG_HEIGHT = 320



def set_seed(SEED):
    """
    set the seeds for random, numpy and tensorflow packages
    """
    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    return 


def check_gpu():
    """
    check the availability of the GPU
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)
    return

def fit_param(batch_size):
    """
    the function computes and returns the number of steps per epoch for both 
    training and validation. 
    input param: batch size
    
    output param:
    e_steps = number of steps per epoch
    v_steps = number of validation steps
    max_que = number of batches that will be loaded in the background during training 
                (for training time optimization)
    """
    max_que = 1200 // batch_size 
    e_steps = N_TRAIN//batch_size
    v_steps = (N_SAMPLES-N_TRAIN)//batch_size
    return max_que, e_steps, v_steps

def create_tokens():
    """
    this function will read the data from memory and create a tokenizer object (already fitted on the text)
    """
    
    #read from data
    with open(os.path.join(ROOT, 'train_data.json'), 'r') as f:
          data = json.load(f)
    f.close()
    tot_txt = []
    for sample in data['questions']:
        tot_txt.append(sample['question'])

    # create and fit the tokenizer
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(tot_txt)
    return tokenizer

def get_dataset(sentence_tokenizer, batch_size=64,is_training=True):
    """
    this function create trainig/validation datasets. 
    It comprehends all the needed function for preparing the data. 
    
    input param: 
    sentence_tokenizer: tokenizer to process the questions
    batch_size: for divide the datasets in batches
    is_training: Boolean, for distinguish between train and valid
    
    output: 
    dataset
    """
    def cast_array(data):
        """
        utility function cast all the inputs into float32 type.
        """
        return tf.dtypes.cast(data, dtype=tf.float32)

    def tokenize(sentences):
        """
        converts a text to a sequence of words using the tokenizer given as input.
        """
        return sentence_tokenizer.texts_to_sequences(sentences)

    def question_preprocessing(qst):
        """
        utility function for processing the questions. add 'begin_of_sentence' and 
        'end_of_sentence' special character, tokenize and pad to maximum question length 
        (computed in advance and treated as hyperparameter)
        """
        
        # add begin and end of sentence tokens to questions
        temp = []
        for question in qst: 
            question = question + ' <eos>'
            temp.append('<sos> ' + question)
        
        # tokenize the question 
        qst_tokenized = tokenize(qst) 
        
        # return the padding questions
        return pad_sequences(qst_tokenized, maxlen=SENTENCE_LENGTH, padding='post')

    def one_hot_setup():
        """
        utility function used for setting up the one_hot_encoder for the custum classes of the problem. 
        uses sklearn.preprocessing methos LabelEncoder, OneHotEncoder
        """
        classes = ['0','1','10','2','3','4','5','6','7','8','9','no','yes']

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse = False,categories='auto') #categories='auto' to avoid the warning
        
        # fit the classes as integers
        integer_encoder_ = label_encoder.fit(classes)
        integer_encoded = integer_encoder_.transform(classes)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) #reshaping
        
        # fit to one hot encoding 
        onehot_encoder_ = onehot_encoder.fit(integer_encoded)
        return integer_encoder_,onehot_encoder_

    def onehot_data(data):
        """
        utility function that given the answers (classes) converts them to one hot encoding
        using the procedures of one_hot_setup()
        """
        integer_encoder_,onehot_encoder_ = one_hot_setup()
        
        # convert the data to np.array, this allow to standardize the data
        values = np.array(data)
        # transform to integer classes
        values = integer_encoder_.transform(values)
        values = values.reshape(len(values), 1)
        # return the one hot values
        return onehot_encoder_.transform(values)

    def answer_preprocessing(ans):
        """
        utility function for answers preprocessing. 
        It transform the data in one hot encoded vectors and cast them as float32.
        """
        ans_out = onehot_data(ans)
        return cast_array(ans_out)

    def create_dataset(qst_input, img_filenames, ans_input,batch_size=64):
        """
        utility function for the creation of the datasets.
        """
        import tensorflow
        #use the from_tensor_slices, we give as input the preprocessed questions, preprocessed answers and the images filepath
        dataset = tf.data.Dataset.from_tensor_slices((qst_input,img_filenames,ans_input))
        # shuffle the dataset
        dataset = dataset.shuffle(N_TRAIN).repeat()
        # use the parse_function to load the images (given the filepath) and num_parallel_calls for parallelize the process
        dataset = dataset.map(parse_function, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        # divide in batches
        dataset = dataset.batch(batch_size)
        # further optimization, prefetch in memory the max number of batches the memory can sustain
        dataset = dataset.prefetch(buffer_size=tensorflow.data.experimental.AUTOTUNE)
        return dataset

    def read_data(is_training=True):
        """
        function that read the data from the json file and do some preprocessing to the loaded data.
        """
        with open(os.path.join(ROOT, 'train_data.json'), 'r') as f:
            data = json.load(f)
        f.close()

        qst = []
        ans = []
        img = []

        # shuffle the {question, file_path, answer} triples
        for i in range(len(data['questions'])-1, 0, -1): 
            j = random.randint(0, i + 1)   
            data['questions'][i], data['questions'][j] = data['questions'][j], data['questions'][i]  
    
        # train validation split
        if is_training:
            for question in data['questions'][:N_TRAIN]:
                ans.append(question['answer'])
                qst.append(question['question'])
                img.append(question['image_filename'])
        else:
            for question in data['questions'][N_TRAIN:]:
                ans.append(question['answer'])
                qst.append(question['question'])
                img.append(question['image_filename'])

        # preprocess question and anwer, load the filepath of the images
        qst_input = question_preprocessing(qst)
        ans_input = answer_preprocessing(ans)
        img_filenames = get_full_path(img)
        return qst_input, img_filenames, ans_input

    def get_full_path(image_filename):
        """
        utility function that given a list of images returns the complete file path associated to the image
        """
        path = []
        for file in image_filename:
            path.append(os.path.join(IMG_DIR,file))
        return path 

    def parse_function(qst,img_filenames,ans):
        """
        utility function that allow to load and process the images. (using tf.io methods)
        """
        img = tf.io.read_file(img_filenames)
        img = decode_img(img)
        return {'input_qst': qst, 'input_img': img}, ans

    def decode_img(img):
        """
        utility function that decode the png images, converts them into float32 and do the proper resizing.
        """
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        return img

    if is_training:
        # train dataset
        qst_input, img_filenames, ans_input = read_data()
        return create_dataset(qst_input, img_filenames, ans_input, batch_size)
    else:
        # validation dataset
        qst_input, img_filenames, ans_input = read_data(is_training=False)
        return create_dataset(qst_input, img_filenames, ans_input, batch_size)

    
## MODEL ##
def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


def get_class_weights():
    """
    function that uses sklearn.class_weight method to compute the weights of each class in the train set. 
    the output should be used in the mode.fit() method for taking into account the class inbalance.
    """
    with open(os.path.join(ROOT, 'train_data.json'), 'r') as f:
        data = json.load(f)
    f.close()
    ans = []
    for question in data['questions'][:N_TRAIN]:
        ans.append(question['answer'])
    
    return class_weight.compute_class_weight('balanced',np.unique(ans),ans)

def callb():
    """
    utility function for the callbacks. 
    In this case I only used two, EarlyStopping and ModelCheckpoint to save the model.
    """
    call = []
    call.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=2,
        restore_best_weights=True))
    call.append(tf.keras.callbacks.ModelCheckpoint(
        './best_model.h5', 
        save_weights_only=True, 
        save_best_only=True, 
        mode='min'))
    return call
            
        
def test_dataset(sentence_tokenizer, test_bs=100):
    """
    fuction for the creation of the test dataset. 
    It uses all the utility funciton used for the training/validation dataset creation with 
    the minor change of not treat the answers. 
    """
    def cast_array(data):
        return tf.dtypes.cast(data, dtype=tf.float32)

    def tokenize(sentences, MAX_NUM):
        return sentence_tokenizer.texts_to_sequences(sentences) # converts a text to a sequence of words

    def question_preprocessing(qst):
        temp = []
        for question in qst: 
            question = question + ' <eos>'
            temp.append('<sos> ' + question)
        qst_tokenized = tokenize(qst,MAX_NUM_WORDS) # converts a text to a sequence of words (or tokens).
        return pad_sequences(qst_tokenized, maxlen=42, padding='post')

    def answer_preprocessing(ans):
        ans_out = onehot_data(ans)
        return cast_array(ans_out)

    def get_full_path(image_filename):
        path = []
        for file in image_filename:
            path.append(os.path.join(TST_DIR,file))
        return path 

    def parse_test_function(qst,img_filenames):
        img = tf.io.read_file(img_filenames)
        img = decode_img(img)
        return {'input_qst': qst, 'input_img': img}

    def decode_img(img):
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        return img

    def read_test_data():
        with open(os.path.join(ROOT, 'test_data.json'), 'r') as f:
            data = json.load(f)
        f.close()

        qst = []
        img = []
        for question in data['questions']:
            qst.append(question['question'])
            img.append(question['image_filename'])

        qst_input = question_preprocessing(qst)
        img_filenames = get_full_path(img)
        return qst_input, img_filenames
    
    def create_test_dataset(qst_input, img_filenames,test_bs=100):
        import tensorflow
        dataset = tf.data.Dataset.from_tensor_slices((qst_input,img_filenames))
        dataset = dataset.map(parse_test_function, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
        dataset = dataset.batch(test_bs)
        dataset = dataset.prefetch(buffer_size=tensorflow.data.experimental.AUTOTUNE)
        return dataset

    qst_input, img_filenames = read_test_data()
    test_ds = create_test_dataset(qst_input, img_filenames)
    return test_ds

def make_prediction(model,test_ds):
    """
    function to make the prediction. 
    input: 
    - trained model 
    - test dataset
    
    it will make the prediction, create the results dictionary and the csv for submission 
    """
    def create_csv(results, results_dir='./'):
        """
        function to create the csv for submission
        """
        csv_fname = 'results_'
        csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

        with open(os.path.join(results_dir, csv_fname), 'w') as f:
            f.write('Id,Category\n')
            for key, value in results.items():
                f.write(str(key) + ',' + str(value) + '\n')
            
    res = {}
    pred = [0]*N_TEST

    pred = model.predict(test_ds, steps=30, verbose=1)
    for i in range(N_TEST):
        res[i] = np.argmax(pred[i])
    create_csv(res)
    return res




