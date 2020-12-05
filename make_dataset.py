import pandas as pd
import os
from basic_tokenizers import ByteTextTokenizer
import time
from tqdm import tqdm


def read_lines(path):
    """
    Tokenizes a text file.
    """
    assert os.path.exists(path)
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line)  # + ['<eos>'])
    return lines


def convert_data_to_seq_length(df, seq_length=64000):
    """
    Take a dataframe text data and convert it to a dataframe with the same columns where
    every data sample is of numericalized token length of ~seq_length
    (less than but closest to the value given)
    :param df: a pandas datafram with columns [text, lens]
    :param seq_length: the numericalized token sequence length to split the data into
    :return: the new dataframe with split data samples
    """
    sum_len = 0
    concat_text = ''
    result = pd.DataFrame(columns=['text', 'lens'])
    for i in tqdm(range(len(df)), desc="splitting data", total=len(df)):
        if sum_len + df['lens'].iloc[i] < seq_length:
            sum_len += df['lens'].iloc[i]
            concat_text += df['text'].iloc[i]
        else:
            result = result.append(
                {
                    'text': concat_text,
                    'lens': sum_len,
                },
                ignore_index=True)
            sum_len = df['lens'].iloc[i]
            concat_text = df['text'].iloc[i]
    return result


def read_and_prepare_data(data_path, seq_length=0):
    """
    Read the data from file, and prepare the dataframe.
    This does not include splitting into train and validation sets.
    :param data_path: relative path to the raw data
    :param seq_length: sequence length to split data into, default is don't change data sample length
    :return: the dataframe after preparations
    """
    print("Reading data from path...")
    # Read the data from file
    enwik8 = read_lines(data_path)
    df = pd.DataFrame({'text': enwik8})
    print("Done!")
    # By default we won't change the data sample length
    if seq_length != 0:
        print("Sequence length has been added, "
              "splitting data to samples with sequence length " + str(seq_length))

        time.sleep(0.5)  # this is so the printing of the progress bar is not weird
        # Initialize the BTT
        btt = ByteTextTokenizer(is_lm=True, add_bos=True, add_eos=True)
        tqdm.pandas(desc="preparing data for split")
        # Modify dataset for training
        df['lens'] = df['text'].progress_map(lambda x: len(btt(x)))

        # Convert data samples according to sequence length
        df = convert_data_to_seq_length(df, seq_length)
        print()
        print("Done!")
    else:
        df['lens'] = df['text'].map(lambda x: len(x))

    df['lens_cum_sum'] = df.lens.cumsum()

    return df
