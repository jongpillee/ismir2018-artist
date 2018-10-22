import pickle
import numpy as np
from collections import defaultdict

def load_label(num_artist, train_size=15, valid_size=18, test_size=20):

    file_list = pickle.load(open('./data/msd-artist-list.pkl','rb'))
    file_list = file_list[:num_artist*20]
    np.random.shuffle(file_list)

    song_list = []
    artist_list = []
    song_id_list = []

    for line in file_list:
        line = line.split('/')
        song_id_list.append(line[0])
        artist_list.append(line[1])
        song_list.append(line[2])

    # split train/valid/test
    train_list = []
    valid_list = []
    test_list = []
    y_train = []
    y_valid = []
    y_test = []

    for iter in range(0, len(file_list)):
        # train list
        if int(song_list[iter]) in range(1, train_size+1):
            train_list.append(song_id_list[iter])
            y_train.append(artist_list[iter])
        # valid list
        elif int(song_list[iter]) in range(train_size+1, valid_size+1):
            valid_list.append(song_id_list[iter])
            y_valid.append(artist_list[iter])
        # test list
        elif int(song_list[iter]) in range(valid_size+1, test_size+1):
            test_list.append(song_id_list[iter])
            y_test.append(artist_list[iter])

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    # artist_dict
    dicts = dict(zip(train_list, y_train))
    d = defaultdict(list)
    for k,v in dicts.items():
        d[v].append(k)
    artist_train = dict(d)
	
    dicts = dict(zip(valid_list, y_valid))
    d = defaultdict(list)
    for k,v in dicts.items():
        d[v].append(k)
    artist_valid = dict(d)

    dicts = dict(zip(test_list, y_test))
    d = defaultdict(list)
    for k,v in dicts.items():
        d[v].append(k)
    artist_test = dict(d)

    return train_list, valid_list, test_list, y_train, y_valid, y_test, artist_train, artist_valid, artist_test
