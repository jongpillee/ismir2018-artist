# ismir2018-artist

***An implementation of "Representation Learning of Music using Artist labels, ISMIR, 2018"***

The repository will be updated soon!!.

----------------------------------

## Requirements

- tensorflow-gpu==1.4.0
- keras==2.0.8

These requirements can be easily installed by:
	`pip install -r requirements.txt`

## Scripts

- __data_generator.py__: The base script that contains batch data generator and valid set loader.
- __mp3s_to_mel.py__: Convert audio to mel-spectrogram.
- __model.py__: Basic and siamese models.
- __train.py__: Module for training the models.
- __encoding_cnn.py__: Contains script for extracting feature vector given trained model weight and audio list.

## Usage

Here are examples of how to run the code. (To run 1. and 2., you need MSD audio files and its related metadata from [msd-artist-split](https://github.com/jiyoungpark527/msd-artist-split), [MSD_split](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split))
1. `python mp3s_to_mel.py` 
2. `python train.py siamese --num-sing 1000 --margin 0.4 --N-negs 4`
3. `python encoding_cnn.py siamese ./models/model_siamese_1000_4_0.40/weights.59-12.32.h5 --num-sing 1000`

----------------------------------

## Reference

[1] [Representation Learning of Music Using Artist Labels](http://ismir2018.ircam.fr/doc/pdfs/168_Paper.pdf), Jiyoung Park*, Jongpil Lee*, Jangyeon Park, Jung-Woo Ha and Juhan Nam
Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), 2018
