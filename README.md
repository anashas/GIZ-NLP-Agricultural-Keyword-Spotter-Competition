# GIZ-NLP-Agricultural-Keyword-Spotter-Competition

This is an audio classification problem, you can check the website for more details: https://zindi.africa/competitions/giz-nlp-agricultural-keyword-spotter/leaderboard

The function wav_to_img in utils.py is used to convert the audio files ".wav" into images by extracting the MelSpectrogram, STFT and MFCC and concatenating the three alonside the channel axis
so that the final image size is (224,224,3).

The Notebook GIZ contains the required steps to train the model.

My Profile on Zindi Platform: https://zindi.africa/users/data_scientist 
