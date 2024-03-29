# GIZ-NLP-Agricultural-Keyword-Spotter-Competition

- This is an audio classification problem, you can check the [competition](https://zindi.africa/competitions/giz-nlp-agricultural-keyword-spotter/leaderboard)  website for more details

- The function ***wav_to_img*** in ```utils.py``` is used to convert the audio files ".wav" into images by extracting the **MelSpectrogram**, **STFT**, and **MFCC** and concatenating the three alongside the channel axis
so that the final image size is (224,224,3)

- The Notebook ```GIZ.ipynb``` contains the required steps to train the model
- The ```train_fold.py``` implements **Cross-Validation**

- There is also an implementation of **mixup** and **cutmix** augmentations in case you want to test them

- [My Profile](https://zindi.africa/users/Anas_Hasni) on Zindi Platform
