import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold



if __name__ == '__main__':
    df = pd.read_csv('input/final_data1.csv')
    df['fn'] = df['fn'].str.replace('audio_files/','')
    df['kfold'] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)

    classes = np.unique(df["label"].values.tolist())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    #print(class_to_idx)
    idx_to_class = {val:key for key ,val in class_to_idx.items()}
    df['label'].replace(class_to_idx, inplace=True)
    #df.head()



    y = df.label.values

    kf = StratifiedKFold(n_splits=5)

    for fold, (train_idx,val_idx) in enumerate(kf.split(X=df,y=y)):
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv('input/stratified_train_5_fold_fnl1.csv',index=False)    