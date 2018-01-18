#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
X = train.as_matrix()

NFOLDS = 6
kfold = KFold(n_splits=NFOLDS, shuffle=False, random_state=None)

for train_index,val_index in kfold.split(X):
    print("Train Index:",train_index,",Val Index:",val_index)
    X_train,X_valid = X[train_index],X[val_index]
    y_train,y_valid = targets[train_index],targets[val_index]
    print (X_train.shape)
    print (X_valid.shape)
    print (y_train.shape)
    print (y_valid.shape)
    # fit network
    model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=512, validation_data=(np.array(X_valid), np.array(y_valid)),
            verbose=2, callbacks=[stop_callback, checkpoint], shuffle=False)

    model.load_weights('best_wt_recruit_new.hdf5')

    # get validation score
    pred = np.exp(model.predict(np.array(valid)))
    score = rmsle(np.exp(np.array(y_valid)), pred)

    print('score:',score)
    del X_train,X_valid,y_train,y_valid
