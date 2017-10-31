##################
nFeats = 256
nStack = 8
nModules = 1

LR = 2.5e-4

inputRes = 256
outputRes = 64

meanVal = 0.5
partnum = 14
nPool = 4

LRNKer = 1
####################
#train_setting
####################
gpus = "0"

train_path = "/home/dan/test_img/train.rec"
val_path = "/home/dan/test_img/train.rec"
batch_size = 1
data_shape=(3,256,256)

mean_pixels = [0,0,0]