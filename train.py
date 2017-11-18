import time

import configparser
from dataGenerator.datagen import DataGenerator
import tensorflow as tf
from four_stack.Hourglass import HourglassModel
from train_class import train_class
from hg_models.hgattention import HGattention
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def process_config(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'log':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Training setting':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

def process_network(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params
# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
# 数据集类型

# 模式：训练、测试
tf.app.flags.DEFINE_string('mode',
                           'train',
                           'train or eval.')
# resume
tf.app.flags.DEFINE_string('resume',
                           '',
                           'restore model path')

if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('./config/config_mini.cfg')
    network_params = process_network("./config/hourglass_mini.cfg")
    #network_params = process_network("./config/hgattention.cfg")

    show_step = params["show_step"]
    train_data = DataGenerator(imgdir=params['train_img_path'], nstack= network_params['nstack'],label_dir=params['label_dir'],
                               out_record=params['train_record'],num_txt=params['train_num_txt'],
                               batch_size=params['batch_size'], name="train_mini", is_aug=False,isvalid=False,scale=
                               params['scale'], refine_num = 10000)
    valid_data = DataGenerator(imgdir=params['valid_img_path'], label_dir=params['valid_label'],
                               out_record=params['valid_record'],num_txt=params['valid_num_txt'],
                               batch_size=params['batch_size'], name="valid_mini", is_aug=False,isvalid=True,scale=
                               params['scale'],refine_num=3000)

    img, hm = train_data.getData()


    model = HourglassModel(nFeat=network_params['nfeats'], nStack=network_params['nstack'],
                           nModules=network_params['nmodules'],outputDim=network_params['partnum'])._graph_hourglass

    # model = HGattention(nFeat=network_params['nfeats'], nStack=network_params['nstack'],
    #                     nModules=network_params['nmodules'],outputDim=network_params['partnum'],
    #                     npool=network_params['npool'], lrnker=network_params['lrnker']).createModel
    if FLAGS.mode == 'train':
        trainer = train_class(model=model, nstack=network_params['nstack'], batch_size=params['batch_size'],
                              learn_rate=params['lear_rate'], decay=params['decay'],
                              decay_step=params['decay_step'],logdir_train=params['train_log_dir'],
                              logdir_valid=params['valid_log_dir'],name='tiny_hourglass',
                              train_record=train_data,valid_record=valid_data,
                              save_model_dir=params['model_save_path'],
                              resume=params['resume'],#/media/bnrc2/_backup/golf/model/tiny_hourglass_21
                              gpu=params['gpus'],partnum=network_params['partnum'],
                              val_label=params['valid_label'],human_decay=params['human_decay']
                     )
        trainer.generateModel()
        trainer.training_init(nEpochs=params['nepochs'],valStep=params['val_step'],showStep=show_step )


