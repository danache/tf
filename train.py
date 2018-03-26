import configparser
from dataGenerator.datagen_Multigauss import DataGenerator
from train_class import train_class
from four_stack.ian_hourglass import hourglassnet
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def process_config(conf_file):
    '''
    从config文件中读取配置
    :param conf_file:
    :return:
    '''
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

if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('./config/config_2.cfg')
    network_params = process_network("./config/hourglass.cfg")
    #network_params = process_network("./config/hgattention.cfg")

    show_step = params["show_step"]

    train_data = DataGenerator(imgdir=params["train_img_path"], txt="/media/bnrc2/_backup/dataset/aiclg/train_with_no_see.txt",
                               batch_size=params['batch_size'], is_aug=True,
                               joints_name=
                               params["joints"])  # , refine_num = 10000)
    # valid_data = DataGenerator(imgdir=params["valid_img_path"], txt="/media/bnrc2/_backup/dataset/aiclg/valid.txt",
    #                            batch_size=params['batch_size'], is_aug=False,
    #                            joints_name=
    #                            params["joints"], isTraing=False)


    model = hourglassnet(stacks=network_params['nstack'])


    trainer = train_class(model=model, nstack=network_params['nstack'], batch_size=params['batch_size'],
                          learn_rate=params['lear_rate'], decay=params['decay'],
                          decay_step=params['decay_step'],logdir_train=params['train_log_dir'],
                          logdir_valid=params['valid_log_dir'],name=params["model_name"],
                          train_record=train_data,#valid_record=valid_data,
                          save_model_dir=params['model_save_path'],
                          resume=params['resume'],#/media/bnrc2/_backup/golf/model/tiny_hourglass_21
                          gpu=params['gpus'],partnum=network_params['partnum'],train_label=params['label_dir'],
                          val_label=params['valid_label'],human_decay=params['human_decay'],beginepoch=7,
                 )
    trainer.generateModel()
    ##初始化，开始训练
    trainer.training_init(nEpochs=params['nepochs'],valStep=params['val_step'],showStep=show_step )


