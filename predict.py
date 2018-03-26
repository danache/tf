import sys
sys.path.append(sys.path[0])
del sys.path[0]
import configparser
import tensorflow as tf
from dataGenerator.test_datagen import DataGenerator
from predict_class import test_class
import os
from four_stack.ian_hourglass import hourglassnet

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def parse_args():
    parser = argparse.ArgumentParser(description='hourglass test code')

    parser.add_argument('--img', dest='img_path', type=str,
                        default='/media/bnrc2/_backup/dataset/crop/byd/',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--js', dest='json_path', type=str,
                        default='/media/bnrc2/_backup/dataset/crop/byd_c.json',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--m', dest='model_path', type=str,
                        default="/media/bnrc2/_backup/models/hg/hg2_23",
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--save', dest='save_path', type=str,
                        default='/media/bnrc2/_backup/dataset/325/',
                        help='run demo with images, use comma to seperate multiple images')
    args = parser.parse_args()
    return args


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

    args = parse_args()
    img_path = args.img_path
    json_path = args.json_path
    model_path = args.model_path
    save_path = args.save_path

    model = hourglassnet(stacks=network_params['nstack'])

    valid_data = DataGenerator(json=json_path,img_path=img_path, resize=256,normalize=True)

    test = test_class(model=model, nstack=network_params['nstack'],

                              resume=model_path,
                              gpu=[0],partnum=network_params['partnum'],dategen=valid_data,save_dir=save_path
                             )

    test.generateModel()
    test.test_init()
    test.pred()
    #test.test("/home/bnrc2/data/")