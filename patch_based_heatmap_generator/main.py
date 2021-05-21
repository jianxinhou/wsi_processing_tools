'''
Author: jianxinhou
Date: 2021-04-01 21:06:00
LastEditTime: 2021-05-19 19:32:18
LastEditors: Please set LastEditors
Description: 这是一个包含如何使用PatchBasedHeatmapGenerator的示例
FilePath: /patch_based_heatmap_generator/main.py
'''

import os
import argparse
import h5py
import numpy as np
from PIL import Image
from core.PatchBasedHeatmapGenerator import PatchBasedHeatmapGenerator

def main(h5_dir, wsi_dir, thumbnail_dir = '.', heatmap_dir = '.'):
    '''
    @description: 主函数
    '''    
    assert(True == os.path.isdir(h5_dir))
    assert(True == os.path.isdir(wsi_dir))
    # 列出wsi目录中的所有文件
    all_wsi = os.listdir(wsi_dir)
    # 开始生成热图
    wsi_num = len(all_wsi)
    print('共{}张WSI，开始生成'.format(wsi_num))
    for index, wsi in enumerate(all_wsi, start = 1):
        # 生成wsi的文件名，和不带后缀的文件名
        wsi_name, ext = os.path.splitext(wsi)
        wsi_path = os.path.join(wsi_dir, wsi)
        # 生成h5数据库的文件名
        h5_path = os.path.join(h5_dir, "{}.h5".format(wsi_name))
        # 从h5数据库读取数据
        with h5py.File(h5_path, 'r') as data:
            coordinates = np.array(data['coordinates'])
            patch_level = int(data['coordinates'].attrs['patch_level'])
            patch_size = tuple(data['coordinates'].attrs['patch_size'])
            scores = np.array(data['scores'])
        print('现在开始生成{}（{}/{}）的热图缩略图'.format(wsi_name, index, wsi_num))
        # 生成热图
        heatmap_generator = PatchBasedHeatmapGenerator(wsi_path, patch_level, coordinates, scores, patch_size)
        thumbnail_coolwarm, heatmap_coolwarm = heatmap_generator.generate_heatmap((0.125, 0.125), 'coolwarm', 0.5, 'sigmod')
        heatmap_coolwarm.save(os.path.join(heatmap_dir, '{}_heatmap_coolwarm.png'.format(wsi_name)))
        thumbnail_coolwarm.save(os.path.join(thumbnail_dir, '{}_thumbnail.png'.format(wsi_name)))
        del heatmap_coolwarm
        del thumbnail_coolwarm
        thumbnail_rank, heatmap_rank = heatmap_generator.generate_heatmap((0.125, 0.125), 'coolwarm', 0.5, 'rank')
        heatmap_rank.save(os.path.join(heatmap_dir, '{}_heatmap_rank.png'.format(wsi_name)))
        del heatmap_rank
        del thumbnail_rank
        thumbnail_seismic, heatmap_seismic = heatmap_generator.generate_heatmap((0.125, 0.125), 'seismic', 0.5, 'sigmod')
        heatmap_seismic.save(os.path.join(heatmap_dir, '{}_heatmap_seismic.png'.format(wsi_name)))
        del heatmap_seismic
        del thumbnail_seismic
        del heatmap_generator
    print('生成完毕')


if '__main__' == __name__:
    # 参数
    parser = argparse.ArgumentParser(description='Patch based heatmap generator')
    parser.add_argument('--h5_dir', type=str, default='./patches/', help='包含坐标及预测patch得分数据的目录')
    parser.add_argument('--wsi_dir', type=str, default='/repository01/houjianxin_build/clam/heatmap_test/wsi/', help='包含WSI的目录')
    parser.add_argument('--thumbnail_dir', type=str, default='./thumbnails/', help='准备保存缩略图的目录，默认为./thumbnails/')
    parser.add_argument('--heatmap_dir', type=str, default='./heatmaps/', help='准备保存热图缩略图的目录，默认为./heatmaps/')
    args = parser.parse_args()
    # start（你需要提供的参数）
    #       包含坐标及预测得分数据的目录
    h5_dir = args.h5_dir
    #       包含WSI的目录，h5_dir中的文件和wsi_dir中的文件应该是一一对应的
    wsi_dir = args.wsi_dir
    #       准备保存缩略图的目录
    thumbnail_dir = args.thumbnail_dir
    #       准备保存热图缩略图的目录
    heatmap_dir = args.heatmap_dir
    # end
    if False == os.path.exists(thumbnail_dir):
        os.mkdir(thumbnail_dir)
    if False == os.path.exists(heatmap_dir):
        os.mkdir(heatmap_dir)
    main(h5_dir = h5_dir, wsi_dir = wsi_dir, thumbnail_dir = thumbnail_dir, heatmap_dir = heatmap_dir)