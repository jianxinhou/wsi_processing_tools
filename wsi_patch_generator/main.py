'''
Author: jianxinhou
Date: 2021-05-20 22:07:57
LastEditTime: 2021-05-21 20:17:44
LastEditors: jianxinhou
Description: 这是一个包含如何使用PatchBasedHeatmapGenerator的示例，切割的图像以Camelyon16为例
FilePath: /wsi_patch_generator/main.py

                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑       永不宕机     永无BUG
'''

import os
import argparse
from numpy.lib.npyio import save
import h5py
import numpy as np
from core.WSIPatchGenerator import WSIPatchGenerator
from utils.tool import load_contour_from_xml_file

def main(wsi_dir, annotation_dir = None, mask_dir='./mask', patches_dir='./patches'):
    '''
    @description: 主函数
    '''   
    assert(True == os.path.isdir(wsi_dir))
    segment_level = 6
    patch_level = 0
    patch_size = (256, 256)
    step_size = (256, 256)
    # 列出wsi目录中的所有文件
    all_wsi = os.listdir(wsi_dir)
    # 开始切图
    wsi_num = len(all_wsi)
    print('共{}张WSI，开始生成'.format(wsi_num))
    for index, wsi in enumerate(all_wsi, start = 1):
        print('开始处理第{}/{}张wsi'.format(index, wsi_num))
        tumor_contours = None
        # 生成wsi不带后缀的文件名
        wsi_name, ext = os.path.splitext(wsi)
        # 生成wsi图像，xml文件，mask图像，h5数据库的路径
        wsi_path = os.path.join(wsi_dir, wsi)
        xml_path = None
        if None != annotation_dir:
            xml_path = os.path.join(annotation_dir, '{}.xml'.format(wsi_name))
        if None != xml_path and True == os.path.exists(xml_path):
            tumor_contours = load_contour_from_xml_file(xml_path=xml_path)
        h5_path = os.path.join(patches_dir, "{}.h5".format(wsi_name))
        mask_path = os.path.join(mask_dir, "{}.png".format(wsi_name))
        # 开始切图
        patch_generator = WSIPatchGenerator(slide_path=wsi_path, tumor_contours=tumor_contours)
        # 分割组织区域
        mask = patch_generator.segment_tissue(segment_level=segment_level,
                    min_threshold=8,
                    min_tissue_area=26214400,   # 512*512*100，检测出的组织轮廓面积至少能够包含100个512*512的图块
                    min_hole_area=4194304)      # 512*512*16，检测出的孔洞区域面积
        mask.save(mask_path)
        # 根据组织区域切小patch
        data = patch_generator.draw_patch_within_contours(patch_level, patch_size, step_size, max_thread_number=10)
        # 将patch信息全部放入一个大数组
        patches = np.empty((0,2),dtype='int32', order='C')
        labels = np.empty((0),dtype='int32', order='C') 
        for value in data.values():
            patches = np.append(patches, value['patches'], axis=0)
            labels = np.append(labels, value['labels'], axis=0)
        # 保存h5文件
        with h5py.File(h5_path, mode='w') as f:
            f.create_dataset('patches', data = patches)
            f.create_dataset('labels', data = labels)
            f['patches'].attrs['patch_level'] = 0
            f['patches'].attrs['patch_size'] = (256, 256)
        print()
    print('处理完成！')

if '__main__' == __name__:
    # 参数
    parser = argparse.ArgumentParser(description='Patch based heatmap generator')
    parser.add_argument('--save_dir', type=str, default='./patches/', help='保存patches等数据的目录')
    parser.add_argument('--wsi_dir', type=str, default='/repository02/houjianxin_build/dataset_code/CAMELYON16/testing/images', help='包含WSI的目录')
    parser.add_argument('--annotation_dir', type=str, default='/repository02/houjianxin_build/dataset_code/CAMELYON16/testing/annotation', help='包含对WSI肿瘤区域标注的目录')
    args = parser.parse_args()
    # start（你需要提供的参数）
    #       保存patches和mask的目录
    save_dir = args.save_dir
    #       包含WSI的目录
    wsi_dir = args.wsi_dir
    # end
    mask_dir = os.path.join(save_dir, 'mask')
    patchs_dir = os.path.join(save_dir, 'patches')
    if False == os.path.exists(save_dir):
        os.mkdir(save_dir)
    if False == os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    if False == os.path.exists(patchs_dir):
        os.mkdir(patchs_dir)
    main(wsi_dir=wsi_dir, annotation_dir=args.annotation_dir, mask_dir=mask_dir, patches_dir=patchs_dir)
    