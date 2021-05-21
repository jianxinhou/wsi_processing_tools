'''
Author: jianxinhou
Date: 2021-04-01 20:48:07
LastEditTime: 2021-05-19 19:27:02
LastEditors: Please set LastEditors
Description:    
            摘要：
                PatchBasedHeatmapGenerator 是一个基于Patch分类方法的WSI热图生成器
                所有基于patch分类的深度学习方法均可使用PatchBasedHeatmapGenerator来对WSI生成热图
                可以反映Patch对分类方法的贡献度或整张WSI中可能的异常区域等
            使用示例:
                heatmap_generator = PatchBasedHeatmapGenerator(wsi_path, patch_level, coordinates, scores, patch_size)
                thumbnail, heatmap = heatmap_generator.generate_heatmap((0.0625, 0.0625), 'coolwarm', 0.5, 'sigmod')
FilePath: /patch_based_heatmap_generator/core/PatchBasedHeatmapGenerator.py
'''

import cv2
import numpy as np
import openslide
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import rankdata

class PatchBasedHeatmapGenerator(object):
    '''
    用于为WSI生成基于Patch分类的、可使用一般图像查看工具打开的WSI热图缩略图。

    Attributes:
        __slide: 被生成热图的WSI，
        __patch_level: 取patch的等级
        __coordinates: patch的坐标，
        __scores: 每个patch被分类为异常区域的概率，
        __patch_size： patch的尺寸
    '''

    def __init__(self, slide_path, patch_level, coordinates, scores, patch_size):
        '''
        @description: 初始化
        @param {
            slide_path: 待生成热图的WSI图像路径，
            patch_level: 取patch的wsi缩放等级，
            coordinates: 用于生成热图的patch的坐标，尺寸为N*2，N为patch数目，坐标为二维坐标，
            scores: 每个patch为异常区域的分数，尺寸为N*1，N为patch数目，模型不同值域不同，
            patch_size: patch的尺寸。 
        }
        '''        
        self.__slide = openslide.open_slide(slide_path)
        self.__patch_level = patch_level
        self.__coordinates = coordinates
        self.__scores = scores
        self.__patch_size = patch_size
        
    def __del__():
        '''
        @description: 释放资源
        '''        
        del self.__slide
        del self.__patch_level
        del self.__coordinates
        del self.__scores
        del self.__patch_size



    def generate_heatmap(self, 
                         thumbnail_size_scale = (0.5, 0.5), 
                         style = 'coolwarm', 
                         alpha = 0.5,
                         normalize_method = "close" 
                         ):
        '''
        @description: 完成构造后，使用此方法生成自定义热图缩略图
        @param {
            thumbnail_size: 缩略图与原始WSI的宽高比例，被限制在THUMBNAIL_SIZE_LOWER_LIMIT到THUMBNAIL_SIZE_UPPER_LIMIT之间，同时生成的缩略图尺寸最大为THUMBNAIL_MAX_SIZE，最小为THUMBNAIL_MIN_SIZE，
            style: 热图样式，默认为coolwarm
            alpha: 热图的透明度，默认为0.5
            normalize_method: 将scores映射到0和1区间的方法，close为无需映射，默认为close
        }
        @return {
            thumbnail：缩略图原图
            heatmap：生成的缩略图热图
        }
        '''        
        # 确保scores和coordinates的个数是一样的
        assert(len(self.__scores) == len(self.__coordinates))
        # 确保patch_size包含两个元素
        assert(2 == len(self.__patch_size))
        # 确保当前访问的slide级别可用
        slide_level_count = self.__slide.level_count
        assert(self.__patch_level >= 0 and self.__patch_level < slide_level_count)
        slide_size = self.__slide.level_dimensions[self.__patch_level]    
        # 确保 thumbnail_size_scale 只包含两个元素，且每个元素都在0.1到0.5之间，同时保证尺寸在合理区间内
        assert(2 == len(thumbnail_size_scale))
        assert(self.THUMBNAIL_SIZE_SCALE_LOWER_LIMIT <= thumbnail_size_scale[0] and self.THUMBNAIL_SIZE_SCALE_UPPER_LIMIT >= thumbnail_size_scale[0])
        assert(self.THUMBNAIL_SIZE_SCALE_LOWER_LIMIT <= thumbnail_size_scale[1] and self.THUMBNAIL_SIZE_SCALE_UPPER_LIMIT >= thumbnail_size_scale[1])
        width = int(thumbnail_size_scale[0] * slide_size[0]) + 1
        height = int(thumbnail_size_scale[1] * slide_size[1]) + 1
        assert(self.THUMBNAIL_MIN_SIZE <= width and self.THUMBNAIL_MAX_SIZE >= width)
        assert(self.THUMBNAIL_MIN_SIZE <= height and self.THUMBNAIL_MAX_SIZE >= height)
        # 确保style是可用的
        assert(style in self.AVAILABLE_HEATMAP_STYLE)
        # 确保alpha在合理范围内
        assert(0 <= alpha and 1 >= alpha)
        # 确保normalize_method是可用的
        assert(normalize_method in self.AVAILABLE_NORMALIZE_METHOD)
        # 参数校验完毕，开始生成热图
        print('开始生成热图，原始WSI尺寸为({}, {})，生成的缩略图尺寸为({}, {})'.format(slide_size[0], slide_size[1], width, height))
        #       缩略图
        thumbnail_size = (width, height)
        thumbnail = self.__slide.get_thumbnail(thumbnail_size)
        thumbnail_size = thumbnail.size
        #       WSI图像patch映射到heatmap中patch的大小
        heatmap_patch_size = (int(self.__patch_size[0] * thumbnail_size_scale[0]), int(self.__patch_size[0] * thumbnail_size_scale[1]))
        #       对scores执行normalize
        scores = self.__scores.copy()
        if normalize_method == self.AVAILABLE_NORMALIZE_METHOD[self.SIGMOD]:
            #       sigmod
            scores = 1 / (1 + np.exp(-self.__scores))
        elif normalize_method == self.AVAILABLE_NORMALIZE_METHOD[self.RANK]:
            #       根据排名normalize
            scores = rankdata(self.__scores, 'average')
            scores = scores / len(scores) 
        else:
            pass
        #       确定样式
        selected_style = 'coolwarm'
        if style in self.AVAILABLE_HEATMAP_STYLE:
            selected_style = style
        color_map = plt.get_cmap(selected_style)
        #       用于保存每个像素的累计heat值
        overlay = np.full(np.flip(thumbnail_size), 0).astype(np.float64)
        #       用于保存经过每个像素的patch个数
        counter = np.full(np.flip(thumbnail_size), 0).astype(np.uint16) 
        #       开始计算overlay和counter
        patch_num = len(self.__coordinates)
        for index in range(len(self.__coordinates)):
            if index % (patch_num // 10) == 0:
                print('进度： {} / {}'.format(index, patch_num))
            #       当前patch的分数
            score = scores[index]
            #       当前patch的坐标
            coordinate = self.__coordinates[index]
            position_x = int(coordinate[0] * thumbnail_size_scale[0])
            position_y = int(coordinate[1] * thumbnail_size_scale[1])
            #       累计patch的heat值和计数
            overlay[position_y : position_y + heatmap_patch_size[1], position_x : position_x + heatmap_patch_size[0]] += score
            counter[position_y : position_y + heatmap_patch_size[1], position_x : position_x + heatmap_patch_size[0]] += 1
        print('进度： {} / {}'.format(patch_num, patch_num))
        # 计算每个像素的heat值
        zero_mask = (0 != counter)
        overlay[zero_mask] = overlay[zero_mask] / counter[zero_mask]
        # 缩略图副本
        thumbnail_copy = np.array(thumbnail.convert("RGB"))
        color = (color_map(overlay) * 255)[:,:,:3].astype(np.uint8)
        # 生成热图
        heatmap = cv2.addWeighted(thumbnail_copy, 1 - alpha, color, alpha, 0) 
        heatmap_image = Image.fromarray(heatmap)
        print('完成！')
        # 释放资源 --
        del color
        del thumbnail_copy
        del zero_mask
        del counter
        del overlay
        del scores
        # 返回热图
        return thumbnail, heatmap_image
    
    # 一些常量
    #   缩略图缩放比例阈值
    THUMBNAIL_SIZE_SCALE_UPPER_LIMIT = 0.5
    THUMBNAIL_SIZE_SCALE_LOWER_LIMIT = 0.01
    #   缩略图尺寸阈值
    THUMBNAIL_MAX_SIZE = 30000
    THUMBNAIL_MIN_SIZE = 500
    #   可用样式
    AVAILABLE_HEATMAP_STYLE = ('coolwarm', 'hot', 'bwr','Spectral', 'seismic')
    #   可用normalize_method
    SIGMOD = 1
    RANK = 2
    AVAILABLE_NORMALIZE_METHOD = ('close', 'sigmod', 'rank')
        