'''
Author: jianxinhou
Date: 2021-05-19 16:39:14
LastEditTime: 2021-05-25 12:55:18
LastEditors: jianxinhou
Description: 
            摘要:
                WSIPatchGenerator是一个组织病理WSI的patch提取器，可以分割组织图像中的有效组织区域并从分割出的区域中提取patch
            使用示例:
                # 建立切割对象
                patch_generator = WSIPatchGenerator(slide_path=wsi_path, tumor_contours=tumor_contours)
                # 分割组织区域
                mask = patch_generator.segment_tissue(segment_level=segment_level,
                    min_threshold=8,
                    min_tissue_area=26214400,   # 512*512*100，检测出的组织轮廓面积至少能够包含100个512*512的图块
                    min_hole_area=4194304)      # 512*512*16，检测出的孔洞区域面积
                # 根据组织区域切patch
                data = patch_generator.draw_patch_within_contours(patch_level, patch_size, step_size, max_thread_number=0)
FilePath: /wsi_patch_generator/core/WSIPatchGenerator.py
'''

import openslide
import cv2
import numpy as np
import multiprocessing as mp
from PIL import Image
from utils.tool import scale_contours, scale_holes_contours
from utils.tool import filter_coordinate
from utils.tool import check_patch_in_contour, is_lefttop_in_contour, is_center_in_contour, is_one_point_in_contour, is_four_point_both_in_contour

class WSIPatchGenerator:
    '''
    用于对组织WSI进行分割、提取patch

    Attributes:
        __slide: 待操作的WSI对象;
        __tumor_contours: 肿瘤轮廓，必须为opencv的轮廓格式，默认为None，代表WSI没有异常区域;
    '''
    def __init__(self, slide_path, tumor_contours=None):
        '''
        @description: 初始化WSIPatchGenerator
        @param: 
            slide_path: 待操作的WSI路径;
            tumor_contours: 肿瘤轮廓，必须为opencv的轮廓格式，默认为None，代表此WSI没有肿瘤.
        @return
        '''    
        self.__slide = openslide.open_slide(slide_path)
        self.__tumor_contours = tumor_contours
        self.__tissue_contours = None
        self.__holes_contours = None
    def __del__(self):
        '''
        @description: 释放资源
        ''' 
        del self.__holes_contours
        del self.__tissue_contours
        del self.__tumor_contours
        del self.__slide
    
    def segment_tissue(
            self,
            segment_level, 
            min_threshold,
            min_tissue_area,
            min_hole_area, 
            use_otsu=False, 
            max_num_holes_in_one_tissue = 8,
            median_blur_kernel_size = 7,
            morphology_close_kernel_size = 4):
        '''
        @description: segment_tissue将RGB通道转换为HSV通道后，使用S通道来对组织区域进行阈值分割.
        @param: 
            segment_level: 分割使用的WSI等级;
            min_threshold: 阈值分割使用的阈值，use_otsu为True时，此参数无效;
            min_tissue_area：组织区域的最小面积，用于筛选组织区域;
            min_hole_area: 孔洞的最小面积，用于筛选组织区域中的孔洞;
            use_otsu: 是否自动计算阈值（min_threshold）;
            median_blur_kernel_size: 用于中值滤波的卷积核大小;
            morphology_close_kernel_size: 形态学操作的卷积核大小，这里使用的是闭操作
        @return:
            self.__tissue_contours: WSI缩放等级为0的组织区域的轮廓;
            self.__holes_contours: WSI缩放等级为0的属于组织区域的孔洞区域轮廓;
            以及包含轮廓的缩略图，可以直观看到分割结果.
        '''       
        def filter_contours(all_contours, hierarchy, min_tissue_area, min_hole_area, max_num_holes_in_one_tissue):
            '''
            @description: filter_contours用于从轮廓中筛选出合法轮廓.
            @param: 
                all_contours: 所有的可用轮廓;
                hierarchy: 轮廓间关系;
                min_tissue_area：组织轮廓的最小面积，用于筛选组织轮廓;
                min_hole_area: 孔洞轮廓的最小面积，用于筛选组织轮廓中的孔洞;
                max_num_holes_in_one_tissue: 单个组织轮廓中最大孔洞轮廓个数.
            @return:
                filtered_tissue_contours：WSI缩放等级为segment_level的通过筛选的组织区域轮廓;
                filtered_hole_contours: WSI缩放等级为segment_level的属于filtered_tissue_contours的孔洞轮廓;
            '''
            # 过滤完成的轮廓id
            filtered_tissue_contour_ids = []
            # 所有的孔洞轮廓
            hole_of_tissue_contour_ids = []
            # 所有的前景轮廓
            all_tissue_contour_ids = np.flatnonzero(hierarchy[:, 1] == -1)
            for contour_id in all_tissue_contour_ids:
                selected_contour = all_contours[contour_id]
                selected_hole_ids = np.flatnonzero(hierarchy[:, 1] == contour_id)
                 # 当前轮廓总面积
                area = cv2.contourArea(selected_contour)
                if area == 0: continue
                # 孔洞面积
                holes_area = [cv2.contourArea(all_contours[hole_id]) for hole_id in selected_hole_ids]
                # 轮廓真实面积
                area = area - np.array(holes_area).sum()
                if area == 0: continue
                if int(min_tissue_area) < int(area): 
                    filtered_tissue_contour_ids.append(contour_id)
                    hole_of_tissue_contour_ids.append(selected_hole_ids)
             # 所有完成过滤的前景轮廓
            filtered_tissue_contours = [all_contours[contour_id] for contour_id in filtered_tissue_contour_ids]
            # 孔洞轮廓
            filtered_hole_contours = []
            for hole_ids in hole_of_tissue_contour_ids:
                unfiltered_holes = [all_contours[id] for id in hole_ids]
                # 使用面积对孔洞排序
                unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # 根据max_num_holes_in_one_tissue筛选
                unfiltered_holes = unfiltered_holes[:max_num_holes_in_one_tissue]
                filtered_holes = []
                # 根据面积对孔洞过滤
                for hole in unfiltered_holes:
                    if cv2.contourArea(hole) > min_hole_area:
                        filtered_holes.append(hole)
                filtered_hole_contours.append(filtered_holes)
            return filtered_tissue_contours, filtered_hole_contours
        #数据准备
        patch_downsample = self.__slide.level_downsamples[segment_level]
        min_tissue_area = int(min_tissue_area / (patch_downsample * patch_downsample))
        min_hole_area = int(min_hole_area / (patch_downsample * patch_downsample))
        # RGB图像矩阵
        image_rgb_array = np.array(
                self.__slide.read_region((0,0), 
                segment_level, 
                self.__slide.level_dimensions[segment_level])
            )[:,:,0:3]
        # HSV图像矩阵
        image_hsv_array = cv2.cvtColor(image_rgb_array, cv2.COLOR_RGB2HSV)
        # 对S通道进行中值滤波
        image_median_s_array = cv2.medianBlur(image_hsv_array[:,:,1], median_blur_kernel_size)
        print('开始提取轮廓')
        # 阈值分割
        if True == use_otsu:
            _, image_s_segment = cv2.threshold(image_median_s_array, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, image_s_segment = cv2.threshold(image_median_s_array, min_threshold, 255, cv2.THRESH_BINARY)
        # 闭操作
        morphology_kernel = np.ones((morphology_close_kernel_size, morphology_close_kernel_size), np.uint8)
        image_s_segment = cv2.morphologyEx(image_s_segment, cv2.MORPH_CLOSE, morphology_kernel)
        # 寻找轮廓
        _, contours, hierarchy = cv2.findContours(image_s_segment, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        # 根据面积筛选轮廓
        tissue_contours, hole_contours = filter_contours(contours, hierarchy,min_tissue_area=min_tissue_area, min_hole_area=min_hole_area, max_num_holes_in_one_tissue=max_num_holes_in_one_tissue)
        print('共提取了{}个轮廓'.format(len(tissue_contours)))
        # 绘制轮廓图
        image_rgb_array_copy = image_rgb_array.copy()
        cv2.drawContours(image_rgb_array_copy, tissue_contours, -1, (69, 183, 135), 3, lineType=cv2.LINE_8)
        for contours in hole_contours:
            for contour in contours:
                cv2.drawContours(image_rgb_array_copy, contour, -1, (47, 144, 185), 3, lineType=cv2.LINE_8)
        scale = self.__slide.level_downsamples[segment_level]
        if None != self.__tumor_contours:
            tumor = [np.array(cont / scale, dtype='int32') for cont in self.__tumor_contours]
        else:
            tumor = []
        for contour in tumor:
            cv2.drawContours(image_rgb_array_copy, contour, -1, (238,63,77), 3, lineType=cv2.LINE_8) 
        # 保存轮廓
        self.__tissue_contours = scale_contours(tissue_contours, scale)
        self.__holes_contours = scale_holes_contours(hole_contours, scale)
        # 返回分割图像
        return Image.fromarray(image_rgb_array_copy)
        
    def draw_patch_within_contours(self, patch_level, patch_size, step_size, max_thread_number=10, check_method='four_point_easy'):
        '''
        @description: 
            从轮廓区域内提取patch
        @param:
            patch_level: 取patch的WSI缩放等级;
            patch_size: patch大小;
            step_size: 滑动窗口步长;
            max_thread_number: 线程数;
            check_method: 判断patch是否在轮廓内的函数，在以下选项中选择:
                'four_point_easy': patch四个角点和中心点中的一点在轮廓中;
                'four_point_hard': patch四个角点都在轮廓中;
                'center': patch中心点在轮廓中;
                'basic': patch左上角点在轮廓中. 
        @return:
            data: 是一个字典，包含每个轮廓内的patch坐标和标签，.
        '''
        assert(check_method in ['four_point_easy', 'four_point_hard', 'center', 'basic'])        
        # 下采样块大小
        patch_downsample = self.__slide.level_downsamples[patch_level]
        # 在放大等级0下的patch_size
        ref_patch_size = (int(patch_size[0]*patch_downsample), int(patch_size[1]*patch_downsample))
        # 在放大等级0下的滑动窗口移动步长
        step_size = (int(step_size[0]*patch_downsample), int(step_size[1]*patch_downsample))
        # 保存patch的字典，key为轮廓id，value为patch坐标
        data = {}
        # 开始提取patch
        for contour_id, contour in enumerate(self.__tissue_contours):
            print('开始提取第{}个轮廓的patch'.format(contour_id))
            # 最小外接矩形
            start_x, start_y, w, h = cv2.boundingRect(contour) if contour is not None else (0, 0, self.__slide.level_dimensions[0][0], self.__slide.level_dimensions[0][1])
            stop_x, stop_y = start_x + w, start_y + h
            if isinstance(check_method, str):
                if check_method == 'four_point_easy':
                    cont_check_fn = is_one_point_in_contour(contour=contour, patch_size=ref_patch_size, center_shift=(0.5,0.5))
                elif check_method == 'four_point_hard':
                    cont_check_fn = is_four_point_both_in_contour(contour=contour, patch_size=ref_patch_size, center_shift=(0.5,0.5))
                elif check_method == 'center':
                    cont_check_fn = is_center_in_contour(contour=contour, patch_size=ref_patch_size)
                elif check_method == 'basic':
                    cont_check_fn = is_lefttop_in_contour(contour=contour)
                else:
                    raise NotImplementedError
            else:
                assert isinstance(check_method, check_patch_in_contour)
                cont_check_fn = check_method
            # 获取矩形中符合要求的所有x,y坐标
            x_range = np.arange(start_x, stop_x, step=step_size[0])
            y_range = np.arange(start_y, stop_y, step=step_size[1])
            # 孔洞轮廓
            hole_contours = self.__holes_contours[contour_id]
            # 将x,y坐标拼成网格
            x_coordinates, y_coordinates = np.meshgrid(x_range, y_range, indexing='ij')
            # 得到没有经过筛选的patch坐标
            unfiltered_coordinates = np.array([x_coordinates.flatten(), y_coordinates.flatten()]).transpose()
            # 筛选patch
            iterable = [(coordinate, hole_contours ,self.__tumor_contours, ref_patch_size, cont_check_fn) for coordinate in unfiltered_coordinates]
            cpu_number = mp.cpu_count()
            if max_thread_number > cpu_number:
                max_thread_number = cpu_number
            if max_thread_number < 1:
                # 单线程，用于VSCode调试
                results = np.empty(unfiltered_coordinates.shape[0], dtype=object, order='C')
                for coordinate_id, item in enumerate(iterable):
                    record = filter_coordinate(item[0],item[1],item[2],item[3],item[4])
                    if None != record:
                        results[coordinate_id] = np.array(record)        
                results = np.array([result for result in results if result is not None])           
            else:
                # 多线程，快
                pool = mp.Pool(max_thread_number)
                results = pool.starmap(filter_coordinate, iterable)
                pool.close()
                results = np.array([result for result in results if result is not None])
            # 筛选完成，整理数据
            print('共{}个patch'.format(results.shape[0]))
            temp_coordinates = np.array([np.array(result[0]) for result in results], dtype = 'int32')
            temp_labels = np.array([int(result[1]) for result in results], dtype='int32')
            temp_data = {}
            temp_data['coordinates'] = temp_coordinates
            temp_data['labels'] = temp_labels
            # 保存
            data[contour_id] = temp_data
        # 返回
        return data


                
            
            


    