'''
Author: jianxinhou
Date: 2021-05-20 20:21:41
LastEditTime: 2021-05-21 19:28:17
LastEditors: jianxinhou
Description: 提供其他代码需要用到的工具函数
FilePath: /wsi_patch_generator/core/tool.py
'''

import cv2
import numpy as np
from xml.dom import minidom

def load_contour_from_xml_file(xml_path):
    '''
    @description: 从xml文件中读取肿瘤轮廓.
    @param:
        xml_path: 包含标注信息的xml文件路径.
    @return:
        从xml文件中读取到的肿瘤轮廓.
    '''    
    def _createContour(coord_list):
        return np.array([[[int(float(coord.attributes['X'].value)), 
                           int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

    xmldoc = minidom.parse(xml_path)
    annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
    contours_tumor = [_createContour(coord_list) for coord_list in annotations]
    contours_tumor = sorted(contours_tumor, key=cv2.contourArea, reverse=True)
    return contours_tumor

def scale_contours(contours, scale):
    '''
    @description: 将肿瘤轮廓缩放至相应比例.
    @param:
        contours: 被缩放轮廓;
        scale: 比例.
    @return:
        缩放后的轮廓.
    '''    
    return [np.array(cont * scale, dtype='int32') for cont in contours]

def scale_holes_contours(contours, scale):
    '''
    @description: 将孔洞轮廓缩放至相应比例.
    @param :
        contours: 被缩放的孔洞轮廓;
        scale: 比例.
    @return:
        缩放后的孔洞轮廓.
    '''    
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

def is_patch_in_tumor(point, tumor_contours, patch_size):
    '''
    @description: 
        根据patch四个角点和中心点判断patch是否为肿瘤，是则返回1，否则返回0;
        To Fix:这里有一个问题，这里只用了patch的四个角点和中心点来进行判断，当patch较大或肿瘤区域较小的时候，可能发生肿瘤区域位于patch中，但patch的四个角点和中点都不在肿瘤区域中的情况.
    @param :
        point: 点;
        tumor_contours: 肿瘤轮廓;
        patch_size: patch大小
    @return 
        0: 不属于肿瘤;
        1: 属于肿瘤.
    '''    
    if None == tumor_contours:
	    return 0
    for tumor in tumor_contours:
        shift = (int(patch_size[0]//2), int(patch_size[1]//2))
        center = (point[0]+shift[0], point[1]+shift[0])
        all_points = [(center[0]-shift[0], center[1]-shift[0]), 
                        (center[0]+shift[0], center[1]+shift[0]), 
                        (center[0]+shift[0], center[1]-shift[0]),
                        (center[0]-shift[0], center[1]+shift[0]),
                        (center[0], center[1])]
        for point in all_points:
            if cv2.pointPolygonTest(tumor, point, False) >= 0:
                return 1
    return 0

def is_patch_in_tissue_holes(point, hole_contours, patch_size):
    '''
    @description: 
        判断patch是否在孔洞区域中，是则返回1，否则返回0.
    @param: 
        point: 点
        hole_contours: 孔洞轮廓;
        patch_size: patch大小
    @return:
        0: 不在孔洞中;
        1: 在孔洞中.
    '''    
    for hole in hole_contours:
        if cv2.pointPolygonTest(hole, (point[0]+patch_size[0]//2, point[1]+patch_size[1]//2), False) > 0:
            return 1
    return 0

def is_patch_in_tissue(point, patch_size, hole_contours, contour_check_fn):
    '''
    @description: 
        判断patch是否在组织区域中，是则返回1，否则返回0.
    @param: 
        point: 点;
        hole_contours：孔洞轮廓;
        patch_size: patch大小;
        contour_check_fn: 检测函数类（check_patch_in_contour的子类），其中已经包含了组织区域轮廓.
    @return:
        0: 不在孔洞中;
        1: 在孔洞中.
    '''    
    if contour_check_fn(point):
        if hole_contours is not None:
            return not is_patch_in_tissue_holes(point=point,hole_contours=hole_contours,patch_size=patch_size)
        else:
            return 1
    return 0

def filter_coordinate(coordinate, hole_contours ,tumor_contours, patch_size, contour_check_fn):
    '''
    @description: 
        根据轮廓过滤坐标.
    @param: 
        coordinate: 待过滤的坐标;
        hole_contours：孔洞轮廓;
        tumor_contours: 肿瘤轮廓;
        patch_size: patch大小;
        contour_check_fn: 检测函数类（check_patch_in_contour的子类），其中已经包含了组织区域轮廓.
    @return:
        (coordinate, label): 坐标通过过滤机制，没有被筛掉，返回的分别为坐标及其标签（取值为0或1，0为非肿瘤，1为肿瘤）;
        None: 被过滤掉.
    '''    
    if is_patch_in_tissue(point=coordinate,patch_size=patch_size, hole_contours=hole_contours, contour_check_fn=contour_check_fn):
        label = is_patch_in_tumor(point=coordinate, tumor_contours=tumor_contours, patch_size=patch_size)
        return (coordinate, label)
    else:
        return None

# 以下五个为一组类，功能都是判断patch是否在轮廓中

class check_patch_in_contour(object):
    '''
    基类
    ''' 
    def __call__(self, pt): 
	    raise NotImplementedError

class is_lefttop_in_contour(check_patch_in_contour):
    '''
    只判断patch左上角坐标是否在轮廓中，是则返回true，否则返回false
    ''' 
    def __init__(self, contour):
        self.__contour = contour

    def __call__(self, point): 
	    return 1 if cv2.pointPolygonTest(self.__contour, point, False) >= 0 else 0

class is_center_in_contour(check_patch_in_contour):
    '''
    只判断patch中心点坐标是否在轮廓中，是则返回true，否则返回false
    ''' 
    def __init__(self, contour, patch_size):
	    self.__contour = contour
	    self.__patch_size = patch_size

    def __call__(self, point): 
	    return 1 if cv2.pointPolygonTest(self.__contour, (point[0]+self.__patch_size//2, point[1]+self.__patch_size//2), False) >= 0 else 0

class is_one_point_in_contour(check_patch_in_contour):
    '''
    判断patch内某个矩形区域的四个点坐标是否至少有一个在轮廓中，是则返回true，否则返回false
    ''' 
    def __init__(self, contour, patch_size, center_shift=(0.5,0.5)):
	    self.__contour = contour
	    self.__patch_size = patch_size
	    self.__shift = (int(patch_size[0]//2*center_shift[0]), int(patch_size[1]//2*center_shift[1]))
    def __call__(self, point): 
	    center = (point[0]+self.__patch_size[0]//2, point[1]+self.__patch_size[1]//2)
	    if self.__shift[0] > 0 and self.__shift[1] > 0:
		    all_points = [(center[0]-self.__shift[0], center[1]-self.__shift[1]),
						  (center[0]+self.__shift[0], center[1]+self.__shift[1]),
						  (center[0]+self.__shift[0], center[1]-self.__shift[1]),
						  (center[0]-self.__shift[0], center[1]+self.__shift[1])
						  ]
	    else:
		    all_points = [center]
	    for one_point in all_points:
		    if cv2.pointPolygonTest(self.__contour, one_point, False) >= 0:
			    return 1
	    return 0

class is_four_point_both_in_contour(check_patch_in_contour):
    '''
    判断patch内某个矩形区域的四个点坐标是否全部在轮廓中，是则返回true，否则返回false
    ''' 
    def __init__(self, contour, patch_size, center_shift=(0.5, 0.5)):
        self.__contour = contour
        self.__patch_size = patch_size
        self.__shift = (int(patch_size[0]//2*center_shift[0]), int(patch_size[1]//2*center_shift[1]))
    def __call__(self, point): 
        center = (point[0]+self.__patch_size[0]//2, point[1]+self.__patch_size[1]//2)
        if self.__shift > 0:
            all_points = [(center[0]-self.__shift[0], center[1]-self.__shift[1]),
                          (center[0]+self.__shift[0], center[1]+self.__shift[1]),
                          (center[0]+self.__shift[0], center[1]-self.__shift[1]),
                          (center[0]-self.__shift[0], center[1]+self.__shift[1])
                          ]
        else:
            all_points = [center]
        
        for one_point in all_points:
            if cv2.pointPolygonTest(self.__contour, one_point, False) < 0:
                return 0
        return 1