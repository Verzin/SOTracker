# import matplotlib

# from PIL import Image, ImageTk
#import tkinter as tk

# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import os
import numpy as np
import torch
import cvui


#import turtle
import ffmpeg
import datetime
from interval import Interval
import math 
import threading
import time
import queue
import pickle
import xlsxwriter
import pandas as pd
from collections import deque
#import plotly_express as px

from pytracking.Beisaier import smoothing_base_bezier


##*********************计算物体运动速度前，需要先测量出物体的宽度(单位:cm)
ObjectLength_real = 4.51

#定义一个存储中心点坐标的列表，用作轨迹绘制
CenterPointList = []
CenterPointListX =[]
CenterPointListY =[]
#定义队列长度为2
Q = queue.Queue()
q1 = queue.Queue(maxsize = 2)
q2 = queue.Queue(maxsize = 2)
q3 = queue.Queue(maxsize = 2)
q4 = queue.Queue(maxsize = 2)
#把取得的角度定义为一个全局变量
RotatingAngle = 0
Vel_Angle = 0
dir_Angle = 0
Cor_Angle = 0
dir_true_side = 0
draw_point1 = (0,0)
draw_point2 = (0,0)
#获取程序执行时的时间
now_time = time.strftime("%Y-%m-%d_%H.%M.%S",time.localtime())

BG = np.zeros((300, 600, 3), np.uint8)
basic = [True]
trajectory = [False]
all_info = [False]
direction =[False]


class Point:
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
    def getx(self):
        return self.x
    def gety(self):
        return self.y
    
#定义直线函数   
class Getlen:
    def __init__(self,p1,p2):
        #创建列表进行排序
        xlist = [p1.getx(),p2.getx()]
        ylist = [p1.gety(),p2.gety()]
        xlist.sort()
        ylist.sort()
        #得出两点坐标中分别的大小值
        self.px_MAX = xlist [1]
        self.px_MIN = xlist [0]
        self.py_MAX = ylist [1]
        self.py_MIN = ylist [0]
        #得出使用勾股定理的两条边边长
        self.x=p1.getx()-p2.getx()
        self.y=p1.gety()-p2.gety()
        #用math.sqrt（）求平方根，得到两点长度
        self.len= math.sqrt((self.x**2)+(self.y**2))
    #定义得到直线长度的函数
    def getlen(self):
        return self.len        
#定义获取中心坐标的函数
def get_centerpoint(lis):
        area = 0.0
        x,y = 0.0,0.0
    
        a = len(lis)
        for i in range(a):
            lat = lis[i][0] #weidu
            lng = lis[i][1] #jingdu
    
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
    
            else:
                lat1 = lis[i-1][0]
                lng1 = lis[i-1][1]
    
            fg = (lat*lng1 - lng*lat1)/2.0
    
            area += fg
            x += fg*(lat+lat1)/3.0
            y += fg*(lng+lng1)/3.0
    
        x = x/area
        y = y/area
    
        return x,y
#把数据保存起来
def save_text(time,name1,result1,name2,result2,name3,result3,name4,result4):
    my_file = open(str(Outputvideofilepath)+now_time+'_output.txt','a')
    my_file.write(str('time')+str(' ')+str('the_output of ')+str(name1)+
                  str( )+str('the angle of ')+str(name2)+
                  str( )+str('the angle of ')+str(name3)+
                  str( )+str('the angle of ')+str(name4)+'\n')
    my_file.write(str(time)+str( )+str(result1)+
                  str( )+str(result2)+
                  str( )+str(result3)+
                  str( )+str(result4)+'\n')
    my_file.close

#读取保存的text
def read_text(file):
    read = open(file , r)
    content = read.readlines()
    print(content)

#定义获取线段长宽的函数，并返回长宽的中心点坐标及宽的长度
def Get_half_legth_point(p1,p2,p3,p4):
    l1 = Getlen(p1,p2)
    l2 = Getlen(p2,p3)
    l3 = Getlen(p3,p4)
    l4 = Getlen(p1,p4)
    dirt = {l1 : l1.getlen(),l2:l2.getlen(),l3:l3.getlen(),l4 :l4.getlen()}  #把类与类中的方法用键值关联起来
    Lists = sorted(dirt.items(),key=lambda dict_items:dict_items[1])     #升序排序，去键值中的值进行排序
    short1 ,short2 = Lists[0][0] , Lists[1][0]      #获取两条真宽
    long1 , long2 = Lists[2][0] , Lists[3][0]       #获取两条真长
    #设定XY的区间
    short1_X_zoom = Interval(short1.px_MIN,short1.px_MAX)
    short1_Y_zoom = Interval(short1.py_MIN,short1.py_MAX)
    short2_X_zoom = Interval(short2.px_MIN,short2.px_MAX)
    short2_Y_zoom = Interval(short2.py_MIN,short2.py_MAX)
    long1_X_zoom = Interval(long1.px_MIN,long1.px_MAX)
    long1_Y_zoom = Interval(long1.py_MIN,long1.py_MAX)
    long2_X_zoom = Interval(long2.px_MIN,long2.px_MAX)
    long2_Y_zoom = Interval(long2.py_MIN,long2.py_MAX)
    #用ife lse语句，加区间（min,max）判断加一半的点是否落在区域内
    if(short1.px_MIN + short1.x/2 in short1_X_zoom):
       half_short1_x = short1.px_MIN + short1.x/2
    else:
       half_short1_x = short1.px_MIN - short1.x/2

    if(short1.py_MIN + short1.y/2 in short1_Y_zoom):
       half_short1_y = short1.py_MIN + short1.y/2
    else:
       half_short1_y = short1.py_MIN - short1.y/2  
       
    if(short2.px_MIN + short2.x/2 in short2_X_zoom):
       half_short2_x = short2.px_MIN + short2.x/2
    else:
       half_short2_x = short2.px_MIN - short2.x/2   
       
    if(short2.py_MIN + short2.y/2 in short2_Y_zoom):
       half_short2_y = short2.py_MIN + short2.y/2
    else:
       half_short2_y = short2.py_MIN - short2.y/2

    if(long1.px_MIN + long1.x/2 in long1_X_zoom):
       half_long1_x = long1.px_MIN + long1.x/2
    else:
       half_long1_x = long1.px_MIN - long1.x/2

    if(long1.py_MIN + long1.y/2 in long1_Y_zoom):
       half_long1_y = long1.py_MIN + long1.y/2
    else:
       half_long1_y = long1.py_MIN - long1.y/2

    if(long2.px_MIN + long2.x/2 in long2_X_zoom):
       half_long2_x = long2.px_MIN + long2.x/2
    else:
       half_long2_x = long2.px_MIN - long2.x/2

    if(long2.py_MIN + long2.y/2 in long2_Y_zoom):
       half_long2_y = long2.py_MIN + long2.y/2
    else:
       half_long2_y = long2.py_MIN - long2.y/2

    #取每个坐标点最小值加上长度的一半
    draw_half_short1 = (int(half_short1_x) , int(half_short1_y))    #此处改为int
    draw_half_short2 = (int(half_short2_x) , int(half_short2_y))    #
    half_short1 = (half_short1_x , half_short1_y)    
    half_short2 = (half_short2_x , half_short2_y)  

    draw_half_long1 = (int(half_long1_x) , int(half_long1_y))    #此处改为int
    draw_half_long2 = (int(half_long2_x) , int(half_long2_y))    #
    half_long1 = (half_long1_x , half_long1_y)    
    half_long2 = (half_long2_x , half_long2_y)  

    return draw_half_short1 , draw_half_short2 , half_short1 , half_short2,draw_half_long1,draw_half_long2,half_long1,half_long2

#定义获取向量与X轴夹角的函数
def Get_CorX_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180/math.pi
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180/math.pi
    # print(angle2)
    if abs(angle1 - angle2)> 180:
        included_angle = -(360 - abs(angle1-angle2))

    else:
        included_angle = abs(angle1 - angle2)
    return included_angle

#定义获取两向量角度的函数
def Get_V_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180/math.pi
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180/math.pi
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(abs(angle1)-abs(angle2))
        if included_angle > 180:
            included_angle = 360 - included_angle
        if included_angle > 90:
            included_angle = 180 - included_angle
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
        if included_angle > 90:
            included_angle = 180 - included_angle
    return included_angle

#定义一个中心坐标和四角坐标的线程
def Get_Data(pol,half_s1,half_s2,half_l1,half_l2,q1,q2,q3,X_zhou,Counterclockwise_line):
    X_Angle = Get_CorX_angle(X_zhou,Counterclockwise_line)#.astype(np.int64)
    Center_point = get_centerpoint(pol)
    short_half_point = (half_s1 + half_s2)
    long_half_point = (half_l1 + half_l2)
    q1.put(X_Angle)
    q2.put(Center_point)
    q3.put(short_half_point)
    q4.put(long_half_point)
    CenterPointListX.append(int(Center_point[0]))
    CenterPointListY.append(int(Center_point[1]))


#定义一个实时展示角度图与轨迹的线程
def Show_Guiji(CenterPointListX,CenterPointListY,video_width,video_hight):  
    
    #实时展示轨迹
    plt.ion()#开启interactive mode 成功的关键函数
    plt.clf() #清空画布上的所有内容
    plt.plot(CenterPointListX, CenterPointListY, 'ro')
    x_curve, y_curve = smoothing_base_bezier(CenterPointListX, CenterPointListY, k=0.3, closed=False)
    #把图像与视频分辨率对应起来
    plt.xlim(0,video_width)
    plt.ylim(0,video_hight)
    #反转坐标轴
    ax = plt.gca() 
    ax.xaxis.set_ticks_position('top') 
    ax.invert_yaxis()
    #cbar = plt.colorbar(orientation = 'horizontal')
    plt.plot(x_curve, y_curve, label='$k=0.3$')
    #     x_curve, y_curve = smoothing_base_bezier(x, y, k=0.4, closed=True)
    #     plt.plot(x_curve, y_curve, label='$k=0.4$')
    #     x_curve, y_curve = smoothing_base_bezier(x, y, k=0.5, closed=True)
    #     plt.plot(x_curve, y_curve, label='$k=0.5$')
    #     x_curve, y_curve = smoothing_base_bezier(x, y, k=0.6, closed=True)
    #     plt.plot(x_curve, y_curve, label='$k=0.6$')
    #     plt.legend(loc='best')
            
    #     plt.show() 
    #plt.ioff()   
    plt.show()
    plt.pause(0.00000000000000001)

def Show_angle(data_frame):

    plt.ion()#开启interactive mode 成功的关键函数
    plt.clf() #清空画布上的所有内容 
    data_frame.plot(x = 'frame number' , y = ['the RotatingAngle',
                                                'the CornerAngle',
                                                'the VelocityAngle',
                                                'the ObjectDirectionAngle'],
                                                title = 'Angle')
    plt.legend()
    plt.show()
    plt.pause(0.01)

def Show_Speed(data_frame):
    
    plt.ion()#开启interactive mode 成功的关键函数
    plt.clf() #清空画布上的所有内容 
    data_frame.plot(x = 'frame number' , y = ['the velocity'],title = 'Speed')
    plt.legend()
    plt.show()
    plt.pause(0.01)




#定义一个获取各种角度的函数
def Get_Angle (pol,half_s1,half_s2,half_l1,half_l2,X_zhou,Counterclockwise_line):
    global RotatingAngle
    global draw_point1
    global draw_point2
    global Cor_Angle
    global Vel_Angle
    global dir_Angle
    global dir_true_side
    #判断是否队满
    if q1.full():
            #查看数据
            #print ('队列旋转角度有:%s'%(list(q1.queue)))
            #print ('队列里中心坐标有:%s'%(list(q2.queue)))
            #print ('队列里宽的中点坐标有:%s'%(list(q3.queue)))
            #把数据相减，得出物体旋转角度
            q1_first = q1.get()
            q1_second = q1.get()
            RotatingAngle = abs(q1_second - q1_first)
            #print('物体旋转角度为:%f'%RotatingAngle)
            #my_file = open('/home/ai/Verzin/pysot-master/the_output/the_RotatingAngle_result_%s.txt'%now_time,'a')
            #my_file.write(str(RotatingAngle)+'\n')
            #my_file.close
            q1.put(q1_second)
            #print('来看看第一个数放进来没有，队列中现在有'+str(q.qsize())+'个元素')
            #根据中心坐标得出运动方向，并得出相应角度
            q2_first = q2.get()
            q2_second = q2.get()
            v = (q2_first[0],q2_first[1]) + (q2_second[0],q2_second[1])
            #print('v1坐标为：%s'%(list(v1)))
            q3_Q = q3.get()
            o1 = (q3_Q[0],q3_Q[1],q3_Q[2],q3_Q[3])
            q4_Q = q4.get()
            o2 = (q4_Q[0],q4_Q[1],q4_Q[2],q4_Q[3])
            judege_Short_angle = Get_V_angle(v,o1)
            judege_long_angle = Get_V_angle(v,o2)

            if judege_Short_angle > judege_long_angle:
                o_true = o2
                dir_true_side = 1

            else:
                o_true = o1 
                dir_true_side = 0

            

            dir_Angle = Get_CorX_angle(X_zhou,o_true)
            #print('v2坐标为：%s'%(list(v2)))
            Cor_Angle = Get_V_angle (v,o_true)
            #print('夹角坐标在这：%s'%(list(v1,(v2))))
            #print ('速度与物体运动方向夹角为%s'%Cor_Angle)
            #print("取出的V1坐标为：%s"%(list(v1)))
            Vel_Angle = Get_CorX_angle(X_zhou,v)
            #print("取出的V2坐标为：%s"%(list(v2)))
            #print("物体方向夹角:%s"%dir_Angle)
            #得到两帧中心点坐标
            draw_point1 = (int(q2_first[0]),int(q2_first[1]))
            draw_point2 = (int(q2_second[0]),int(q2_second[1]))
            q2.put(q2_second)           
    else:
        #print('队未满，存放数据ing')
        get_pol = threading.Thread(target = Get_Data,args = (pol,half_s1,half_s2,half_l1,half_l2,q1,q2,q3,X_zhou,Counterclockwise_line))
        get_pol.setDaemon(True)
        get_pol.start()
    return RotatingAngle,Cor_Angle,Vel_Angle,dir_Angle,draw_point1,draw_point2,CenterPointListX,CenterPointListY,dir_true_side

def Get_Vel(x,y,scale,fps):

    length = Getlen(Point(x[0],x[1]),Point(y[0],y[1])).getlen()
    Vel_value = (length * scale)*fps
    return Vel_value


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError
    

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        #self.sequence_name = sequence.name
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state,_,_ = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times

    def track_videofile(self, Inputvideofilepath,Outputvideofilepath, ObjectLength,TypeVideo,optional_box=None,):
        """Run track with a video file input."""

        global ObjectLength_real
        global BG
        global basic
        global trajectory
        global all_info
        global direction

        assert os.path.isfile(Inputvideofilepath), "Invalid param {}".format(Inputvideofilepath)
        ", videofilepath must be a valid videofile"

        if hasattr(self, 'initialize_features'):
            self.initialize_features()
        
        if TypeVideo =='basic':
            basic = [True]
        elif TypeVideo == 'trajectory':
            trajectory = [True]
        elif TypeVideo == 'direction':
            direction =[True]
        elif TypeVideo == 'all':
            all_info = [True]
        
        print(type(TypeVideo))
        print(TypeVideo)


        cap = cv.VideoCapture(Inputvideofilepath)
        #display_name = 'Display: ' + self.params.tracker_name
        display_name = 'SOTracker:main window'
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
        cv.resizeWindow(display_name,720,480)
        cv.moveWindow(display_name,0,0)

        #获取视频的基本信息
        fps = cap.get(cv.CAP_PROP_FPS) #视频帧率
        total_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) #视频总帧率
        video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) #视频宽
        video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) #视频高
        video_time = float(total_frame/fps)

        print ("The frame rate:%d\n"%fps+
       "The resolution:%d * %d \n"%(video_width,video_height)+
       "The total frame:%d\n"%total_frame+
       "Total time:%f\n"%video_time)

        #输入底板的长度与宽度，长宽比跟视频比例大小一致


        current_frame = 1
        #写入视频分辨率，帧率，度量尺，作为X轴的最右边顶点
        Resolution = (video_width,video_height)
        #unit_dist_cm    #每一百个像素代表现实距离多少cm
        X_zhou = (0,0) + Resolution
        
        cvui.init('SOTracker:operation panel')
        BG[:] = (49, 52, 49)
        cvui.window(BG,0,0,300,300, 'information')

        cvui.printf(BG,10, 30, 0.4, 0xffffff, "The frame rate is :%d",fps)
        cvui.printf(BG,10, 50, 0.4, 0xffffff, "The resolution is:%d * %d",video_width,video_height)
        cvui.printf(BG,10, 70, 0.4, 0xffffff, "The total frame is:%d",total_frame)
        cvui.printf(BG,10, 90, 0.4, 0xffffff, "Total time:%fs",video_time)
        cvui.printf(BG,10, 110, 0.4, 0x00ff00, "Current frame:%d",current_frame)
        #cvui.printf(BG,10, 130, 0.4, 0x00ff00, "Select the information you want to view in the right pane")


        cvui.checkbox(BG,310,40,'Only show basic information', basic)
        cvui.checkbox(BG,310,80,'Show trajectory', trajectory)
        cvui.checkbox(BG,310,120,'Show direction', direction)        
        cvui.checkbox(BG,310,160,'Show all the information in main window',all_info)       
        #cvui.checkbox(BG,310,160,'Show all the information in main window',all_info)

        #cvui.button(BG,300,200,150,100, 'OK')
        #cvui.button(BG,450,200,150,100, 'Wait')
        cvui.button(BG,300,200,150,50, 'Start(Space/Enter)')
        cvui.button(BG,450,200,150,50, 'Reselect ROI(r)')
        cvui.button(BG,300,250,300,50, 'Quit(q)')

            # Display the lib version at the bottom of the screen
        cvui.printf(BG, 600 - 80, 300 - 10,0.3, 0xCECECE, 'SOTracker v0.12')
        cvui.update()
                        
        cv.imshow('SOTracker:operation panel', BG)        
        # while(True):

        #     BG[:] = (49, 52, 49)
        #     cvui.window(BG,0,0,300,300, 'information')

        #     # cvui.checkbox(BG,310,50,'Only show basic information', basic)
        #     # cvui.checkbox(BG,310,100,'Build trajectory window', trajectory)
        #     # cvui.checkbox(BG,310,150,'Show all the information in main window',all_info)

        #     cvui.printf(BG,10, 30, 0.4, 0x00ff00, "The frame rate is :%d",fps)
        #     cvui.printf(BG,10, 50, 0.4, 0x00ff00, "The resolution is:%d * %d",video_width,video_height)
        #     cvui.printf(BG,10, 70, 0.4, 0x00ff00, "The total frame is:%d",total_frame)
        #     cvui.printf(BG,10, 90, 0.4, 0x00ff00, "Total time:%fs",video_time)
        #     cvui.printf(BG,10, 110, 0.4, 0x00ff00, "Current frame:%d",current_frame)
        #     cvui.printf(BG,10, 130, 0.4, 0x00ff00, "Select the information you want to view in the right pane")


        #     cvui.checkbox(BG,310,40,'Only show basic information', basic)
        #     cvui.checkbox(BG,310,80,'show trajectory', trajectory)
        #     cvui.checkbox(BG,310,120,'show direction', direction)        
        #     cvui.checkbox(BG,310,160,'Show all the information in main window',all_info)

        #     cvui.button(BG,300,200,150,100, 'OK')
        #     cvui.button(BG,450,200,150,100, 'Wait')


        #     # Display the lib version at the bottom of the screen
        #     cvui.printf(BG, 600 - 80, 300 - 10,0.3, 0xCECECE, 'SOTracker v0.12')
        #     cvui.update()
                        
        #     cv.imshow('SOTracker:operation panel', BG)
        #     if cvui.button(BG,300,200,150,100, 'OK') or cvui.button(BG,450,200,150,100, 'Wait'):
        #         break                        
        #cvui.init('Tracking video')
        success, frame = cap.read()

        cv.imshow(display_name, frame)
        data_frame = pd.DataFrame(columns = 
            ('frame number','the RotatingAngle','the CornerAngle',
            'the VelocityAngle','the ObjectDirectionAngle','the velocity'))#初始化data的列名


        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, list, tuple)
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            self.initialize(frame, optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                # cv.putText(frame_disp, 'Select ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                #            1.5, (0, 0, 0), 1)
                cvui.printf(BG,10, 130, 0.4, 0xff0000, "Select ROI and press ENTER")
                cvui.update('SOTracker:operation panel')
                cv.imshow('SOTracker:operation panel', BG)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)

                init_state = [x, y, w, h]
                cvui.printf(BG,10, 150, 0.4, 0xffffff, "The init_state is:[%s,%s,%s,%s]",x,y,w,h)
                cvui.update('SOTracker:operation panel')
                cv.imshow('SOTracker:operation panel', BG)
                self.initialize(frame, init_state)
                break
        time_start=time.time()
        
        fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
        video_save = cv.VideoWriter(str(Outputvideofilepath)+now_time+'_video_angel.avi', fourcc, fps, (video_width, video_height)) #创建输出视频 
        #video_trajectory = cv.VideoWriter(str(Outputvideofilepath)+now_time+'video_trajectory.avi', fourcc, fps, (video_width, video_height)) #创建输出视频 

        while True:


            ret, frame = cap.read()

            if frame is None:
               break

            frame_disp = frame.copy()

            current_frame += 1

            cvui.window(BG,0,0,300,300, 'information')

            cvui.printf(BG,10, 30, 0.4, 0xffffff, "The frame rate is :%d",fps)
            cvui.printf(BG,10, 50, 0.4, 0xffffff, "The resolution is:%d * %d",video_width,video_height)
            cvui.printf(BG,10, 70, 0.4, 0xffffff, "The total frame is:%d",total_frame)
            cvui.printf(BG,10, 90, 0.4, 0xffffff, "Total time:%fs",video_time)
            cvui.printf(BG,10, 110, 0.4, 0x00ff00, "Current frame:%d",current_frame)
            
            cvui.checkbox(BG,310,40,'Only show basic information', basic)
            cvui.checkbox(BG,310,80,'Show trajectory', trajectory)
            cvui.checkbox(BG,310,120,'Show direction', direction)        
            cvui.checkbox(BG,310,160,'Show all the information in main window',all_info)

            cvui.button(BG,300,200,150,50, 'Start(Space/Enter)')
            cvui.button(BG,450,200,150,50, 'Reselect ROI(r)')
            cvui.button(BG,300,250,300,50, 'Quit(q)')


                # Display the lib version at the bottom of the screen
            cvui.printf(BG, 600 - 80, 300 - 10,0.3, 0xCECECE, 'SOTracker v0.12')

            #####################    
            # Draw box，rotat_img 是仿射变换
            state,poly,rotat_img = self.track(frame)
            state = [int(s) for s in state]

            #cv.imshow('img',rotat_img)
            poly = np.array(poly).astype(np.int32)

            polygon = poly.reshape((-1,1,2))
            #画出物体方向
            ls = poly.flatten().tolist()
            #最小矩形框各顶点坐标
            box0 = Point(ls[0],ls[1])
            box1 = Point(ls[2],ls[3])
            box2 = Point(ls[4],ls[5])
            box3 = Point(ls[6],ls[7])
            #水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边
            Counterclockwise_line = (ls[0],ls[1]) + (ls[6],ls[7])
            draw_half_short1,draw_half_short2,half_short1,half_short2,draw_half_long1,draw_half_long2,half_long1,half_long2 = Get_half_legth_point(box0,box1,box2,box3)

            #把视频中的物体长度与测量值进行比对，仅需要在第一帧进行计算
            if (current_frame)<=2 :
                ObjectLength_video = Getlen(Point(half_short1[0],half_short1[1]),Point(half_short2[0],half_short2[1]))
                PixelLength = ObjectLength_real / float(ObjectLength_video.getlen())#单位像素映射到现实中的长度

            Rotatingangle,Cor_Angle,Vel_Angle,dir_Angle,center1,center2,CenterPointListX,CenterPointListY,dir_true_side = Get_Angle (poly,half_short1,half_short2,half_long1,half_long2,X_zhou,Counterclockwise_line)               
            #画出轨迹
            tra_point = []
            if len(CenterPointListX)>2:

                #frame_tra = frame_disp.copy()
                # cv.namedWindow('trajectory', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
                # cv.resizeWindow('trajectory',720,480)
                # cv.moveWindow('trajectory',730,0)
                tra_x,tra_y = smoothing_base_bezier(CenterPointListX,CenterPointListY,k=0.3, closed=False)
                for i in range(0,len(tra_x)):
                    tra_point.append([tra_x[i],tra_y[i]])
                tra_point = np.array(tra_point,np.int32)#.reshape((int(len(tra_x)/2),2))
                tra_point = tra_point.reshape((-1,1,2))
                # cv.polylines(frame_tra,[tra_point],False,(255,255,255),thickness=3)
                # video_trajectory.write(frame_tra)
                # cv.imshow('trajectory',frame_tra)
                
                #Show_Guiji(CenterPointListX,CenterPointListY,video_width,video_height)

            if (abs(abs(dir_Angle)-abs(Vel_Angle)) != Cor_Angle):
    
                    dir_Angle = dir_Angle - 180

            
            Vel_value = Get_Vel(center1,center2,PixelLength,fps)
            
            data_frame = data_frame.append(pd.DataFrame({'frame number':[current_frame],
                                                          'the RotatingAngle':[Rotatingangle],
                                                          'the CornerAngle':[Cor_Angle],
                                                          'the VelocityAngle':[Vel_Angle],
                                                          'the ObjectDirectionAngle':[dir_Angle],
                                                          'the velocity':[Vel_value]}))


            #print(data_frame)
                
            #px.line(data_frame,x = 'time',y = 'the RotatingAngle').show()

            #把数据保存到excel文档中去
            data_frame.to_excel(str(Outputvideofilepath)+now_time+'_output.xls',index = False)

                #在视频上把信息打印出来
            font=cv.FONT_HERSHEY_SIMPLEX

            if trajectory[0] is True:
                cv.polylines(frame_disp,[tra_point],False,(255,255,255),thickness=3)

            if direction[0] is True:
                cv.putText(frame_disp,'The Rotatingangle is:'+str(Rotatingangle),(0,1050), font, 1,(125,125,125),4)
                cv.putText(frame_disp,'The angle of velocity is:'+str(Vel_Angle),(0,50), font, 1,(255,0,0),4)
                cv.putText(frame_disp,'The angle of object direction is:'+str(dir_Angle),(0,100), font, 1,(0,0,255),4)
                cv.putText(frame_disp,'The angle betweem V and OD is:'+str(Cor_Angle),(0,150), font, 1,(0,255,0),4)        
                cv.line(frame_disp,center1,center2,(255,0,0),3)
                            #画出物体运动方向
                if dir_true_side == 1:
                    cv.line(frame_disp,draw_half_long1,draw_half_long2,(0,0,255),3)
                elif dir_true_side == 0:                    
                    cv.line(frame_disp,draw_half_short1,draw_half_short2,(0,0,255),3)
                else:
                    break

            if all_info[0] is True:
                cv.putText(frame_disp,'The Rotatingangle is:'+str(Rotatingangle),(0,1050), font, 1,(125,125,125),4)
                cv.putText(frame_disp,'The angle of velocity is:'+str(Vel_Angle),(0,50), font, 1,(255,0,0),4)
                cv.putText(frame_disp,'The angle of object direction is:'+str(dir_Angle),(0,100), font, 1,(0,0,255),4)
                cv.putText(frame_disp,'The angle betweem V and OD is:'+str(Cor_Angle),(0,150), font, 1,(0,255,0),4)  
                cv.putText(frame_disp,str(current_frame),(video_width - 150,video_height - 50), font,2,(255,255,255),3)
                cv.polylines(frame_disp,[tra_point],False,(255,255,255),thickness=3)        
                cv.line(frame_disp,center1,center2,(255,0,0),3)
                            #画出物体运动方向
                if dir_true_side == 1:
                    cv.line(frame_disp,draw_half_long1,draw_half_long2,(0,0,255),3)
                elif dir_true_side == 0:                    
                    cv.line(frame_disp,draw_half_short1,draw_half_short2,(0,0,255),3)
                else:
                    break

 
            
            
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)
            cv.polylines(frame_disp,[polygon],True,(0,255,255),thickness=3)

            font_color = (0, 0, 0)
            # cv.putText(frame_disp, 'Tracking!', (video_width - 200, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press r to reselect', (video_width - 200, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press q to quit', (video_width - 200, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # Display the resulting frame
            cvui.printf(BG,10, 130, 0.4, 0xff0000, "Tracking ! Press r to reselect,Press q to quit")

            cv.imshow(display_name, frame_disp)
            # show_dataset = threading.Thread(target = Show_Data,args = (data_frame))
            # show_dataset.setDaemon(True)
            # show_dataset.start()
            video_save.write(frame_disp)
            cvui.update('SOTracker:operation panel')
            cv.imshow('SOTracker:operation panel', BG)
            key = cv.waitKey(1)
            if cv.waitKey(20) == 27 or cvui.button(BG,300,250,300,50, 'Quit(q)'):
                break
            elif key == ord('q') or cvui.button(BG,300,250,300,50, 'Quit(q)'):
                break
            elif key == ord('r') or cvui.button(BG,450,200,150,50, 'Reselect ROI(r)') :
                ret, frame = cap.read()
                frame_disp = frame.copy()

                # cv.putText(frame_disp, 'Select ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                #            (0, 0, 0), 1)
                cvui.printf(BG,10, 150, 0.4, 0xff0000, "Interrupt! Select ROI and press ENTER")
                cvui.update('SOTracker:operation panel')
                cv.imshow('SOTracker:operation panel', BG)
                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                cvui.printf(BG,10, 0, 0.4, 0xffffff, "The init_state is:[%s,%s,%s,%s]",x,y,w,h)
                cvui.update('SOTracker:operation panel')
                cv.imshow('SOTracker:operation panel', BG)
                self.initialize(frame, init_state)
            #yield data_frame
        # #plt.savefig(r'the_output/%s_'%now_time+'_guiji.png',dpi=500)
        time_end=time.time()
        print('Time cost = %fs' % (time_end - time_start))
        # plt.savefig(r"Outputvideofilepath+now_time+'_guiji.eps'",dpi=500)
        # #plt.savefig(r'the_output/%s_'%now_time+'_guiji.png',dpi=500)
        # plt.figure()
        # data_frame.plot(x = 'frame number' , y = ['the RotatingAngle','the CornerAngle','the VelocityAngle','the ObjectDirectionAngle'],title = 'Angle')
        # plt.legend()
        # #plt.show()
        # #plt.savefig(r'the_output/%s_'%now_time+'_angle.png',dpi=500)
        # plt.savefig(r"Outputvideofilepath+now_timee+'_angle.eps'",dpi=500)
        # #plt.savefig(r'the_output/%s_'%now_time+'_angle.png',dpi=500)
        # data_frame.plot(x = 'frame number' , y = ['the velocity'],title = 'Speed')
        # plt.legend()
        # #plt.show()
        # plt.savefig(r"Outputvideofilepath+now_time+'_velocity.eps'",dpi=500)
        # #plt.savefig(r'the_output/%s_'%now_time+'_velocity.png',dpi=500)

        cv.waitKey(0)
        #window.mainloop()
        # When everything done, release the capture
        video_save.release()
        cap.release()
        cv.destroyAllWindows()

        


    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reselect', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        # self.fig.canvas.manager.window.move(800, 50)
        self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (100, 50))

        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)

        if len(state) == 4:
            pred = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=2, edgecolor='r', facecolor='none')
        elif len(state) == 8:
            p_ = np.array(state).reshape((4, 2))
            pred = patches.Polygon(p_, linewidth=2, edgecolor='r', facecolor='none')
        else:
            print('Error: Unknown prediction region format.')
            exit(-1)

        self.ax.add_patch(pred)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

