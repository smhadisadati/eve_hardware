from urllib import response
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import serial
# import time
from datetime import datetime
# import matlab.engine
import re
# import os

from time import perf_counter, sleep
import autocath
import os
# import eve
import math

# import keyboard
# from inputs import get_key
import msvcrt


# colors
WHITE = '\033[37m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RED = '\033[31m'
CYAN = '\033[36m'
PURPLE = '\033[35m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

## parameters
# catheter parameters
CATH_NUMBERS = 1 # the number of catheters present in the video
l0_cath = [0, 0, 0] # inserted cath points

# controller parameters
image_frequency = 7.5 # controller frequency
# control2cnc = [[2, 5], [1, 4], [0, 3]] # [[translation1, rotation1]_innermost, ..., [translationN, rotationN]_outermost], trans and rot correspond to cnc board actuator number [XYZ_outer2inner rot_outer2inner]
control2cnc = [[0, 3], [1, 4], [0, 3]]
target_converg_range = 0.1 # target convergance radius to stop the controller

# cnc controller parameters:
CNC_CONNECT = False # should connect to the board or not
board_com = [ 'COM3' ] # port addresses for layer (top) 1 - 4 (bottom), my laptop
motor_code = [ 'X' , 'Y' , 'Z' , 'A' , 'B' , 'C' ] # motor gcodes, XYZ_outer2inner: translation, ABC_outer2inner: rotation, XA: for outermost
feedrate = 3000 # axis feedrate in mm/min (the commands are in mm)
dist_scale = [10, 10, 10, 1/9, 1/9, 1/9] # motor step scale factor to get mm and deg units
travel_step = 5 # mm, travel distance step
rot_step = 15 # deg, rotation angle step

# camera Parameters
CAM_CONNECT = False # use video feed or dummy video input 
res_scale = 2 # resolution scale
search_r = [ res_scale*3 , res_scale*2 ] # pixel search radius [ static region, dynamic]
track_converg_range = 2 # tracking onverges if repeatative measurments fall in a region with this length
vid_res = [ res_scale*240, res_scale*320 ]
gray_thresh = [ 120 , 130 ] # [ static region, catheter] - 0:dark - 255:white ], threshold for detection
# crop_percent = [ 0.1, 0.2 ] # crop factor from each side [x, y]


## initialize camera feed
# axes.append( fig.add_subplot(grid_rows, grid_cols, 3) )
print(WHITE + 'Opening the camera interface...')
dir = os.path.dirname(os.path.realpath(__file__))
if CAM_CONNECT: # live camera input
    cap = cv2.VideoCapture(0) # camera number starting with 0
else: # dummy video input
    cap = cv2.VideoCapture(dir + "\input.avi") # camera number starting with 0
font = cv2.FONT_HERSHEY_SIMPLEX # describe the type of font to be used.
ret, frame = cap.read() # Capture frame-by-frame
cv2.imshow('frame',frame) # Display the resulting frame 

output_vid_path = dir + "\output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
crop_res = int( np.min( vid_res ) ) # crop to square shape to match the controller
# crop_res = int( np.min( [vid_res[0]*(1-2*crop_percent[0]), vid_res[1]*(1-2*crop_percent[1])] ) ) # crop to square shape to match the controller
# cap_detected = cv2.VideoWriter(output_vid_path,fourcc, 30, (crop_res, crop_res))
cap_detected = cv2.VideoWriter(output_vid_path,fourcc, 30, (vid_res[0], vid_res[1]))


## initiate controller
cp = dir + "\controller\example\checkpoint4800089"
config = dir + "\controller\example\config.yml"
controller = autocath.RViMController1(
    checkpoint=cp,
    config=config,
    tracking_high=[1, 1],
    tracking_low=[-1, -1],
)


## functions
# video click event
xy_clicked = [0, 0] # clicked point
def click_event(event, x, y, flags, params):
    global xy_clicked, WHITE
    if event == cv2.EVENT_LBUTTONDOWN:
        xy_clicked[0] = x
        xy_clicked[1] = y
        print(WHITE + str(xy_clicked) ) # test

# keyboard input
def keyboard_input():
    if msvcrt.kbhit():
        keyboard_input = ord( msvcrt.getch() )
    else:
        keyboard_input = -1
    # print(keyboard_input) # test
    return keyboard_input

# continous frame aquisition
i_frame = -1 # frame counter
height = 0 # framr processing size
width = 0
xy_base = [0, 0] # catheter base
xy_range = [[0, 0], [0, 0]] # phantom range [x_min_max, y_min_max]
xy_range_mid_span = [[0, 0], [0, 0]] # phantom mid range and half span [x_mid_span, y_mid_span]
target_coords = [[0, 0]] # target point
target_pixel = [0, 0] # target point pixel coordinates
tracking = np.zeros((5,2)) * 2 - 1 # initialize backbone tracking points
xy_static = [] # record static points to neglect in racking
def image_processing(i_frame, target_coords):
    global height, width, crop_res, xy_static, xy_base, xy_clicked, xy_range, xy_range_mid_span, vid_res, target_pixel
    BREAK = False

    i_frame += 1
    ret, frame = cap.read()
    if ret == True:
        frame_rt = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_r = cv2.resize(frame_rt,(vid_res[0],vid_res[1]),fx=0,fy=0, interpolation = cv2.INTER_AREA) # INTER_AREA, INTER_CUBIC
        # image frame has [0,0] at top left, also it is frame[y,x], i.e. frame[height, width], not as standard x coordinat
        # frame_r = frame_rz[int((1-crop_percent[1])*vid_res[1]-crop_res):int((1-crop_percent[1])*vid_res[1]), int(vid_res[0]/2-crop_res/2):int(vid_res[0]/2+crop_res/2)] # resie frame
        frame_g = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frame_g, cv2.COLOR_GRAY2RGB) # for test with pre-recorded video

        # cv2.imshow('frame',frame) # test the original frame
        # cv2.imshow('frame',frame_r) # test the resulting resized frame 
        # cv2.imshow('frame',frame_g) # test the resulting grayscaled frame
        # cv2.waitKey() # test

        # for the first captured frame
        if i_frame == 0:
            # static data capture based on first frame
            frame_s = frame_g # static frame
            # print(frame_g) # test
            height, width = frame_g.shape # entrance from top
            print( [width,height] ) # test
            xy_static = np.zeros((height, width)) # static area: xy_static[y,x] = 1
            # print(xy_static) # test
            x_cath_0 = [0] * height # catheter pixels in first frame
            cv2.imshow('frame',frame_g) # get catheter base in the first frame            
            if CATH_NUMBERS > 0:
                print(YELLOW + "Click the catheter base (also the phnatom minimum height), then hit any key...")
                cv2.setMouseCallback('frame', click_event)
                cv2.waitKey() # wait for click
                xy_base[0] = xy_clicked[0]
                xy_base[1] = xy_clicked[1]
                xy_range[1][0] = xy_clicked[1] # set the phantom min height (y) range
                # print(xy_base) # test
                
                print(YELLOW + "Click the phantom maximum height (top) point, then hit any key...")
                cv2.setMouseCallback('frame', click_event)
                cv2.waitKey() # wait for click
                xy_range[1][1] = xy_clicked[1] # set the phantom max height (y) range
                
                print(YELLOW + "Click the phantom minimum width (left) point, then hit any key...")
                cv2.setMouseCallback('frame', click_event)
                cv2.waitKey() # wait for click
                xy_range[0][1] = xy_clicked[0] # set the phantom min width (x) range
                
                print(YELLOW + "Click the phantom maximum width (right) point, then hit any key...")
                cv2.setMouseCallback('frame', click_event)
                cv2.waitKey() # wait for click
                xy_range[0][0] = xy_clicked[0] # set the phantom max width (x) range

                # phantom range calculation
                xy_range_mid_span[0][0] = int( ( xy_range[0][0]  + xy_range[0][1] ) / 2 ) # x mid range
                xy_range_mid_span[0][1] = int( ( xy_range[0][0]  - xy_range[0][1] ) / 2 ) # x half span
                xy_range_mid_span[1][0] = int( ( xy_range[1][0]  + xy_range[1][1] ) / 2 ) # y mid range
                xy_range_mid_span[1][1] = int( ( xy_range[1][0]  - xy_range[1][1] ) / 2 ) # y half span
                
                print(GREEN + "Click the target point, then hit any key...")
                cv2.setMouseCallback('frame', click_event)
                cv2.waitKey() # wait for click
                # image frame has [0,0] at top left, but controller has [-1,-1] at bottom left with range -1:1
                target_pixel[0] = xy_clicked[0]
                target_pixel[1] = xy_clicked[1]
                target_coords[0][0] = ( xy_clicked[0] + 1 - xy_range_mid_span[0][0] ) / xy_range_mid_span[0][1] # scale x to -1:1 in the x-range
                target_coords[0][1] = - ( xy_clicked[1] + 1 - xy_range_mid_span[1][0] ) / xy_range_mid_span[1][1] # scale y to -1:1 in the y-range
                print(target_coords) # test

            prev_x_cath = xy_base[0]
            # for y in range(0, height) : # start from top
            # for y in range(height-1, -1, -1) : # start from bottom
            # for y in range(xy_base[1], -1, -1) : # start from bottom
            for y in range(xy_range[1][0], xy_range[1][1]-1, -1) : # start from bottom
                x_pix_n = 0 # counter for detected pixels of a catheter
                # for x in range(0, width) : # start from left
                # for x in range(width-1, -1, -1) : # start from right
                for x in range(xy_range[0][0], xy_range[0][1]-1, -1) : # start from right
                    if frame_g[y,x] < gray_thresh[0]: # grayscale intensity threshhold for static area
                        # print (frame_g[y,x]) # test
                        if CATH_NUMBERS > 0 and prev_x_cath != -1 and np.abs( x - prev_x_cath ) < search_r[0]: # close to catheter base
                            x_pix_n += 1
                            x_cath_0[y] += x
                            frame_s[y,x] = 0 # blacken the catheter pixel
                            # print([x,y,xy_static[y,x]]) # test
                        else:
                            xy_static[y,x] = 1 # white area is active
                            frame_s[y,x] = 255 # remove the static pixel
                if CATH_NUMBERS > 0 and x_pix_n != 0:
                    x_cath_0[y] = int( np.floor( x_cath_0[y] / x_pix_n ) ) # x_mean of detected pixles in the current height
                    frame_s[y,x_cath_0[y]] = 0 # blacken the catheter mean pixel
                    prev_x_cath = x_cath_0[y] # update cath mean x value for next height search
                    # print([x_cath_0[y], y, x_pix_n]) # test
                else:
                    prev_x_cath = -1 # no catheter found at this height

            print(YELLOW + "Static Pixles removed, hit any key to continue...")
            cv2.imshow('frame',frame_s) # test the resulting frame without static pixles
            cv2.waitKey()
            print(WHITE + "controller is running...")


        # print(xy_static[160,163]) # test

        # detect catheter starting from the entrance side
        # print(frame_g) # test
        scan_dir = 1 # scan direction
        # ix0 = 2 * height - 1 # cath index
        # ix0 = 2 * xy_base[1] - 1 # cath index
        ix0 = 4 * xy_range_mid_span[1][0] - 1 # cath index
        ix = ix0
        x_cath = [0] * ( ix0 + 1 ) # catheter pixels
        y_cath = [0] * ( ix0 + 1 ) # catheter pixels
        # initially search the entire entrance side
        # x_range = range(0, width) # start from left
        # x_range = range(width-1, -1, -1) # start from right
        x_range = range(xy_range[0][0], xy_range[0][1]-1, -1) # start from right
        y_range = range(xy_base[1]+search_r[1], xy_base[1]-3*search_r[1], -1)
        # for y in range(0, height) : # start from top
        # for y in range(height-1, -1, -1) : # start from bottom
        # y0 = height - 1 # start from the bottom
        # y0 = xy_base[1] # start from the bottom
        y0 = xy_range[1][0] # start from the bottom
        y = y0
        srch_dir = -1 # search direction
        dir_chngd_count = 0 # search direction alreaady changes
        while True:
            x_pix_n = 0 # counter for detected pixels of a catheter
            # print(x_range) # test
            for y in y_range:
                for x in x_range :
                    # print (frame_g[y,x]) # test intensity for gray scale
                    # print( [ xy_static[y,x], frame_g[y,x] ] ) # test
                    if xy_static[y,x] == 0 and frame_g[y,x] < gray_thresh[1]: # detect motion of dark pixels in dynamic area
                        x_pix_n += 1
                        x_cath[ix] += x
                        y_cath[ix] += y
                        frame_r[y,x,0] = 0 # blue: colour the original video
                        frame_r[y,x,1] = 0 # green
                        frame_r[y,x,2] = 255 # red
                        # print([ix,x,y,x_pix_n,xy_static[y,x]]) # test
            if x_pix_n == 0: # only continue if a catheter is detected at the current line
                break
            #     # print([ix,y,x_pix_n,dir_chngd_count,])
            #     if dir_chngd_count < 2 : # number of allowed change search direction
            #         dir_chngd_count += 1 # direction changes
            #         # srch_dir *= -1 # change search direction
            #         srch_dir *= (-1)^(np.mod(dir_chngd_count,2)) # change search direction every two direction change trials
            #         y += ( 1 * srch_dir ) # height for search in the opposite direction
            #         if y <= y0 and y > xy_range[1][1] - 1 : # still in the frame
            #             continue # repeat the search with opposite direction
            #         else:
            #             # print('outside of frame')
            #             # print(y) # tests
            #             break
            #     else: # both directions already checked
            #         # print('already searched both y directions')
            #         break
            x_cath[ix] = int( np.floor( x_cath[ix] / x_pix_n ) ) # x_mean of detected pixles in the current height
            y_cath[ix] = int( np.floor( y_cath[ix] / x_pix_n ) ) # x_mean of detected pixles in the current height
            if x_cath[ix] >= xy_range[0][0] : # out of right margin
                x_cath[ix] = xy_range[0][0]  - 1
                # print('outside of frame')
                break
            if x_cath[ix] < xy_range[0][1] : # out left margin
                x_cath[ix] = xy_range[0][1] 
                # print('outside of frame')
                break
            if y_cath[ix] >= xy_range[1][0] : # out of right margin
                 y_cath[ix] = xy_range[1][0]  - 1
                # print('outside of frame')
                # break
            if y_cath[ix] < xy_range[1][1] : # out left margin
                y_cath[ix] = xy_range[1][1] 
                # print('outside of frame')
                break
            # print( [x_cath[ix], y, x_pix_n] ) # test
            if ix != ix0 and np.sqrt( np.power(x_cath[ix]-x_cath[ix+1],2) + np.power(y_cath[ix]-y_cath[ix+1],2) ) < track_converg_range: # search end condition
                ix += 1 # last element is repeated
                break
            # update the search range for compuattional efficiency
            # x_range = range(x_cath[ix-1]-search_r[1], x_cath[ix-1]+search_r[1]) # start from left
            # x_range = range(x_cath[ix]+2*search_r[1], x_cath[ix]-2*search_r[1], -1) # start from right
            if ix == ix0 or x_cath[ix] == x_cath[ix+1]: # first instance or straight cath
                x_range = range(x_cath[ix]+2*search_r[1], x_cath[ix]-2*search_r[1], -1) # start from right
            elif x_cath[ix] > x_cath[ix+1]: # cath moved right
                x_range = range(x_cath[ix]+3*search_r[1], x_cath[ix]-search_r[1], -1) # start from right   
            else: # cath moved left
                x_range = range(x_cath[ix]+search_r[1], x_cath[ix]-3*search_r[1], -1) # start from right 
            if ix != ix0 and y_cath[ix] > y_cath[ix+1]: # cath moved down
                y_range = range(y_cath[ix]+3*search_r[1], y_cath[ix]-search_r[1], -1) # start from right   
            else: # first step or cath moved up
                y_range = range(y_cath[ix]+search_r[1], y_cath[ix]-3*search_r[1], -1) # start from right 
            # print(x_range) # test
            # color the catheter mean
            frame_r[y_cath[ix],x_cath[ix],0] = 0 # blue: colour the original video
            frame_r[y_cath[ix],x_cath[ix],1] = 0 # green
            frame_r[y_cath[ix],x_cath[ix],2] = 255 # red
            # update par.s for the next iteration
            # dir_chngd_count = 0 # reset direction change indicator
            ix -= 1 # search for the next cath pixel
            # y += srch_dir # cotnue search in the same direction
            # print( y ) # test
            # if y > y0 or y < xy_range[1][1] : # check to be still in the range
                # print('outside of frame')
                # print(y) # tests
                # break
            # print([ix,y,dir_chngd_count]) # test

        # display & save detected catheter
        frame_r[target_pixel[1],target_pixel[0],0] = 0 # blue: colour the target point
        frame_r[target_pixel[1],target_pixel[0],1] = 255 # green
        frame_r[target_pixel[1],target_pixel[0],2] = 0 # red
        cap_detected.write(frame_r)
        cv2.imshow('frame',frame_r) # Display the resulting resized frame 
        cv2.waitKey(1) # wait for the frAME TO BE SHOWN

    else: # video stream stopped
        tracking = np.zeros((1,2))
        BREAK = True
        return i_frame, target_coords, tracking, BREAK

    # collocate tracking pixels
    cath_pixel_no = ix0 - ix + 1 # number of found catheter pixels
    tracking = np.zeros((cath_pixel_no,2))
    i_track = 0 # tracking pixel counter
    for i in range(ix,ix0+1): # first element of "tracking" should be the tip pixel
        # image frame has [0,0] at top left, but controller has [-1,-1] at bottom left with range -1:1
        tracking[i_track][0] = ( x_cath[i] + 1 - xy_range_mid_span[0][0] ) / xy_range_mid_span[0][1] # it crops the width symmetrically w.r.t. the image centre
        tracking[i_track][1] = - ( y_cath[i] + 1  - xy_range_mid_span[1][0] ) / xy_range_mid_span[1][1] # it crops the height from the base
        i_track += 1
    # print(tracking) # test
    
    return i_frame, target_coords, tracking, BREAK
        
# send/receive gcode commands
def command(board_serials, board_no, command):
    global CNC_CONNECT        
    response = '' # default pos response
    
    if CNC_CONNECT:
        # print( YELLOW + 'board ' + str( board_no) + ' - ' + command )        
        board_serials[board_no].write(str.encode(command + '\r\n')) 
        # sleep(0.1)

        while True:
            line = board_serials[board_no].readline()
            if line == b'ok\n':
                break
            # print( BLUE + 'Board response: ' )
            # print( line )
            response = line

    return response

# get current position
def get_pos(board_serials):
    global CNC_CONNECT
    pos = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ] # position vector for all motors on 1st cnc board
    if CNC_CONNECT:
        for i_board in range(0,1):
            response = command(board_serials, i_board, 'M114 R')
            # command(board_serials, i_board, 'M400') # wait until the motion is finished
            pos_list = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response.decode('utf-8')) # find float numnbers in the line
            # print(pos_list)
            for i in range(0,6):
                pos[i_board][i] = float( pos_list[i] ) / dist_scale[i] # scale to mm and deg
    # print(pos) # test
    return pos

# pass line length values
def dist2gcode( board_serials , dist ):
    global feedrate
    command_str = [ 'G1' ]
    for i in range(0,6) : # itterate tendon#
        # command_str[ 0 ] += ' ' + motor_code[ i ] + str( int( dist[i]*dist_scale[i] ) ) # scale to mm and deg
        command_str[ 0 ] += ' ' + motor_code[ i ] + str( ( dist[i]*dist_scale[i] ) ) # scale to mm and deg
        
    for i in range(0,1) : # call the cnc drivers
        command_str[i] += ( ' F' + str(feedrate) ) # add velocity info
        print( BLUE + board_com[i] + ': ' + command_str[i] )
        command( board_serials, i , "G91" ) # set to relative positioning
        command( board_serials, i , command_str[i] )
    
# print on cv2 frame
def printCV( print_text ):
    cv2.putText(frame, print_text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow('video', frame)

# controller call
def controller_call(board_serials,controller,CATH_NUMBERS,l0_cath,t_prev_cmd,INITIATE_CONTROLLER,target_coords,tracking): 
    ## call ontroller        
    pos = get_pos(board_serials) # get current pos
    device_lengths_inserted = [0] * CATH_NUMBERS # initiate & update actuator state
    for i in range(0,CATH_NUMBERS):
        device_lengths_inserted[i] = l0_cath[i] + pos[0][i]

    # reset controller
    if INITIATE_CONTROLLER: # initialize the controller
        controller.reset(
            tracking,
            target_coords,
            device_lengths_inserted,
            [1, 1],
            [-1, -1],
        )

    # update controller
    UPDATED_INPUT = 0 # control is not updates
    dstate = [0, 0, 0, 0, 0, 0] # inputs for the system [X_outermost_trans, Y-trans, Z_innermost-trans, then rotations]
    # action = [0] * CATH_NUMBERS # action is state velocity [[translation1, rotation1], [translation2, rotation2], ...]
    t_duration = perf_counter() - t_prev_cmd
    t_duration_d = 1 / image_frequency
    if t_duration > t_duration_d: # only update the input after the controller set frequency
        UPDATED_INPUT = 1
        t_prev_cmd = perf_counter()
        # this is the important step:
        action = controller.step(tracking, target_coords, device_lengths_inserted)
        for i_cath in range(0,CATH_NUMBERS):
            dstate[control2cnc[i_cath][0]] = np.floor( action[i_cath*2+0] * t_duration_d ) # relative displacement in mm
            dstate[control2cnc[i_cath][1]] = np.floor( action[i_cath*2+1] * t_duration_d * 360 / 3.14 ) # relative rotation in deg
        # print(dstate) # test
    
    return dstate, UPDATED_INPUT, t_prev_cmd

# check if the motion is finished
def motion_completed(board_serials,abs_encoder):
    pos = get_pos(board_serials) # get current pos
    # print(WHITE + str(pos)) # test
    # print(WHITE + str(abs_encoder)) # test
    
    for i in range(0,6):
        if abs( pos[0][i] - abs_encoder[i] ) < 0.1 :
            MOTOIN_STATE = True # remain True only if all axes are reached
        else:
            MOTOIN_STATE = False # returns false and terminate the code with first not reached axis
            break    
    return MOTOIN_STATE

# stop the controller if the target is reached
def target_reached(target_coords,tracking):
    TARGET_REACHED = False
    error = np.sqrt( np.power(target_coords[0][0]-tracking[0][0],2) + np.power(target_coords[0][1]-tracking[0][1],2) )
    if error < target_converg_range:
        TARGET_REACHED = True
    return TARGET_REACHED

## initialize cnc board
board_serials = [0]
if CNC_CONNECT:
    for i in range( 0 , 1 ) : # start the boards
        print(WHITE + 'Starting board on port ' + board_com[i] + '...')
        board_serials[i] = serial.Serial(board_com[i], 250000)
        sleep(2)
        command(board_serials, i, "G92 X0 Y0 Z0 A0 B0 C0") # set current (home) position


# ## initialize matlab
# eng = matlab.engine.start_matlab()
# eng.addpath(r'C:\Hadi\Postdoc\2_KCL\5. Spinoff\1. STREAM - SCS + Kaspar\3. Model\0. CC',nargout=0)


## program main loop
dist = [0, 0, 0, 0, 0, 0] # travel distance/rotation reset
abs_encoder = [0, 0, 0, 0, 0, 0]

AI_CONTROL = False # AI or manula control?
UPDATED_INPUT = 0 # is input updated? 1 to calculate the default pose
t_prev_cmd = 0 # time previous command sent

key = 1 # initialize key value to show the instructions in the first run
key_cash = 1 # cashed key value to execute the latest input when the system is still executing the latest ommand
while(True):            

    # call image tracking
    i_frame, target_coords, tracking, BREAK = image_processing(i_frame, target_coords)
    if BREAK: # stream ended
        break

    if key_cash != -1: # only shows this after a key is pressed to avoid repeatadly printing this
        # Instructions:      
        print(GREEN + 'Press:')
        # print(CYAN + '- ''h or main-right or start'' for homing at the system start state,')
        # print(CYAN + '- ''f or main left'' for homing at the fixe end or limit switch locations,')
        # print(CYAN + '- ''c or L2'' for manual gcode command,')
        print(CYAN + '- ''[w,s,a,d] or left axis, [i,k,l,j] or right axis, [t,g,h,f] or main directions'' for moving the outermost to innermost tube (up-down for translation, left-right for trotation),')
        print(CYAN + '- ''[c,v] or [LT, RT]'' for translating all tubes,')
        print(CYAN + '- ''[e,r] or [LB, RB]'' for rotating all tubes,')
        print(CYAN + '- ''m or A'' to switch to AI controller (resets the absolute and axis encoders!),')
        print(CYAN + '- ''x or X'' to switch to manual controller,')
        print(CYAN + '- ''q or Back'' to quit the program!')

    # keyboard inputs    
    # key_cash = cv2.waitKey(33) # does not run in parallel, misses inputs
    # key_cash = ord( keyboard.read_key() ) # parallel run but blocking the code for input
    key_cash = keyboard_input() # non blocking keyboard input
    if key_cash != -1: # new cashed command is arrived
        key = key_cash

    if key == ord('q'): # quit the program
        break
                
    # wait until previous input motion is finished
    if motion_completed(board_serials,abs_encoder): # or True: # Previous motion is completed
        sleep(0.1)

        if AI_CONTROL and target_reached(target_coords,tracking):
            for i in range(0,3): # print multiple times
                print(YELLOW +  'Target Reached! Yay \:D/')
            # break # if you want to stop the controller
            AI_CONTROL = False # switch back to manual to home the system or TODO: select a new target

        # call controller
        if AI_CONTROL:
            dist, UPDATED_INPUT, t_prev_cmd = controller_call(board_serials,controller,CATH_NUMBERS,l0_cath,t_prev_cmd,INITIATE_CONTROLLER,target_coords,tracking)
            # print([UPDATED_INPUT, dist]) # test
            INITIATE_CONTROLLER = False # don't initiate the controller for the next itterations
            if key == ord('x'): # quit the program
                print(WHITE + 'Switching to Manual controller... ')
                AI_CONTROL = False
                continue 
        else:
            dist = [0, 0, 0, 0, 0, 0] # travel distance/rotation reset
            UPDATED_INPUT = 0

            if key != -1:
                # print(WHITE + 'key pressed: ') # keys: w119 s115 d100 a97 i105 k107 l108 j106 o111 u117
                # print(key) # test keys: w119 s115 d100 a97 i105 k107 l108 j106 o111 u117
                
                if key == ord('m'): # quit the program
                    print(WHITE + 'Switching to AI controller... ')
                    AI_CONTROL = True
                    INITIATE_CONTROLLER = True # to initiate the controller in the next itteration
                    abs_encoder = [0, 0, 0, 0, 0, 0] # reset the absolute encoder
                    command( board_serials, 0 , "G92 X0 Y0 Z0 A0 B0 C0") # reset the axis positions
                    continue 

                elif key in { ord('n') }: # homing at pre-tensioned locations
                    print(WHITE + 'homing at pre-tensioned position in progress...')  
                    abs_encoder = [0, 0, 0, 0, 0, 0] # reset the absolute encoder         
                    for i in range(0,1) : # move all axis to the home position
                        print(WHITE +  'Homing motors on board ' + board_com[i] )
                        command( board_serials, i , "G90") # set to absolute positioning
                        command( board_serials, i , 'G1 X0 Y0 Z0 A0 B0 C0' ) # home at the pre-tensioned state
            
                elif key == ord('w'):
                    dist[0] += travel_step
                    UPDATED_INPUT = 1
                elif key == ord('s'):
                    dist[0] -= travel_step
                    UPDATED_INPUT = 1
                    
                elif key == ord('t'):
                    dist[1] += travel_step
                    UPDATED_INPUT = 1
                elif key == ord('g'):
                    dist[1] -= travel_step
                    UPDATED_INPUT = 1

                elif key == ord('i'):
                    dist[2] += travel_step
                    UPDATED_INPUT = 1
                elif key == ord('k'):
                    dist[2] -= travel_step
                    UPDATED_INPUT = 1

                elif key == ord('a'):
                    dist[3] += rot_step
                    UPDATED_INPUT = 1
                elif key == ord('d'):
                    dist[3] -= rot_step
                    UPDATED_INPUT = 1

                elif key == ord('f'):
                    dist[4] += rot_step
                    UPDATED_INPUT = 1
                elif key == ord('h'):
                    dist[4] -= rot_step
                    UPDATED_INPUT = 1
                    
                elif key == ord('j'):
                    dist[5] += rot_step
                    UPDATED_INPUT = 1
                elif key == ord('l'):
                    dist[5] -= rot_step
                    UPDATED_INPUT = 1
                    
                elif key == ord('v'):
                    dist[0] += travel_step
                    dist[1] += travel_step
                    dist[2] += travel_step
                    UPDATED_INPUT = 1
                elif key == ord('c'):
                    dist[0] -= travel_step
                    dist[1] -= travel_step
                    dist[2] -= travel_step
                    UPDATED_INPUT = 1
                    
                elif key == ord('e'):
                    dist[3] += rot_step
                    dist[4] += rot_step
                    dist[5] += rot_step
                    UPDATED_INPUT = 1
                elif key == ord('r'):
                    dist[3] -= rot_step
                    dist[4] -= rot_step
                    dist[5] -= rot_step
                    UPDATED_INPUT = 1
            
        key = -1 # acknowledge that the last input is executed
        if UPDATED_INPUT:            
            print(WHITE +  'dist = ')
            print(dist)
            dist2gcode( board_serials , dist ) # relative motion
            for i in range(0,6): # record motors' absolute position
                abs_encoder[i] += dist[i]
            

## release the camera capture 
print(WHITE + 'Closing down the camera feed...')
cap.release()
cap_detected.release()
cv2.destroyAllWindows()


## treminate cnc board
for i in range(0,1) : # move all axis to the home position
    # print(WHITE +  'Homing motors on board ' + board_com[i] )
    # command( board_serials, i , "G90") # set to absolute positioning
    # command( board_serials, i , 'G1 X0 Y0 Z0 A0 B0 C0' ) home system upon exit
    sleep(2)
    print(WHITE +  'Closing down serial port ' + board_com[i])
if CNC_CONNECT:
        board_serials[i].close()