import serial
import time

def haptic_serial_open():    
    # open serial
    ser = serial.Serial('COM11', 115200, timeout=1)
    time.sleep(1)
    return ser

def haptic_interface( ser , haptic_feedback_set ):
    haptic_pos = [ 0, 0, 0, 0, 0, 0 ]
    haptic_vel = [ 0, 0, 0, 0, 0, 0 ]
    haptic_force_get = [0, 0]

    # send serial inputs
    # ser.write(bytes("10,10\n", 'utf-8'))
    ser.write( bytes( str( haptic_feedback_set[0] ) + ',' + str( haptic_feedback_set[1] ) + '\n' , 'utf-8') )
    time.sleep(0.01)

    isWaiting = ser.inWaiting()
    if ( isWaiting > 0):
        # print( ser.read(ser.inWaiting()) )

        try:
            data_str = ser.read(ser.inWaiting()).decode('ascii')
        except:
            return haptic_pos , haptic_vel , haptic_force_get
        # print(data_str, end='\n')
        time.sleep(0.01)

        haptic_encoder = data_str.split('\r\n')
        he_size = len( haptic_encoder )

        while( he_size > 2 ): # looking for valid data
            he_split = haptic_encoder[he_size-2].split(',') # skip the last entry since it is usually incomplete!
            if len( he_split ) == 14:
                lin_wire, rot_wire, lin_mcath, rot_mcath, lin_cath, rot_cath, vlin_wire, vrot_wire, vlin_mcath, vrot_mcath, vlin_cath, vrot_cath, lin_pwm, rot_pwm = haptic_encoder[he_size-2].split(',')
                haptic_pos = [ int(lin_wire), int(rot_wire), int(lin_mcath), int(rot_mcath), int(lin_cath), int(rot_cath) ]
                haptic_vel = [ int(vlin_wire), int(vrot_wire), int(vlin_mcath), int(vrot_mcath), int(vlin_cath), int(vrot_cath) ]
                haptic_force_get = [ int(lin_pwm), int(rot_pwm) ]
                # print( he_split ) # test
                break
            he_size = he_size - 1

    return haptic_pos , haptic_vel , haptic_force_get
     

if __name__ == "__main__": # test
    haptic_feedback_set = [ 0 , 0 ]

    haptic_serial = haptic_serial_open()
    while (True):
        haptic_pos , haptic_vel , haptic_force_get = haptic_interface( haptic_serial , haptic_feedback_set )
        print( [ haptic_pos , haptic_vel , haptic_force_get ] )
        time.sleep(0.01)


