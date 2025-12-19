import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/roboworks/Visual-Localization/colcon_ws/install/landmarks_detection'
