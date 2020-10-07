__author__ = 'mhgou'
__version__ = '1.0'

from graspnetAPI import GraspNet

# GraspNetAPI example for checking the data completeness.
# change the graspnet_root path

if __name__ == '__main__':

    ####################################################################
    graspnet_root = '/home/gmh/graspnet'  ### ROOT PATH FOR GRASPNET ###
    ####################################################################

    g = GraspNet(graspnet_root, 'kinect', 'all')
    if g.check_data_completeness():
        print('Check for kinect passed')


    g = GraspNet(graspnet_root, 'realsense', 'all')
    if g.check_data_completeness():
        print('Check for kinect passed')