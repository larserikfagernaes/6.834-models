"""
Various utiltiy and testing functions
"""
from IPython.display import display, HTML, clear_output
import numpy as np
from numpy.linalg import  norm
import torch
# from RMPC import RMPC
from nose.tools import assert_equal, assert_almost_equal

def test_ok():
    """If execution gets to this point, print out a happy message."""
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")

def check_horizontal_kernel(kernel,img):
    canvas = cv2.imread('/content/modelsSemanticSegmentation/imgs/uniform_rectilinear_grid_2d.png',0)
    kernel_horizontal = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
    kernel_horizontal = kernel_horizontal/norm(kernel_horizontal)
    img_horizontal_feature = cv2.filter2D(canvas,-1,kernel_horizontal)

    #The kernel should be normalized
    assert_equal(norm(kernel/norm(kernel)-kernel_horizontal/norm(kernel_horizontal))<1e-6,True)
    assert_equal(norm(img_horizontal_feature-img)<1e-6,True)
    pass

def check_vertical_kernek(kernel,img):
    canvas = cv2.imread('/content/modelsSemanticSegmentation/imgs/uniform_rectilinear_grid_2d.png',0)
    kernel_vertical = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]).T
    kernel_vertical = kernel_vertical/norm(kernel_vertical)
    img_vertical_feature = cv2.filter2D(canvas,-1,kernel_vertical)


    assert_equal(norm(kernel/norm(kernel)-kernel_vertical/norm(kernel_vertical))<1e-6,True)
    assert_equal(norm(img_vertical_feature-img)<1e-6,True)

def check_numberOfObjects(number):
    assert_equal(number,9)

class Box_check:
    def __init__(self, xTopLeft, yTopLeft, xBottomRight, yBottomRight):
        self.xTopLeft = xTopLeft
        self.yTopLeft = yTopLeft 
        self.xBottomRight = xBottomRight 
        self.yBottomRight = yBottomRight 
    def __eq__(self, others):
        if self.xTopLeft == others.xTopLeft \
            and self.yTopLeft == others.yTopLeft \
            and self.xBottomRight == others.xBottomRight \
            and self.yBottomRight == others.yBottomRight:
            return True
        else:
            return False
class Pixel:
    def __init__(self, x, y):
        self.x = x 
        self.y = y 

def check_enclosingBox(func,img):
    assert_equal(Box_check(0,86,211,198), func(img,Pixel(0,88)))
    assert_equal(Box_check(0,66,20,85), func(img,Pixel(3,70)))
    assert_equal(Box_check(0,118,468,263), func(img,Pixel(210,150)))
    assert_equal(Box_check(208,92,250,200), func(img,Pixel(220,150)))
    assert_equal(Box_check(0,0,466,108), func(img,Pixel(150,0)))
    assert_equal(Box_check(248,104,468,213), func(img,Pixel(400,150)))

def check_personOnRoad(func,img):
    assert_equal(func(img),True)
