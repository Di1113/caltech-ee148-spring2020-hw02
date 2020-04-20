import os
import numpy as np
import json
from PIL import Image

# util function used in compute_convolution
def padimage(image, kernel, stride):
    s = stride
    (ir, ic) = image.shape
    (kr, kc) = kernel.shape
    # padding for rows
    pr = kr + (ir - 1)*s - ir
    # padding on the right side edge 
    prleft = int((pr - (pr % 2))/2  )
    # padding for columns
    pc = kc + (ic - 1)*s - ic
    # padding on the top side edge 
    pctop = int((pc - (pc % 2))/2)
    # create a matrix with padded image size 
    (newimg_r, newimg_c) = (ir+pr, ic+pc)
    newimg = np.zeros((newimg_r, newimg_c))
    # fill in the center of the matrix with old image pixel values 
    newimg[prleft:prleft+ir, pctop:pctop+ic] = image 
    return newimg, newimg_r, newimg_c

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    # only takes in rgb images 
    assert n_channels == 3 

     # init heatmap grid value 
    corrsum = 0 #correlation sum of kernel and scanned image window 
    # init an empty 3d heatmap 
    heatmap3d = np.zeros(I.shape)

    # pad image to have full convolution (result heatmap size is the same as the image)
    pddimg_red, pddimg_red_row, pddimg_red_col = padimage(I[:,:,0], T[:,:,0], stride)
    pddimg_grn, pddimg_grn_row, pddimg_grn_col = padimage(I[:,:,1], T[:,:,1], stride)
    pddimg_ble, pddimg_ble_row, pddimg_ble_col = padimage(I[:,:,2], T[:,:,2], stride)

    pddimg = np.dstack((pddimg_red, pddimg_grn, pddimg_ble))

    assert (pddimg_red_row == pddimg_grn_row) and (pddimg_red_col == pddimg_grn_col)
    assert (pddimg_red_row == pddimg_ble_row) and (pddimg_red_col == pddimg_ble_col)
    assert (pddimg_grn_row == pddimg_ble_row) and (pddimg_grn_col == pddimg_ble_col)

    # store padded image rol and col size to ir and ic 
    (ir, ic) = (pddimg_red_row, pddimg_red_col)
    # store kernel rol and col size to kr and kc 
    (kr, kc) = T[:, :, 0].shape

    # init covolution starting position 
    (sr, sc) = (0, 0)
    stride_rowspace = ir-kr+1 #row space for kernel to move 
    stride_colspace = ic-kc+1 #column space for kernel to move 

    for ch in range(0,3):
        # scan the image from left to right, stepping with stride size 
        for i in range(0, stride_rowspace, stride):
            # scan the image from top to bottom, stepping with stride size 
            for j in range(0, stride_colspace, stride):
                # compute convolution of the scanned image window and given kernel 
                for k in range(0, kr):
                    ii = k + i #image window row position
                    # perform dot product/convolution row by row 
                    rsum = np.dot(pddimg[ii,j:(j+kc),ch], T[k,:,ch])
                    # sum up all row sums 
                    corrsum += rsum 
                # finished computing one kenerl-size correlation, store in new image array 
                heatmap3d[i//stride,j//stride,ch] = corrsum
                corrsum = 0 #clear current keneral-image correlation sum for next scan 
    heatmap = np.sum(heatmap3d, axis=2)
    # print(heatmap.shape)
    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6

    # You may use multiple stages and combine the results
    T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_Path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
