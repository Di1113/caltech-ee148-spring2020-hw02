import os
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# from numba import jit

# ---- testrun0-17-----------
# T_img = Image.open('redlightspot.jpg')

# # convert template to numpy array and copy to T 
# T = np.asarray(T_img).copy()[:, :, :3 ]
# template_width, template_height, dc = T.shape
# print(T.shape)
# timg = Image.fromarray(T)

# bin_size = 3
# output_width = template_width // bin_size #column pixels 
# output_height = template_height // bin_size #row pixels
# T = T.reshape((output_height, bin_size, output_width, bin_size, 3)).max(3).max(1)

# ---- testrun18 ------------
# T_img = Image.open('redcircle.jpg')

# ---- testrun19 ------------
T_img = Image.open('smallredcircle.jpg')

# weights for red cicle color channels
red_weight = 5 
grn_weight = -1
ble_weight = -1  
# convert template to numpy array and copy to T 
T = np.asarray(T_img).copy()[:, :, :3 ]
T_red = (T[:,:,2]*(-1)+255)/255 * red_weight
T_grn = (T[:,:,2]*(-1)+255)/255 * grn_weight
T_ble = (T[:,:,2]*(-1)+255)/255 * ble_weight
T = np.dstack((T_red, T_grn, T_ble))


template_width, template_height, dc = T.shape


# print(T)
# print(np.ravel(T[:,:,0]).shape)
# r = np.ravel(T[:,:,0])
# r = np.array([5,6,7])
# print(r)
# print(sum(r * r))
# print(np.dot(np.ravel(T[:,:,0]), np.ravel(T[:,:,0])))
# ?? why does dot not work? 
# T_dotsum_r = np.dot(np.ravel(T[:,:,0]), np.ravel(T[:,:,0]))
# T_dotsum_g = np.dot(np.ravel(T[:,:,1]), np.ravel(T[:,:,1]))
# T_dotsum_b = np.dot(np.ravel(T[:,:,2]), np.ravel(T[:,:,2]))

r = np.ravel(T[:,:,0])
g = np.ravel(T[:,:,1])
b = np.ravel(T[:,:,2])
T_dotsum_r = np.sum(r * r)
T_dotsum_g = np.sum(g * g)
T_dotsum_b = np.sum(b * b)
print(T_dotsum_r, T_dotsum_g, T_dotsum_b)
T_dotsum = np.sum([T_dotsum_r, T_dotsum_g, T_dotsum_b])
print(T_dotsum)
# util function used in compute_convolution
# @jit
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

# @jit
def compute_convolution(I, T, stride):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    print("Log: start convoluting...")

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

    print("Log: finished convoluting...")

    return heatmap

# @jit
def predict_boxes(heatmap, box_height, box_width):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    print("Log: start predeciting boxes...")

    output = []

    # 1. sort and find the top 10 max pixel value as bounding box center  

    topn = 1
    # convert 2d -> 1d for finding max 
    heatmap1d = heatmap.flatten() 

    # ------- testrun1 -------------
    # # testrun1: finding top 10 pixel value after convolution 
    # #     and perferm softmax on these 10 values for confidence score 
    # # finding top n values' indices 
    # maxindices = np.argpartition(heatmap1d, -topn)[-topn:]
    # # sort these indices by descending values 
    # maxindices = maxindices[np.argsort(-heatmap1d[maxindices])]
    # # convert indices 1d -> 2d  
    # maxindices = np.unravel_index(maxindices, heatmap.shape)

    # # get the top n max values 
    # topnmax = heatmap[maxindices]
    # print("debug: topnmax:")
    # print(topnmax)

    # # assign confidence score using softmax 
    # # sum(topscores) == 1 
    # topnmax -= np.max(topnmax) # to avoid overflow 
    # topscores = np.exp(topnmax) / np.sum(np.exp(topnmax), axis=0)


    # ------- testrun2 -------------
    # # testrun2: perferm softmax on all pixel values and pick the top 10 values  
    # heatmap1d -= np.max(heatmap1d) # to avoid overflow 
    # topscores = np.exp(heatmap1d) / np.sum(np.exp(heatmap1d), axis=0)

    # # finding top n values' indices 
    # maxindices = np.argpartition(topscores, -topn)[-topn:]
    # # sort these indices by descending values 
    # maxindices = maxindices[np.argsort(-topscores[maxindices])]

    # topscores = topscores[maxindices]
    # print("debug: 10 topscores:")
    # print(topscores)
    # # convert indices 1d -> 2d  
    # maxindices = np.unravel_index(maxindices, heatmap.shape)
    # # get the top n max values 
    # topnmax = heatmap[maxindices]
    # print("debug: values of topscores:")
    # print(topnmax)

    # ------- testrun3 -------------
    # # testrun3: find max pixel value, add a base offset, 
    # #   use max pixel value and base offset to be divided by top 10 pixel values to 
    # #   produce confidence score 
    # # finding top n values' indices 
    # maxindices = np.argpartition(heatmap1d, -topn)[-topn:]
    # # sort these indices by descending values 
    # maxindices = maxindices[np.argsort(-heatmap1d[maxindices])]
    # # convert indices 1d -> 2d  
    # maxindices = np.unravel_index(maxindices, heatmap.shape)
    # # get the top n max values 
    # topnmax = heatmap[maxindices]
    # offset = 1000000
    # maxvalue = topnmax[0]
    # topscores = topnmax / (maxvalue + offset)

    # ------- testrun4 -------------
    # changing stride to 20 

    # ------- testrun5 -------------
    # # changing stride to 60, changing baseoffset so that top box has 80% confidence 
    # # finding top n values' indices 
    # maxindices = np.argpartition(heatmap1d, -topn)[-topn:]
    # # sort these indices by descending values 
    # maxindices = maxindices[np.argsort(-heatmap1d[maxindices])]
    # # convert indices 1d -> 2d  
    # maxindices = np.unravel_index(maxindices, heatmap.shape)
    # # get the top n max values 
    # topnmax = heatmap[maxindices]
    # maxvalue = topnmax[0]
    # base = maxvalue / 0.8 
    # topscores = topnmax / base 

    # ------- testrun6 -------------
    # changing stride to 200

    # ------- testrun7 -------------
    # take max value from image, ie one bounding box per image 
    # changing stride to 20, confidence base to 100000000
    maxindices = np.argmax(heatmap1d)
    maxindices = np.unravel_index(maxindices, heatmap.shape)
    maxvalue = heatmap[maxindices]
    # base = 100000000 


    # ------- testrun8 -------------
    # changing kernel to red light spot, confidence base to 50000000
    # base = 50000000 
    # topscores = maxvalue / base 

    # ------- testrun9 -------------
    # changing kernel to red circle,  confidence base to 500000000
    # base = 800000000 
    # topscores = maxvalue / base 

    # ------- testrun10 -------------
    # changing kernel back to red light spot, confidence score to be a pixel 
    # value divided by sum of total pixel values plus an offset 
    # pixel values picked from top 2000 max values divided into 10 
    # groups and select each group's top 
    # # finding top n values' indices 

    # ------- testrun11 -------------
    # changing stride to 1

    # ------- testrun12 -------------
    # changing topn to 10000, sumratio_offset to 0.05

    # ------- testrun13 -------------
    # changing topn to 20000, sumratio_offset to 0.09
    
    # ------- testrun14 -------------
    # downsampling template from 24 x 24 to 8 x 8     

    # ------- testrun15 -------------
    # changing topn to 10000

    # ------- testrun16 -------------
    # changing topn to 50000, subtopgroup to 10

    # ------- testrun17 -------------
    # changing topn to 100000, subtopgroup to 10

    # ------- testrun18 -------------
    # changing kernel to contrived 3d circle image with weights
    # of 5, -1, -1 on red, green and blue channels 

    # ------- testrun19 -------------
    # changing topn to 10000, changing kernel to a smaller circle template
    topn = 10000 
    maxindices = np.argpartition(heatmap1d, -topn)[-topn:]
    # sort these indices by descending values 
    maxindices = maxindices[np.argsort(-heatmap1d[maxindices])]
    # convert indices 1d -> 2d  
    subtop_group = 10
    maxindices = np.reshape(maxindices, (subtop_group,topn//subtop_group))
    subtop_n = maxindices[:, 0]
    subtop_n = np.unravel_index(subtop_n, heatmap.shape)

    # get the selected n max values 
    topnmax = heatmap[subtop_n]
    print("debug: topnmax:")
    print(topnmax)

    # assign confidence score
    # sum(topscores) == 1 
    sumratio_offset = 0.09
    print("diff btw max and sum: %d" % (np.sum(topnmax) - topnmax[0]))
    topscores = topnmax / (sumratio_offset * np.sum(topnmax) + topnmax[0])
    print("debug: topscores:")
    print(topscores)


    # for testrun7-9
    # center_row, center_col = maxindices

    # tl_row = center_row -  box_height // 2 
    # tl_col = center_col - box_width // 2 
    # br_row = tl_row + box_height 
    # br_col = tl_col + box_width 
    # score = topscores
    # # add the box and confidence score in output[]  
    # output.append([tl_row,tl_col,br_row,br_col, score])

    # for testrun0-6 and 18    
    center_rows, center_cols = subtop_n
    centerpts = list(zip(center_rows, center_cols))
    for i in range(len(centerpts)):
        pt = centerpts[i]
        tl_row = pt[0] -  box_height // 2 
        tl_col = pt[1] - box_width // 2 
        br_row = tl_row + box_height 
        br_col = tl_col + box_width 
        score = topscores[i]
        # add the box and confidence score in output[]  
        output.append([tl_row,tl_col,br_row,br_col, score])

    print("Log: finished predicting boxes...")

    return output


def detect_red_light_mf(I, I_name):
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
    print("Log: start picking boxes...")

    # T_img = Image.open('redlight.png')

    # # convert template to numpy array and copy to T 
    # T = np.asarray(T_img).copy()

    stride = 1
    heatmap = compute_convolution(I, T, stride)
    # template_width, template_height, dc = T.shape
    output = np.array(predict_boxes(heatmap, template_height, template_width))
    # get the confidence scores 
    confid_list = np.array(output)[:,4]
    threshold = 0.5 
    # keep boxes with confidence score higher than threshold 
    # note: output and confid_list's descrips of the bbs are aligned 
    output = output[np.where(confid_list>threshold)]
    output = output.tolist()
    print("debug: output boxes:")
    print(output)
    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    print("Log: finihsed picking boxes...")

    saveimg_wbox(output, I, I_name)

    print("Log: saved picked boxes to image...")

    return output

def saveimg_wbox(boxes, image_array, imagename):
    rimg = Image.fromarray(image_array)
    for box in boxes: 
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        print("debug: bounding box coordinates: (%d, %d), (%d, %d)" % (x0, y0, x1, y1))
        # draw bounding box on original image
        draw = ImageDraw.Draw(rimg)
        draw.rectangle([x0, y0, x1, y1], fill=None, outline=(255, 0, 0))

    # save image to destinated folder 
    save_path = './testimages'
    os.makedirs(save_path, exist_ok=True)
    resfn = 'test_%s' % (imagename)
    rimg.save(os.path.join(save_path, resfn))

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../../data/RedLights2011_Medium'

# load splits: 
split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    icount = len(file_names_train) - i 
    print("log: detecting image %s, %d images left." % (str(file_names_train[i]), icount))

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    # preds_train[file_names_train[i]] = detect_red_light_mf(I)
    preds_train[file_names_train[i]] = detect_red_light_mf(I, file_names_train[i])

    if(i % 5 == 0):
        # save preds (overwrites any previous predictions!)
        ff = open(os.path.join(preds_path,('preds%s.json' % (str(i)))),'w+')
        with ff:
            json.dump(preds_train,ff)

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
