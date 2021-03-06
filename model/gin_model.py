"""YOLO_v3 Model Defined in Keras."""

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import cv2
from model.language_backbone import build_nlp_model
from model.bert import build_bert
from model.utils import *
from model.visual_backbone import *
from model.garan import global_attentive_reason_unit

def simple_fusion(F_v,f_q,dim=1024):
    """
    :param F_v: visual features (N,w,h,d)
    :param f_q: GRU output (N,d_q)
    :param dim: project dimensions default: 1024
    :return: F_m: simple fusion of Fv and fq (N,w,h,d)
    """
    out_size = K.int_shape(F_v)[1]
    #Fv project (use darknet_resblock get better performance)
    F_v_proj=darknet_resblock(F_v,dim//2)
    #fq_project
    f_q_proj=Dense(dim,activation='linear')(f_q)
    f_q_proj=LeakyReLU(alpha=0.1)(BatchNormalization()(f_q_proj))
    f_q_proj=Lambda(expand_and_tile,arguments={'outsize':out_size})(f_q_proj)
    #simple elemwise multipy
    F_m=Multiply()([F_v_proj,f_q_proj])
    F_m = LeakyReLU(alpha=0.1)(BatchNormalization()(F_m))
    return F_m
def adaptive_feature_selection(featrues,f_q):
    """
    :param featrues: a list of low level featrues with the same shape  [(N,w,h,d),(N,w,h,d)...]
    :param f_q:  GRU output (N,q_d)
    :return: F_v: Adaptive visual features  (N,w,h,d)
    """
    featrues=[Lambda(K.expand_dims)(x) for x in featrues]
    featrues=Concatenate()(featrues)    #visual featrues (N,w,h,d,3)
    shape=K.int_shape(featrues)

    #use fq to predict weights for visual features
    featrue_weights=Dense(3,activation='softmax')(f_q) # feature weights (N,3)
    featrues=Reshape([shape[1]*shape[2]*shape[3],shape[4]])(featrues)

    F_v=Dot([-1,1])([featrues,featrue_weights]) # dot
    F_v=Reshape([shape[1],shape[2],shape[3]])(F_v) #(N,w,h,d)
    return F_v
def featrues_normlization(feats,down_scales,out_dims):
    shape=K.int_shape(feats)
    feats=DarknetConv2D_BN_Leaky(shape[-1], (3,3),strides=down_scales,padding='same')(feats) #featrue downscale
    feats=DarknetConv2D(out_dims,(1,1))(feats) #featrue dims normalization
    return feats

def make_multi_modal_regression(featrue_feats,f_q,out_filters,config):
    """
    :param featrue_feats:
    :param num_filters:
    :param out_filters:
    :param f_q:
    :return:
    """
    y0=featrue_feats[0]
    #low level featrue normalize
    y1=featrues_normlization(featrue_feats[1],down_scales=2,out_dims=config['jemb_dim'])
    y2=featrues_normlization(featrue_feats[2],down_scales=4,out_dims=config['jemb_dim'])
    # adaptive feature selection
    y= adaptive_feature_selection([y0,y1,y2],f_q)
    #cascade_global_attentive reason unit
    F_v, E = global_attentive_reason_unit(y, f_q)
    #simple fusion
    y = simple_fusion(F_v,f_q)
    #regression
    y=  DarknetConv2D(out_filters, (1,1))(y)
    return y,E

def yolo_body(inputs,q_input, num_anchors,config):
    """

    :param inputs:  image
    :param q_input:  word embeding
    :param num_anchors:  defalt : 3
    :return:  regresion , attention map
    """
    """Create Multi-Modal YOLO_V3 model CNN body in Keras."""
    assert config['backbone'] in ["darknet","vgg"]
    if config['backbone'] =="darknet":
        darknet = Model(inputs, darknet_body(inputs))
        Fv = [darknet.output, darknet.layers[152].output, darknet.layers[92].output]
    else:
        Fv = vgg16(inputs)

    if config['use_bert']:
        q_input,fq=build_bert(config['bert_path'],poolings=['POOL_NSP'],output_layer_num=4)
    else:
        fq=build_nlp_model(q_input,config['rnn_hidden_size'],config['bidirectional'],config['rnn_drop_out'],config['lang_att'])  #build nlp model for fusion

    y,E = make_multi_modal_regression(Fv,fq, num_anchors*5,config)

    return Model([inputs,q_input], [y,E])

def yolo_head(feats, anchors, input_shape, calc_loss=False,att_map=None):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    if att_map is not  None:
        seg_map=K.tile(att_map,[1,1,1,3])
        seg_map=K.expand_dims(seg_map,axis=-1)
        box_confidence = K.sigmoid(feats[..., 4:5])#*.8+seg_map*.2  ##denote if add attention score to confidence score
    else:
        box_confidence = K.sigmoid(feats[..., 4:5])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, input_shape, image_shape,att_map=None):
    '''Process Conv layer output'''
    if att_map is not  None:
        # print('recalcu conf')
        box_xy, box_wh, box_confidence = yolo_head(feats, anchors, input_shape,att_map=att_map)
    else:
        box_xy, box_wh, box_confidence = yolo_head(feats, anchors, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence
    box_scores = K.reshape(box_scores, [-1, 1])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = 1
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[0,1,2]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    seg_maps_ = K.sigmoid(yolo_outputs[1])
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], input_shape, image_shape,seg_maps_)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []

    class_boxes = tf.boolean_mask(boxes, mask[:, 0])
    class_box_scores = tf.boolean_mask(box_scores[:, 0], mask[:, 0])
    nms_index = tf.image.non_max_suppression(
        class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
    class_boxes = K.gather(class_boxes, nms_index)
    class_box_scores = K.gather(class_box_scores, nms_index)
    boxes_.append(class_boxes)
    scores_.append(class_box_scores)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)

    return boxes_, scores_,seg_maps_
def yolo_eval_v2(yolo_outputs_shape,
                 anchors,
                 image_shape,
                 max_boxes=1,
                 score_threshold=.1,
                 iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    inputs = [K.placeholder(shape=(1, ) + yolo_outputs_shape[1:])]  #change list to single
    num_layers = len(inputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[0,1,2]] # default setting
    input_shape = K.shape(inputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(inputs[l],
            anchors[anchor_mask[l]], input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    class_boxes = tf.boolean_mask(boxes, mask[:, 0])
    class_box_scores = tf.boolean_mask(box_scores[:, 0], mask[:, 0])
    nms_index = tf.image.non_max_suppression(
        class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
    class_boxes = K.gather(class_boxes, nms_index)
    class_box_scores = K.gather(class_box_scores, nms_index)
    boxes_.append(class_boxes)
    scores_.append(class_box_scores)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    boxes_=K.reshape(boxes_,[-1,4])
    scores_=K.reshape(scores_,[-1,1])
    return boxes_, scores_, inputs
def preprocess_true_boxes(true_boxes, input_shape, anchors): #10.1 delete classfy
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')   #[32,1,4] xmin ,ymin ,xmax ,ymax
    input_shape = np.array(input_shape, dtype='int32')  #[416,416]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  #xcenter,ycenter
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  #width,height
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]    # true box normalize to (0,1)

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5),
        dtype='float32') for l in range(num_layers)]       # shape is [[32,52,52,3,5],[32,26,26,3,5],[32,13,13,3,5]]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0
    iou_gt_true = np.zeros((m, grid_shapes[0][0], grid_shapes[0][1], 1), dtype='float32')
    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    iou_gt_true[b, ...] = np.expand_dims(
                        generate_iou_groundtruth(grid_shapes[l], [j,i],
                                                                [true_boxes[b,t,3]*float(grid_shapes[l][0]),true_boxes[b,t,2]*float(grid_shapes[l][1])])
                        ,-1)

    return y_true,iou_gt_true

def generate_iou_groundtruth(grid_shapes,true_anchor,true_wh):
    """
    :param grid_shapes:   widths and heights for generation (w,h)
    :param true_anchor:  anchor's x and y (x,y)
    :param true_wh:  anchor's width and height (w,h) use for calculate iou
    :return: general iou distribution without any hyperparameter for attention loss
    """
    def cal_single_iou(box1, box2):
        smooth = 1e-7
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max((yi2 - yi1), 0.) * max((xi2 - xi1), 0.)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = (inter_area + smooth) / (union_area + smooth)
        return iou
    IMAGE_WIDTH = grid_shapes[0]
    IMAGE_HEIGHT = grid_shapes[1]


    t_w,t_h=true_wh
    t_x,t_y=true_anchor

    gt_box=[t_x-t_w/2,t_y-t_h/2,t_x+t_w/2,t_y+t_h/2]

    iou_map=np.zeros([IMAGE_WIDTH,IMAGE_HEIGHT])
    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            iou_map[i,j]=cal_single_iou(gt_box,[max(i-t_w/2,0.),max(j-t_h/2,0.),min(i+t_w/2,IMAGE_WIDTH),min(j+t_h/2,IMAGE_HEIGHT)])


   #  plt.figure()
   #  plt.imshow(iou_map, plt.cm.gray)
   # # plt.imshow(ground_truth, plt.cm.gray)
   #  plt.show()
    return iou_map

def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou



def yolo_loss(args, anchors, ignore_thresh=.5,seg_loss_weight=0.1, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    print(args)
    yolo_outputs = args[:1]
    att_map=args[1]
    y_true = args[2:3]
    gt_map=args[3]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[0,1,2]] ##due to deleting 2 scales  change [[6,7,8], [3,4,5], [0,1,2]] to [[0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))   # x32 is original size
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)] #3 degree scales output
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        # true_class_probs = y_true[l][..., 5:]  #... ==????

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        def smooth_L1(y_true, y_pred,sigma=3.0):
            """ Create a smooth L1 loss functor.

            Args
                sigma: This argument defines the point where the loss changes from L2 to L1.

            Returns
                A functor for computing the smooth L1 loss given target data and predicted data.
            """
            sigma_squared = sigma ** 2

            # compute smooth L1 loss
            # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
            #        |x| - 0.5 / sigma / sigma    otherwise
            regression_diff = y_true - y_pred
            regression_diff = K.abs(regression_diff)
            regression_loss = tf.where(
                K.less(regression_diff, 1.0 / sigma_squared),
                0.5 * sigma_squared * K.pow(regression_diff, 2),
                regression_diff - 0.5 / sigma_squared
            )
            return regression_loss
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * smooth_L1(raw_true_wh,raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        att_loss = K.binary_crossentropy(gt_map, att_map, from_logits=True)
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        att_loss = K.sum(att_loss) / mf * 0.5
        loss += xy_loss + wh_loss + confidence_loss+att_loss

    return  K.expand_dims(loss, axis=0)
