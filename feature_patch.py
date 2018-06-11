import numpy as np


import utils
import Autoencoder as ae

def get_directions_traj(xs,ys):

    list_orientation = []

    dx = float(xs[0]) - float(xs[int(len(xs)/2)])
    dy = float(ys[0]) - float(ys[int(len(ys)/2)])

    ##take care of points that are too close
    if dx <=1 and dy <= 1:
        dx = float(xs[0]) - float(xs[int(len(xs)-1)])
        dy = float(ys[0]) - float(ys[int(len(ys)-1)])

        ##I round the angle to 8 key directions: 0 left, 45 up-left, 90 up, 135 up-right, 180 right, -135 down-right, -90 down, -45 down-left
        list_orientation.append(myround_ofdirection(atan2(dy, dx) / pi * 180))
    else:

        ##I round the angle to 8 key directions: 0 left, 45 up-left, 90 up, 135 up-right, 180 right, -135 down-right, -90 down, -45 down-left
        list_orientation.append(myround_ofdirection(atan2(dy, dx) / pi * 180))

        dx = float(xs[int(len(xs)/2)]) - float(xs[len(xs)-1])
        dy = float(ys[int(len(ys)/2)]) - float(ys[len(ys)-1])

        ##I round the angle to 8 key directions: 0 left, 45 up-left, 90 up, 135 up-right, 180 right, -135 down-right, -90 down, -45 down-left
        t = myround_ofdirection(atan2(dy, dx) / pi * 180)
        list_orientation.append(t)

    return list_orientation


def create_grid(xs, ys, size_mask, directions, scene):
    ## The reason i dont use the already existing function is because this grid doesnt have to be centered at one point but
    ## the corner has to start next to the first point of the trajectory
    margin = 1

    first_chunck_direction = directions[0]

    ###Drawing the central rect and the one next to it

    if first_chunck_direction == -45:

        up_right_corner = [xs[0] + margin, ys[0] - margin]

        first_rect = mplPath.Path(
            np.array([[up_right_corner[0] - size_mask, up_right_corner[1] + size_mask],
                      [up_right_corner[0] - size_mask, up_right_corner[1]],
                      up_right_corner,
                      [up_right_corner[0], up_right_corner[1] + size_mask]]))



    elif first_chunck_direction == -135:

        up_left_corner = [xs[0] - margin, ys[0] - margin]

        first_rect = mplPath.Path(
            np.array([[up_left_corner[0], up_left_corner[1] + size_mask],
                      up_left_corner,
                      [up_left_corner[0]+size_mask, up_left_corner[1]],
                      [up_left_corner[0]+size_mask, up_left_corner[1] + size_mask]]))




    elif first_chunck_direction == 45:

        down_right_corner = [xs[0] + margin, ys[0] + margin]

        first_rect = mplPath.Path(
            np.array([[down_right_corner[0]-size_mask,down_right_corner[1]],
                      [down_right_corner[0]-size_mask,down_right_corner[1]-size_mask],
                      [down_right_corner[0],down_right_corner[1]-size_mask],
                      down_right_corner]))



    elif first_chunck_direction == 135:

        down_left_corner = [xs[0] - margin, ys[0] + margin]

        first_rect = mplPath.Path(
            np.array([down_left_corner,
                      [down_left_corner[0],down_left_corner[1]-size_mask],
                      [down_left_corner[0]+size_mask,down_left_corner[1]-size_mask],
                      [down_left_corner[0]+size_mask,down_left_corner[1]]]))



    ####if direction is straight, i center the rect on the points

    elif first_chunck_direction == -90:

        top_left_corner = [xs[0] - int(size_mask/2),ys[0]-margin]
        top_right_corner = [xs[0] + int(size_mask/2),ys[0]-margin]

        first_rect = mplPath.Path(
            np.array([[top_left_corner[0],top_left_corner[1]+size_mask],
                      top_left_corner,
                      top_right_corner,
                      [top_right_corner[0],top_right_corner[1]+size_mask]]))


    elif first_chunck_direction == 90:

        down_left_corner = [xs[0] - int(size_mask/2),ys[0] + margin]
        down_right_corner = [xs[0] + int(size_mask/2),ys[0] + margin]

        first_rect = mplPath.Path(
            np.array([down_left_corner,
                      [down_left_corner[0],down_left_corner[1] - size_mask],
                      [down_right_corner[0],down_right_corner[1] - size_mask],
                      down_right_corner]))

    elif first_chunck_direction == 180 or first_chunck_direction == -180 :
        top_left_corner = [xs[0] - margin,ys[0] - int(size_mask/2)]
        down_left_corner = [xs[0] - margin,ys[0] + int(size_mask/2)]

        first_rect = mplPath.Path(
            np.array([down_left_corner,
                      top_left_corner,
                      [top_left_corner[0]+size_mask, top_left_corner[1]],
                      [top_left_corner[0]+size_mask,top_left_corner[1]+size_mask]]))

    elif first_chunck_direction == 0:

        top_right_corner = [xs[0] + margin, ys[0] - int(size_mask / 2)]
        down_right_corner = [xs[0] + margin, ys[0] + int(size_mask / 2)]

        first_rect = mplPath.Path(
            np.array([[down_right_corner[0] - size_mask, down_right_corner[1]],
                      [top_right_corner[0] - size_mask, top_right_corner[1]],
                      top_right_corner,
                      down_right_corner]))




    list_rect = [first_rect]#,side_rect,second_side_rect]


    # for rect in list_rect:
    #     cv2.rectangle(scene, (int(rect.vertices[1][0]), int(rect.vertices[1][1])),
    #                   (int(rect.vertices[3][0]), int(rect.vertices[3][1])), (0, 0, 0))
    #
    for i_p, p in enumerate(xrange(len(xs))):
        cv2.circle(scene, (xs[p], ys[p]), 3, (255, 0, 0), -1)
    #
    # cv2.imshow('scene', scene)
    # cv2.waitKey(0)

    return list_rect,scene


def transform_traj_in_pixel_activation(rect_list, x_untilNow, y_untilNow, size_mask, step):
    #size_mask= 18
    #a = [0, 0]
    #b = [size_mask - 5, size_mask - 5]
    #step = np.sqrt(np.power((b[0] - a[0]), 2) + np.power((b[1] - a[1]), 2))


    traj_features = []
    orig_points = []


    for rect in rect_list:

        points_in_mask = []
        map(lambda ci: points_in_mask.append([int(x_untilNow[ci]), int(y_untilNow[ci])]) if rect.contains_point(
            (int(x_untilNow[ci]), int(y_untilNow[ci]))) else False, xrange(len(x_untilNow)))


        origin_mask = [rect.vertices[1][0],rect.vertices[1][1]]

        mask_img = np.zeros((size_mask, size_mask), dtype=np.uint8)
        mask_matrix = np.zeros((size_mask, size_mask))


        if len(points_in_mask) >= 2:
            for i in xrange(len(points_in_mask) - 1):
                distance_metric_value = np.sqrt(np.power(points_in_mask[i + 1][0] - points_in_mask[i][0], 2) + np.power(
                    points_in_mask[i + 1][1] - points_in_mask[i][1], 2)) / step

                ##convert img points to mask coordinate systems
                x_1 = points_in_mask[i + 1][0] - origin_mask[0]
                y_1 = points_in_mask[i + 1][1] - origin_mask[1]
                x = points_in_mask[i][0] - origin_mask[0]
                y = points_in_mask[i][1] - origin_mask[1]

                ##get all pixels lying on the line that pass between two points
                points_on_line = img_proc.createLineIterator(np.array([[x], [y]]), \
                                                                np.array([[x_1], [y_1]]), mask_img)

                ##fill these pixel values with average distance between two points
                for p in points_on_line:
                    ##if we want to display on img

                    mask_img[int(p[1]),int(p[0])] = 255
                    # # print distance_metric_value
                    if int(p[1]) + 1 < size_mask - 1 and int(p[0]) < size_mask - 1:
                        ##right
                        mask_img[int(p[1]) + 1, int(p[0])] = 255
                        ##left
                        mask_img[int(p[1]) - 1, int(p[0])] = 255
                        ##up
                        mask_img[int(p[1]), int(p[0]) - 1] = 255
                        ##down
                        mask_img[int(p[1]), int(p[0]) + 1] = 255


                    ##real value
                    mask_matrix[int(p[1]), int(p[0])] = distance_metric_value
                    # print distance_metric_value
                    if int(p[1]) + 1 < size_mask - 1 and int(p[0]) < size_mask - 1:
                        ##right
                        mask_matrix[int(p[1]) + 1, int(p[0])] = distance_metric_value
                        ##left
                        mask_matrix[int(p[1]) - 1, int(p[0])] = distance_metric_value
                        ##up
                        mask_matrix[int(p[1]), int(p[0]) - 1] = distance_metric_value
                        ##down
                        mask_matrix[int(p[1]), int(p[0]) + 1] = distance_metric_value




            ##if we want to display on img
            # mask_img = cv2.resize(mask_img, (80, 80))
            # cv2.imshow('scene', mask_img)
            # cv2.waitKey(0)

        else:

            mask_matrix = mask_matrix.reshape((1, -1))


        ##store final matrix
        if len(traj_features) > 0:
            traj_features = np.vstack((traj_features, mask_matrix.reshape((1, -1))))
        else:
            traj_features = mask_matrix.reshape((1, -1))

        ##store original points
        if len(orig_points)>0:
            orig_points = np.vstack((orig_points,mask_img.reshape((1,-1))))
        else:
            orig_points = mask_img.reshape((1,-1))


    return traj_features,orig_points


def create_vector_activations_layer_2(directions, list_activations, list_orig_points):


    matrix_activation_layer2 = np.zeros((9,len(list_activations[0])))
    matrix_orig_points_layer2 = np.zeros((9,list_orig_points[0].shape[1]))

    ##the first position is determined by the direction
    first_dir = directions[0]
    if first_dir == 0:
        matrix_activation_layer2[5] = list_activations[0]
        matrix_orig_points_layer2[5] = list_orig_points[0]

    elif first_dir == 45:
        matrix_activation_layer2[8] = list_activations[0]
        matrix_orig_points_layer2[8] = list_orig_points[0]

    elif first_dir == 90:
        matrix_activation_layer2[7] = list_activations[0]
        matrix_orig_points_layer2[7] = list_orig_points[0]

    elif first_dir == 135:
        matrix_activation_layer2[6] = list_activations[0]
        matrix_orig_points_layer2[6] = list_orig_points[0]

    elif first_dir == 180 or first_dir == -180:
        matrix_activation_layer2[3] = list_activations[0]
        matrix_orig_points_layer2[3] = list_orig_points[0]

    elif first_dir == -45:
        matrix_activation_layer2[2] = list_activations[0]
        matrix_orig_points_layer2[2] = list_orig_points[0]

    elif first_dir == -90:
        matrix_activation_layer2[1] = list_activations[0]
        matrix_orig_points_layer2[1] = list_orig_points[0]

    elif first_dir == -135:
        matrix_activation_layer2[0] = list_activations[0]
        matrix_orig_points_layer2[0] = list_orig_points[0]

    ##whereas the second position in the grid is always the center
    matrix_activation_layer2[4] = list_activations[1]
    matrix_orig_points_layer2[4] = list_orig_points[1]

    ## the third position is again determined by the direction
    third_dir = directions[2]

    if third_dir == 0:
        matrix_activation_layer2[3] = list_activations[2]
        matrix_orig_points_layer2[3] = list_orig_points[2]

    elif third_dir == 45:
        matrix_activation_layer2[0] = list_activations[2]
        matrix_orig_points_layer2[0] = list_orig_points[2]

    elif third_dir == 90:
        matrix_activation_layer2[1] = list_activations[2]
        matrix_orig_points_layer2[1] = list_orig_points[2]

    elif third_dir == 135:
        matrix_activation_layer2[2] = list_activations[2]
        matrix_orig_points_layer2[2] = list_orig_points[2]

    elif third_dir == 180 or first_dir == -180:
        matrix_activation_layer2[5] = list_activations[2]
        matrix_orig_points_layer2[5] = list_orig_points[2]

    elif third_dir == -45:
        matrix_activation_layer2[6] = list_activations[2]
        matrix_orig_points_layer2[6] = list_orig_points[2]

    elif third_dir == -90:
        matrix_activation_layer2[7] = list_activations[2]
        matrix_orig_points_layer2[7] = list_orig_points[2]

    elif third_dir == -135:
        matrix_activation_layer2[8] = list_activations[2]
        matrix_orig_points_layer2[8] = list_orig_points[2]


    matrix_activation_layer2 = matrix_activation_layer2.reshape((1,len(list_activations[0])*9))
    matrix_orig_points_layer2 = matrix_orig_points_layer2.reshape((1,list_orig_points[0].shape[1]*9))

    return matrix_activation_layer2,matrix_orig_points_layer2



def traj_to_patch(x_f,y_f,size_mask):

    max_step = np.sqrt(np.power(((size_mask - 3) - 0), 2) + np.power(((size_mask - 3) - 0), 2)) * 1.3
    a = [0, 0]
    b = [size_mask - 5, size_mask - 5]
    activation_norm_value = np.sqrt(np.power((b[0] - a[0]), 2) + np.power((b[1] - a[1]), 2))

    patches = []
    orig_points_history = []


    for i_p in xrange(1, len(x_f)):

        ##accumulate traj points until the distance between the first point and current point is enough for the grid
        d = np.sqrt(((x_f[i_p] - first_point_traj[0]) ** 2) + ((y_f[i_p] - first_point_traj[1]) ** 2))

        ##if the distance is enough compute the grid starting from the first point until the current point
        if abs(d - max_step) < 7:

            xs_untilNow = x_f[start_t:i_p]
            ys_unilNow = y_f[start_t:i_p]

            ##get directions of the traj chunck using first and last point
            # direction = get_direction_traj([x_f[start_t],y_f[start_t]],[x_f[i_p],y_f[i_p]])
            directions = get_directions_traj(xs_untilNow, ys_unilNow)
            if directions[0] == -180: directions[0] = 180
            #directions_history.append(directions[0])

            ##create grid of layer 2 according to the direction of the trajectory
            rects_in_grid, temp_scene = create_grid(xs_untilNow, ys_unilNow, size_mask,
                                                                               directions, temp_scene)
            # rects_history.append(rects_in_grid[0])

            ##compute the features from traj chuncks in rect
            traj_features, orig_points = transform_traj_in_pixel_activation(rects_in_grid, xs_untilNow, ys_unilNow,
                                                                            size_mask,
                                                                            activation_norm_value)
            orig_points_history.append(orig_points)
            patches.append(traj_features)


            ##update the beginning of the trajectory
            start_t = i_p - 1
            first_point_traj = [x_f[start_t], y_f[start_t]]

    return patches, orig_points_history


def traj_to_AE_encoding(x_f,y_f,size_mask,):

    max_step = np.sqrt(np.power(((size_mask - 3) - 0), 2) + np.power(((size_mask - 3) - 0), 2)) * 1.3
    a = [0, 0]
    b = [size_mask - 5, size_mask - 5]
    activation_norm_value = np.sqrt(np.power((b[0] - a[0]), 2) + np.power((b[1] - a[1]), 2))

    patches = []
    orig_points_history = []

    for i_p in xrange(1, len(x_f)):

        ##accumulate traj points until the distance between the first point and current point is enough for the grid
        d = np.sqrt(((x_f[i_p] - first_point_traj[0]) ** 2) + ((y_f[i_p] - first_point_traj[1]) ** 2))

        ##if the distance is enough compute the grid starting from the first point until the current point
        if abs(d - max_step) < 7:

            xs_untilNow = x_f[start_t:i_p]
            ys_unilNow = y_f[start_t:i_p]

            ##get directions of the traj chunck using first and last point
            # direction = get_direction_traj([x_f[start_t],y_f[start_t]],[x_f[i_p],y_f[i_p]])
            directions = get_directions_traj(xs_untilNow, ys_unilNow)
            if directions[0] == -180: directions[0] = 180
            # directions_history.append(directions[0])

            ##create grid of layer 2 according to the direction of the trajectory
            rects_in_grid, temp_scene = create_grid(xs_untilNow, ys_unilNow, size_mask,
                                                    directions, temp_scene)
            # rects_history.append(rects_in_grid[0])

            ##compute the features from traj chuncks in rect
            traj_features, orig_points = transform_traj_in_pixel_activation(rects_in_grid, xs_untilNow, ys_unilNow,
                                                                            size_mask,
                                                                            activation_norm_value)
            orig_points_history.append(orig_points)
            patches.append(traj_features)

            activation = ae.encode_features_using_AE_layer1_cluster_activation(traj_features, 'layer2')

            activation_history.append(activation)

            ## define which features we want to extract

            if len(activation_history) == 3:

                # cv2.imshow('scene', temp_scene)
                # cv2.waitKey(0)

                ##extract features for AE layer2
                matrixt_activation_l2, original_points_l2 = create_vector_activations_layer_2(
                    directions_history, activation_history, orig_points_history)

                ##save activations for layer2
                if len(matrix_activations) > 0:
                    matrix_activations = np.vstack((matrix_activations, matrixt_activation_l2))
                else:
                    matrix_activations = matrixt_activation_l2

                ##save activations for layer2
                if len(matrix_orig_points) > 0:
                    matrix_orig_points = np.vstack((matrix_orig_points, original_points_l2))
                else:
                    matrix_orig_points = original_points_l2

                orig_points_history = []
                directions_history = []
                activation_history = []

            ##update the beginning of the trajectory
            start_t = i_p - 1
            first_point_traj = [x_f[start_t], y_f[start_t]]

    return patches, orig_points_history





