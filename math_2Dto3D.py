import numpy as np


def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])

    return Ry.reshape([3,3])
    # return np.dot(np.dot(Rz,Ry), Rx)

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):

    dx = dimension[2] / 2 # length
    dy = dimension[0] * 2 # height
    dz = dimension[1] / 2 # width
  # height width length
    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners

# this is based on the paper. Math!
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
def calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray):
    #global orientation
    orient = alpha + theta_ray
    #print('orient:',orient)
    R = rotation_matrix(orient) # 为了方便通过loc+ori计算顶点坐标;将世界坐标系转化为相机坐标系
    #print('rotation_matrix:', R)
    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]
    #print('box_corners:',box_corners)
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system ; 原数据集中第一个是height（车高度），第二个是width（车两侧的距离），第三个是length(车头到车尾)
    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width
    #print('dx:',dx,'dy:',dy,'dz:',dz)
    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88): # 左边——车前右，右边——车前左
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92): # 左边——车后左，右边——车后右
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90): # 左边——车后左/右（当alpha<0,左；alpha>0,右）
        left_mult = -1
        right_mult = 1


    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1 #-1
    if alpha > 0:
        switch_mult = 1 #1

    # left and right could either be the front of the car or the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-2,0):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-2,0):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    """
    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        for j in (-1,1):
            for k in (-2,0):
                left_constraints.append([i * dx, k * dy, j * dz])
    for i in (-1,1):
        for j in (-1,1):
            for k in (-2,0):
                right_constraints.append([i * dx, k * dy, j * dz])
    """

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy*2, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, 0, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        #print('constraint:',constraint)
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd] # 4约束，对应上下左右 ，shape=4,3
        #print('X_array:',X_array)
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md] # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            # x_array is four constrains for up bottom left and right;
            # x is one point in world World coordinate system, .shape = 3
            M = M_array[row] # 一个对角是1的4*4方阵 .shape = 4*4

            # create M for corner Xx
            RX = np.dot(R, X) # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            M[:3,3] = RX.reshape(3) # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4

            M = np.dot(proj_matrix, M) # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

            #print('A:',A)
            #print('b:',b)
            #print("M:",M)
            #input()

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    #
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_X

# this is based on the paper. Math!
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
def calc_location_new(dimension, proj_matrix, box_2d, alpha, theta_ray):
    #global orientation
    orient = alpha + theta_ray
    #print('orient:',orient)
    R = rotation_matrix(orient) # 为了方便通过loc+ori计算顶点坐标;将世界坐标系转化为相机坐标系
    #print('rotation_matrix:', R)
    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]
    #print('box_corners:',box_corners)
    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    dx = dimension[2] / 2   # length
    dy = dimension[0] / 2   # height
    dz = dimension[1] / 2   # width
    #print('dx:',dx,'dy:',dy,'dz:',dz)
    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88): # 左边——车前右，右边——车前左
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92): # 左边——车后左，右边——车后右
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90): # 左边——车后左/右（当alpha<0,左；alpha>0,右）
        left_mult = -1
        right_mult = 1




    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1 #-1
    if alpha > 0:
        switch_mult = 1 #1

    # left and right could either be the front of the car or the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-2,0):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-2,0):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    """
    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        for j in (-1,1):
            for k in (-2,0):
                left_constraints.append([i * dx, k * dy, j * dz])
    for i in (-1,1):
        for j in (-1,1):
            for k in (-2,0):
                right_constraints.append([i * dx, k * dy, j * dz])
    """

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy*2, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, 0, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        #print('constraint:',constraint)
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd] # 4约束，对应上下左右 ，shape=4,3
        #print('X_array:',X_array)
        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md] # 4个对角为1的4*4方阵

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])

        # 对于其中某个约束/上/下/左/右
        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            # x_array is four constrains for up bottom left and right;
            # x is one point in world World coordinate system, .shape = 3
            M = M_array[row] # 一个对角是1的4*4方阵 .shape = 4*4

            # create M for corner Xx
            RX = np.dot(R, X) # 某边对应的某点在相机坐标系下的坐标, 维度3 .shape = 3,1
            M[:3,3] = RX.reshape(3) # 对角线是1，最后一列前三行分别是RX对应的相机坐标系下的长宽高 .shape = 4,4

            M = np.dot(proj_matrix, M) # 投影到二维平面的坐标（前三维）.shape = 3,4，前三列是project矩阵，最后一列是二维平面的x，y，1

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

            #print('A:',A)
            #print('b:',b)
            #print("M:",M)
            #input()

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # 3d bounding box dimensions

        # 3d bounding box corners
        x_corners = [dx , dx, -dx , -dx, dx, dx, -dx, -dx]
        y_corners = [0, 0, 0, 0, -2*dy, -2*dy, -2*dy, -2*dy]
        z_corners = [dz, -dz, -dz, dz, dz, -dz, -dz, dz]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # corners_3d = np.dot(R_x, corners_3d)
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + loc[0]
        corners_3d[1, :] = corners_3d[1, :] + loc[1]
        corners_3d[2, :] = corners_3d[2, :] + loc[2]
        corners_3d = np.transpose(corners_3d)
        N = corners_3d.shape[0]
        points = np.hstack([corners_3d, np.ones((N, 1))]).T
        points = np.matmul(proj_matrix, points)
        points /= points[2, :]
        points_2d = (points[0:2, :]).T
        #print(points_2d)
        included = 0
        print('box_corners',box_corners)
        for cor_point_2d in points_2d:
            if cor_point_2d[0]<xmax+3 and cor_point_2d[0]>xmin-3 and cor_point_2d[1]<ymax+3 and cor_point_2d[1]>ymin-3:
                included += 1
        # found a better estimation
        if error < best_error and included == 8:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array
            print('best_loc',best_loc)

    # return best_loc, [left_constraints, right_constraints] # for debugging
    #
    if best_loc is not None:
        best_loc = [best_loc[0], best_loc[1], best_loc[2]]
    return best_loc, best_X

def calc_theta_ray(img, box_2d, proj_matrix):
    width = img.shape[1]
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    dx = center - (width / 2)
    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
    angle = angle * mult
    return angle
