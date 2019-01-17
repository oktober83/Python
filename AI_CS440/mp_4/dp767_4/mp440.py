import inspect
import sys
import math

k = 0.00001
num_labels = 10
P_y = []
P_xi_y = []
feature_list = []

'''
Raise a "not defined" exception as a reminder
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)


'''
Extract 'basic' features, i.e., whether a pixel is background or
foreground (part of the digit)
'''
def extract_basic_features(digit_data, width, height):
    features=[]
    for row in range(0,height):
        for column in range(0, width):
            if digit_data[row][column] == 0:
                features.append(False)
            else:
                features.append(True)
    return features

'''
Extract advanced features that you will come up with 
'''
def extract_advanced_features(digit_data, width, height):
    features=[]

    # Feature 1: Number of Non-Zero Pixels in Diagonals
    diagonals1 = [0 for i in range(0,width+height-1)]
    diagonals2 = [0 for i in range(0,width+height-1)]
    offset = width - 1
    for row in range(0,height):
        for column in range(0, width):
            if digit_data[row][column] != 0:
                diagonals1[row-column+offset] += 1
                diagonals2[row+column-offset] += 1

    # Feature 2: 4 Horizontal-Celled Projection
    n = width
    m = height
    kh = 2
    q = n/kh
    V = [0 for ii in range(0, m*kh)]
    for i in range(0, m):
        for j in range(0, n):
            if digit_data[i][j] != 0:
                V[i + int(m * math.floor((j)/q))] = 1

    # Feature 3: Vertical & Horizontal Crossings

    horizontal = [0 for ii in range(0,m)]
    vertical = [0 for ii in range(0,n)]

    for row in range(0, height):
        if digit_data[row][0] == 0:
            prev = 0
        else:
            prev = 1
        for column in range(1, width):
            if digit_data[row][column] == 0:
                curr = 0
            else:
                curr = 1
            if curr != prev:
                horizontal[row] += 1
                curr = prev

    for column in range(1, width):
        if digit_data[0][column] == 0:
            prev = 0
        else:
            prev = 1
        for row in range(0, height):
            if digit_data[row][column] == 0:
                curr = 0
            else:
                curr = 1
            if curr != prev:
                vertical[column] += 1
                curr = prev

    # Make features list and return
    for row in range(0, height):
        for column in range(0, width):
            features.append([diagonals1[row-column+offset],diagonals2[row+column-offset], horizontal[row], vertical[column], V[row + int(m * math.floor((column)/q))]])

    return features

'''
Extract the final features that you would like to use
'''
def extract_final_features(digit_data, width, height):
    features=[]

    # Feature 1: Number of Non-Zero Pixels in Diagonals
    diagonals1 = [0 for i in range(0, width + height - 1)]
    diagonals2 = [0 for i in range(0, width + height - 1)]
    offset = width - 1
    for row in range(0, height):
        for column in range(0, width):
            if digit_data[row][column] != 0:
                diagonals1[row - column + offset] += 1
                diagonals2[row + column - offset] += 1

    # Feature 2: 4 Horizontal-Celled Projection
    n = width
    m = height
    kh = 2
    q = n / kh
    V = [0 for ii in range(0, m * kh)]
    for i in range(0, m):
        for j in range(0, n):
            if digit_data[i][j] != 0:
                V[i + int(m * math.floor((j) / q))] = 1


    # Feature 3: Vertical & Horizontal Crossings
    horizontal = [0 for ii in range(0, m)]
    vertical = [0 for ii in range(0, n)]

    for row in range(0, height):
        if digit_data[row][0] == 0:
            prev = 0
        else:
            prev = 1
        for column in range(1, width):
            if digit_data[row][column] == 0:
                curr = 0
            else:
                curr = 1
            if curr != prev:
                horizontal[row] += 1
                curr = prev

    for column in range(1, width):
        if digit_data[0][column] == 0:
            prev = 0
        else:
            prev = 1
        for row in range(0, height):
            if digit_data[row][column] == 0:
                curr = 0
            else:
                curr = 1
            if curr != prev:
                vertical[column] += 1
                curr = prev

    f3 = []
    for row in range(0, height):
        for column in range(0, width):
            f3.append(digit_data[row][column])

    # Make features list and return
    for row in range(0, height):
        for column in range(0, width):
            features.append([diagonals1[row - column + offset], diagonals2[row + column - offset], horizontal[row],
                             vertical[column], V[row + int(m * math.floor((column) / q))], f3[row * m + column]])

    return features

'''
Compute the parameters including the prior and and all the P(x_i|y). Note
that the features to be used must be computed using the passed in method
feature_extractor, which takes in a single digit data along with the width
and height of the image. For example, the method extract_basic_features
defined above is a function than be passed in as a feature_extractor
implementation.

The percentage parameter controls what percentage of the example data
should be used for training. 
'''
def compute_statistics(data, label, width, height, feature_extractor, percentage=100.0):
    global P_xi_y
    global P_y
    global feature_list

    n = int(len(data)*percentage/100.0)
    m = width * height

    # Compute prior estimate list P_y based on count of each label
    P_y = [0 for i in range(0, num_labels)]
    for i in range(0, n):
        P_y[label[i]-1] += 1
    P_y[:] = [float(x) / float(n) for x in P_y]

    # Get features list using first data item
    features_0 = feature_extractor(data[0], width, height)
    if isinstance(features_0[0], list) == False:
        for i in range(0,m):
            if features_0[i] not in feature_list:
                feature_list.append(features_0[i])

        # Extract features
        P_xi_y = [[[0 for z in range(0, len(feature_list))] for j in range(0, m)] for i in range(0, num_labels)]
        # Compute conditional probabilities
        # Add counts of features
        for i in range(0, n):
            y_index = label[i]
            f = feature_extractor(data[i], width, height)
            for j in range(0, m):
                index = feature_list.index(f[j])
                P_xi_y[y_index][j][index] += 1

        # Divide by sum of counts per feature with Laplace smoothing
        for i in range(0, num_labels):
            y_sum = float(sum(P_xi_y[i][0])) + float(len(P_xi_y[i][0])*k)
            for j in range(0, m):
                P_xi_y[i][j][:] = [(float(x) + k) / y_sum for x in P_xi_y[i][j]]
    else:
        num_features = len(features_0[0])
        feature_list = [[] for ii in range(0, num_features)]
        feature_list[0] = [i for i in range(0, min(width,height))]
        feature_list[1] = [i for i in range(0, min(width,height))]
        feature_list[2] = [i for i in range(0, width)]
        feature_list[3] = [i for i in range(0, height)]

        for digit in range(0, n):
            features_0 = feature_extractor(data[digit], width, height)
            for i in range(4, m):
                for j in range(1, num_features):
                    if features_0[i][j] not in feature_list[j]:
                        feature_list[j].append(features_0[i][j])

        P_xi_y = [[[[0 for i4 in range(0,len(feature_list[i3]))] for i3 in range(0,len(feature_list))] for i2 in range(0, m)] for i1 in range(0, num_labels)]

        # Compute conditional probabilities
        # Add counts of features
        for i in range(0, n):
            y_index = label[i]
            f = feature_extractor(data[i], width, height)
            for pixel in range(0, m):
                for feat in range(0, num_features):
                    feat_index = feature_list[feat].index(f[pixel][feat])
                    P_xi_y[y_index][pixel][feat][feat_index] += 1

        # Divide by sum of counts per feature with Laplace smoothing
        for i in range(0, num_labels):
            for feat in range(0, num_features):
                y_sum = float(sum(P_xi_y[i][0][feat])) + float(len(P_xi_y[i][0][feat]) * k)
                for j in range(0, m):
                    P_xi_y[i][j][feat][:] = [(float(x) + k) / y_sum for x in P_xi_y[i][j][feat]]

'''
For the given features for a single digit image, compute the class 
'''
def compute_class(features):
    predicted = -1
    m = len(features)   # number of pixels

    if isinstance(features[0], list) == False:
        log_probabilities = [0 for i in range(0, num_labels)]
        for label in range(0, num_labels):
            log_probabilities[label] += math.log(P_y[label])
            for pixel in range(0,m):
                index = feature_list.index(features[pixel])
                log_probabilities[label] += math.log(P_xi_y[label][pixel][index])

    else:
        num_features = len(features[0])

        log_probabilities = [0 for i in range(0, num_labels)]
        for label in range(0, num_labels):
            log_probabilities[label] += math.log(P_y[label])
            for pixel in range(0,m):
                for feat in range(0, num_features):
                    index = feature_list[feat].index(features[pixel][feat])
                    log_probabilities[label] += math.log(P_xi_y[label][pixel][feat][index])

    predicted = max(xrange(len(log_probabilities)), key=log_probabilities.__getitem__)

    return predicted

'''
Compute joint probability for all the classes and make predictions for a list
of data
'''
def classify(data, width, height, feature_extractor):

    predicted=[]

    for i in range(0, len(data)):
        features = feature_extractor(data[i], width, height)
        predicted.append(compute_class(features))

    return predicted

