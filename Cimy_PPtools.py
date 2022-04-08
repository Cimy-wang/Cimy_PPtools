# Cimy's Python Pytorch Toolbox
"""
    In this tools, the image show function show the multimodal fusion result.
    If you want to show the single modal classification result need to modify it!

"""
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import random
import copy
import datetime
from prettytable import PrettyTable
from tqdm import tqdm
import time


def createPatches(X, y, windowSize, removeZeroLabels=False):
    """
        Create the image patches
        Arguments:
             X:                The original input data
             y:                The corresponding label
             windowSize:       Patch window size
             removeZeroLabels: Whether to return the patch results of entire image, default=False.
        Return:
             patchesData:      Patch data
             patchesLabels:    Patch data with corresponding label
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric')
    zeroPaddedX = zeroPaddedX.reshape(zeroPaddedX.shape[2], zeroPaddedX.shape[0], zeroPaddedX.shape[1])
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], X.shape[2], windowSize, windowSize), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float16')
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[2] - margin):
        for r in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[:, r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def random_sample(train_sample, validate_sample, Labels):
    """
        Randomly generate training, validate, and test sets
        Arguments:
             train_sample:                         The vector contains the number of training samples per class
             validate_sample:                      The vector contains the number of validate samples per class
             Labels:                               The ground truth data
        Return:
             TrainIndex, ValidateIndex, TestIndex: The vectorized coordinate values
    """
    num_classes = int(np.max(Labels))
    TrainIndex = []
    TestIndex = []
    ValidateIndex = []
    for i in range(num_classes):
        train_sample_temp = train_sample[i]
        validate_sample_temp = validate_sample[i]
        index = np.where(Labels == (i + 1))[0]
        Train_Validate_Index = random.sample(range(0, int(index.size)), train_sample_temp + validate_sample_temp)
        TrainIndex = np.hstack((TrainIndex, index[Train_Validate_Index[0:train_sample_temp]])).astype(np.int32)
        ValidateIndex = np.hstack((ValidateIndex, index[Train_Validate_Index[train_sample_temp:100000]])).astype(np.int32)
        Test_Index = [index[i] for i in range(0, len(index), 1) if i not in Train_Validate_Index]
        TestIndex = np.hstack((TestIndex, Test_Index)).astype(np.int32)

    return TrainIndex, ValidateIndex, TestIndex


def applyPCA(X, numComponents=75):
    """
        Apply PCA preprocessing for original data
        Arguments:
             X:             The original input data
             numComponents: the hyperparameter of reduced dimension
        Return:
             newX:          Dimensionality reduced data
             pca:           The calculated parameter of pca
    """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def normalization(data):
    """
        Normalizate the original data
        Arguments:
             data: The original input data
        Return:    Normalizated data
    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def reports(y_pred, Labels):
    """
        Obtain the final classification accuracy
        Arguments:
             y_pred:           The predict result of models
             Labels:           The ground truth data
        Return:
             Confusion Matrix: Tensor: class x class
             Accuracy matrix:  Vector: (class + 3) x 1
    """
    classification = classification_report(Labels, y_pred)
    confusion = confusion_matrix(Labels, y_pred)
    oa = np.trace(confusion) / sum(sum(confusion))
    ca = np.diag(confusion) / confusion.sum(axis=1)
    Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
    K = (oa - Pe) / (1 - Pe)
    aa = sum(ca) / len(ca)
    List = []
    List.append(np.array(oa)), List.append(np.array(K)), List.append(np.array(aa))
    List = np.array(List)
    accuracy_matrix = np.concatenate((ca, List), axis=0)
    # ==== Print table accuracy use PrettyTable====
    x = PrettyTable()
    x.add_column('index', [list(range(1, len(ca) + 1, 1)) + ['OA', 'AA', 'KA']][0])
    x.add_column('Accuracy', accuracy_matrix)
    print(x)
    # ==== Print table accuracy use format====
    # ind = [list(range(1, len(ca) + 1, 1)) + ['OA', 'AA', 'KA']][0]
    # target_names = [u'%s' % l for l in ind]
    # last_line_heading = 'avg / total'
    # name_width = max(len(cn) for cn in target_names)
    # width = max(name_width, len(last_line_heading), 2)
    # headers = ["     accuracy"]
    # head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    # report = head_fmt.format(u'', *headers, width=width)
    # report += u'\n\n'
    # rows = zip(target_names, accuracy_matrix)
    # row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}4f}' u'\n'
    # for row1 in rows:
    #     report += row_fmt.format(*row1, width=width, digits=2)
    # report += u'\n'
    # print(report)
    return classification, confusion, accuracy_matrix


def val(model, val_loader, criterion):
    """
        The validate function
        Arguments:
             model:      The trained models
             val_loader: The validate data set
        Return:
             acc:        The accuracy on the validate set
             avg_loss:   The accuracy on the validate set
    """
    global acc, acc_best
    model.eval()
    total_correct = 0
    eye = torch.eye(int(max(val_loader.dataset.tensors[2]) + 1)).cuda()
    avg_loss = 0.0

    start_time = datetime.datetime.now()
    with tqdm(
            iterable=val_loader,
    ) as t:
        with torch.no_grad():
            for i, (data_hsi, data_lidar, labels) in enumerate(val_loader):
                data_hsi, data_lidar, labels = Variable(data_hsi).cuda(), Variable(data_lidar).cuda(), Variable(labels).cuda()
                output = model(data_hsi, data_lidar)
                labels = labels.to(torch.int64)
                target_hot = eye[labels]
                avg_loss = criterion(output, target_hot)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
                acc = float(total_correct) / len(val_loader.dataset.tensors[0])
                cur_time = datetime.datetime.now()
                t.set_description_str(f"\33[39m[  Validation Set  ]")
                t.set_postfix_str(f"Val_Loss = {avg_loss:.6f}, Val_Accuracy = {acc:.6f}, Time: {cur_time - start_time}\33[0m")
                t.update()

        avg_loss /= len(val_loader.dataset.tensors[0])
        acc = float(total_correct) / len(val_loader.dataset.tensors[0])

    return acc, avg_loss


def train(model, criterion, device, train_loader, optimizer, EPOCHS, vis, val_loader, itera=1):
    """
        The train function
        Arguments:
             model:                                               The constructed models
             criterion:                                           The loss function
             device:                                              Use GPU or CPU for training
             train_loader:                                        The training data set
             optimizer:                                           The optimization function
             EPOCHS:                                              The total training epoch
             vis:                                                 whether to visual the training precessing
             val_loader:                                          The validate data set
             itera:                                               The repeated times of experiments
        Return:
             model:                                               The trained models
             (end_time_train - start_time_train).total_seconds(): The training time
    """
    global best_model
    acc_temp = 0
    epoch_temp = 1
    eye = torch.eye(int(max(train_loader.dataset.tensors[2]) + 1)).cuda()
    start_time_train = datetime.datetime.now()
    for epoch in range(1, EPOCHS + 1):
        start_time = datetime.datetime.now()
        model.train()
        number = 0
        with tqdm(
                iterable=train_loader,
        ) as t:
            for batch_idx, (data_hsi, data_lidar, target) in enumerate(train_loader):
                t.set_description_str(f"\33[34m[Epoch {epoch:03d}/ {(EPOCHS):03d}/ {(itera):02d}]")
                data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data_hsi, data_lidar)
                target = target.to(torch.int64)
                target_hot = eye[target]
                loss = criterion(output, target_hot)
                loss.backward()
                optimizer.step()
                output = output.argmax(dim=1)
                number += output.eq(target).float().sum().item()
                cur_time = datetime.datetime.now()
                t.set_postfix_str(
                    f"Tra_Loss = {loss.item():.6f}, Tra_Accuracy = {number / len(train_loader.dataset):.6f}, Time: {cur_time - start_time}\33[0m")
                t.update()
        val_acc, avg_loss = val(model, val_loader, criterion)
        if acc_temp <= val_acc:
            print('Best_Val_Value changed: from %f to %f;' % (acc_temp, val_acc), end="\t")
            epoch_temp = epoch
            acc_temp = val_acc
            best_model = copy.deepcopy(model)
            print('Best Classification Accuracy %f， Best Classification loss %f； Best Epoch： %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")
        else:
            print('Best Classification Accuracy %f， Best Classification loss %f； Best Epoch： %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")
        vis.line(Y=[[number / len(train_loader.dataset), val_acc]],
                 X=[epoch],
                 win='acc {}'.format(itera),
                 opts=dict(title='acc', legend=['acc', 'val_acc']),
                 update='append')
        time.sleep(0.1)
    model = best_model
    end_time_train = datetime.datetime.now()
    print('||======= Train Time for % s' % (end_time_train - start_time_train), '======||')
    return model, (end_time_train - start_time_train).total_seconds()


def test(model, device, test_loader):
    """
        The test function
        Arguments:
             model:                                             The constructed models
             device:                                            Use GPU or CPU for training
             test_loader:                                       The test data set
        Return:
             test_acc_temp:                                     The test accuracy
             test_loss_temp:                                    The test loss
             y_pred:                                            The predicted results
             target_1:                                          The ground truth
             (end_time_test - start_time_test).total_seconds(): The test time
    """
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    target_1 = []
    start_time_test = datetime.datetime.now()
    with torch.no_grad():
        for (data_hsi, data_lidar, target) in test_loader:
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
            target = target.to(torch.int64)
            output = model(data_hsi, data_lidar)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            y_pred_temp = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += y_pred_temp.eq(target.view_as(y_pred_temp)).sum().item()
            y_pred_temp_1 = y_pred_temp.data.cpu().numpy()
            target_temp_1 = target.data.cpu().numpy()
            y_pred.extend(y_pred_temp_1)
            target_1.extend(target_temp_1)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(1, y_pred.size)
        y_pred = np.array(y_pred).astype(np.float32)
        y_pred = y_pred[0]

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc_temp = format(100. * correct / len(test_loader.dataset))
    test_loss_temp = format(test_loss)
    end_time_test = datetime.datetime.now()
    print('||======= Test Time for % s' % (end_time_test - start_time_test), '======||')
    return test_acc_temp, test_loss_temp, y_pred, target_1, (end_time_test - start_time_test).total_seconds()


def label2color(imageID):
    """
        The label2color function:
        Arguments:
             imageID: The data set need to show
        Return:
             row:     The row of result
             col:     The col of result
             palette: The RGB image: Tensor: H x W x 3
    """

    global palette
    global row
    global col

    if imageID == 'PU':  # PaviaU
        row = 610
        col = 340
        palette = np.array([[192, 192, 192], [  0, 255,   0], [  0, 255, 255], [  0, 128,   0], [255,   0, 255],
                            [165,  82,  41], [128,   0, 128], [255,   0,   0], [255, 255,   0]])
    elif imageID == 'PC':  # PaviaC
        row = 1096
        col = 492
        palette = np.array([[  0,   0, 255], [  0, 128,   0], [  0, 255,   0], [255,   0,   0], [142,  71,   2],
                            [192, 192, 192], [  0, 255, 255], [246, 110,   0], [255, 255,   0]])
    elif imageID == 'IN':  # Indian
        row = 145
        col = 145
        palette = np.array([[140,  67,  46], [  0,   0, 255], [255, 100,   0], [  0, 255, 123], [164,  75, 155],
                            [101, 174, 255], [118, 254, 172], [ 60,  91, 112], [255, 255,   0], [255, 255, 125],
                            [255,   0, 255], [100,   0, 255], [  0, 172, 254], [  0, 255,   0], [171, 175,  80],
                            [101, 193, 60]])
    elif imageID == 'SA':  # Salinas
        row = 512
        col = 217
        palette = np.array([[140,  67,  46], [  0,   0, 255], [255, 100,   0], [  0, 255, 123], [164,  75, 155],
                            [101, 174, 255], [118, 254, 172], [ 60,  91, 112], [255, 255,   0], [255, 255, 125],
                            [255,   0, 255], [100,   0, 255], [  0, 172, 254], [  0, 255,   0], [171, 175,  80],
                            [101, 193, 60]])
    elif imageID == 'DC_S':  # Washington_DC_small_map
        row = 280
        col = 307
        palette = np.array([[204, 102, 102], [153,  51,   0], [204, 153,   0], [  0, 255,   0], [  0, 102,   0],
                            [  0,  51, 255], [153, 153, 153]])
    elif imageID == 'DC_B':  # Washington_DC_big_map
        row = 1280
        col = 307
        palette = np.array([[203,  26,   0], [ 64,  64,  64], [251, 118,  19],
                            [102, 254,  77], [ 51, 152,  26], [  0,   0, 254], [254, 254, 254]])
    elif imageID == 'KSC':  # KSC
        row = 512
        col = 614
        palette = np.array([[140,  67,  46], [  0,   0, 255], [255, 100,   0], [  0, 255, 123],  [164, 75, 155],
                            [101, 174, 255], [118, 254, 172], [ 60,  91, 112], [255, 255,   0], [255, 255, 125],
                            [255,   0, 255], [100,   0, 255], [  0, 172, 254]])
    elif imageID == 'HU':  # Huston
        row = 349
        col = 1905
        palette = np.array([[  0, 205,   0], [127, 255,   0], [ 46, 139,  87], [  0, 139,   0], [160,  82,  45],
                            [  0, 255, 255], [255, 255, 255], [216, 191, 216], [255,   0,   0], [139,   0,   0],
                            [  0,   0,   0], [255, 255,   0], [238, 154,   0], [ 85,  26, 139], [255, 127, 80]])
    elif imageID == 'Trento':  # Trento
        row = 166
        col = 600
        palette = np.array([[  0, 217,  89], [203,  26,   0], [251, 118,  19], [ 51, 254,  26], [ 51, 152,  26],
                            [  0,   0, 251]])

    palette = palette * 1.0 / 255

    return row, col, palette


def imshow_multimodal(model, patchesData_1, patchesData_2, patchesLabels, y_pred, DEVICE, BATCH_SIZE, imageID,
                      testIndex, background, dpi):
    """
        The final classification maps imshow function:
        Arguments:
             model:             The trained models
             patchesData_1:     The patched modal 1 data
             patchesData_2:     The patched modal 2 data
             patchesLabels:     The patched data corresponding label
             y_pred:            The predicted results
             DEVICE:            Use GPU or CPU for training
             BATCH_SIZE:        Hyperparameter of network
             imageID:           The data set need to show
             testIndex:         The test set coordinates
             background:        Whether to show final maps with background
             dpi:               The figure dpi
        Return:
             X_result:          The final label2RGB result: Tensor: H x W x 3
    """

    labels = patchesLabels
    num_class = int(labels.max())

    row, col, palette = label2color(imageID)

    X_result = np.zeros((labels.shape[0], 3))
    if background == 1:
        y_pred = []
        patchesData_1_torch, patchesData_2_torch, patchesLabel_torch  = \
            torch.from_numpy(np.array(patchesData_1).astype(np.float32)), \
            torch.from_numpy(np.array(patchesData_2).astype(np.float32)), \
            torch.from_numpy(patchesLabels)

        patchesData_loader = DataLoader(dataset=TensorDataset(patchesData_1_torch,
                                                              patchesData_2_torch,
                                                              patchesLabel_torch),
                                        batch_size=BATCH_SIZE)

        for (patchesData_1, patchesData_2, target) in patchesData_loader:
            patchesData_1, patchesData_2, target = patchesData_1.to(DEVICE), patchesData_2.to(DEVICE), target.to(DEVICE)
            output = model(patchesData_1, patchesData_2)
            y_pred_temp = output.max(1, keepdim=True)[1]
            y_pred_temp_1 = y_pred_temp.data.cpu().numpy()
            y_pred.extend(y_pred_temp_1)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(1, y_pred.size)
        y_pred = np.array(y_pred).astype(np.float32)
        y_pred = y_pred[0]
        Result_all = y_pred+1
    else:
        Result_all = labels
        Result_all[testIndex] = np.array(y_pred).astype(np.float32)+1

    for i in range(1, num_class + 1):
        X_result[np.where(Result_all == i), 0] = palette[i - 1, 0]
        X_result[np.where(Result_all == i), 1] = palette[i - 1, 1]
        X_result[np.where(Result_all == i), 2] = palette[i - 1, 2]
    X_result = np.reshape(X_result, (col, row, 3))
    X_result_1 = X_result.swapaxes(0, 1)
    plt.figure(dpi=dpi)
    plt.axis("off")
    plt.imshow(X_result_1)
    plt.show()
    return X_result


def imshow_singlemodal(patchesLabels, Result_all, testIndex, y_pred, background=1, imageID='PU', dpi = 800):
    """
        The final classification maps imshow function with single modal:
        Arguments:
             Result_all:        The predicted result of all pixels
             patchesLabels:     The patched data corresponding label
             y_pred:            The predicted results
             imageID:           The data set need to show
             testIndex:         The test set coordinates
             background:        Whether to show final maps with background
             dpi:               The figure dpi
        Return:
             X_result:          The final label2RGB result: Tensor: H x W x 3
    """

    labels = patchesLabels
    num_class = int(labels.max())

    row, col, palette = label2color(imageID)

    X_result = np.zeros((labels.shape[0], 3))

    if background == 1:
        Result_all = Result_all
    else:
        Result_all = labels
        Result_all[testIndex] = y_pred

    for i in range(1, num_class + 1):
        X_result[np.where(Result_all == i), 0] = palette[i - 1, 0]
        X_result[np.where(Result_all == i), 1] = palette[i - 1, 1]
        X_result[np.where(Result_all == i), 2] = palette[i - 1, 2]
    X_result = np.reshape(X_result, (col, row, 3))
    X_result_1 = X_result.swapaxes(0, 1)
    plt.figure(dpi=dpi)
    plt.axis("off")
    plt.imshow(X_result_1)
    plt.show()
    return X_result
