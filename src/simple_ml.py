import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 读取图像文件
    with gzip.open(image_filename, 'rb') as f:
        # 跳过前16个字节（魔数、数量、行数、列数）
        _ = np.frombuffer(f.read(16), dtype=np.uint8)
        # 读取图像数据
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        # 将数据转换为二维数组
        X = data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    with gzip.open(label_filename, 'rb') as f:
        # 跳过前8个字节（魔数、数量）
        _ = np.frombuffer(f.read(8), dtype=np.uint8)
        # 读取标签数据
        buf = f.read()
        y = np.frombuffer(buf, dtype=np.uint8)
    
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # 计算指数化的logits
    exp_Z = np.exp(Z)
    # 沿类别轴计算指数化logits的总和
    sum_exp_Z = np.sum(exp_Z, axis=1)
    # 计算指数化logits总和的对数
    log_sum_exp_Z = np.log(sum_exp_Z)
    # 获取真实标签对应的logit值
    correct_logit = Z[np.arange(Z.shape[0]), y]
    # 计算每个样本的Softmax损失
    loss = log_sum_exp_Z - correct_logit
    # 计算批次上的平均损失
    avg_loss = np.mean(loss)

    return avg_loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 获取样本数量和特征维度
    num_examples, input_dim = X.shape
    num_classes = theta.shape[1]

    # 遍历所有批次
    for i in range(0, num_examples, batch):
        # 提取当前批次的数据
        X_batch = X[i:i + batch]
        y_batch = y[i:i + batch]

        # 计算当前批次的预测值 Z (logits)
        Z = X_batch @ theta  # 形状为 (batch_size, num_classes)

        # 计算 Softmax 概率
        exp_Z = np.exp(Z)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)  # 形状为 (batch_size, 1)
        probs = exp_Z / sum_exp_Z  # Softmax 归一化后的概率，形状为 (batch_size, num_classes)

        # 构造 one-hot 编码的真实标签
        batch_size = X_batch.shape[0]
        I_y = np.zeros_like(probs)  # 初始化为零矩阵
        I_y[np.arange(batch_size), y_batch] = 1  # 将真实标签位置设为 1

        # 计算梯度
        grad = X_batch.T @ (probs - I_y) / batch  # 形状为 (input_dim, num_classes)

        # 更新参数 theta
        theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 获取数据的基本信息
    num_examples, input_dim = X.shape
    hidden_dim, num_classes = W2.shape

    # 遍历所有批次
    for i in range(0, num_examples, batch):
        #提取当前批次的数据
        X_batch = X[i:i + batch]
        y_batch = y[i:i + batch]
        batch_size = X_batch.shape[0]  # 当前批次的实际大小

        # Step 1: 前向传播
        Z1 = X_batch @ W1 # 第一层线性变换，形状为 (batch_size, hidden_dim)
        A1 = np.maximum(Z1, 0)  # ReLU 激活函数，形状为 (batch_size, hidden_dim)
        Z2 = A1 @ W2

        # Step 2: Softmax 和 one-hot 编码
        exp_Z2 = np.exp(Z2)
        sum_exp_Z2 = np.sum(exp_Z2, axis=1, keepdims=True)  # 形状为 (batch_size, 1)
        probs = exp_Z2 / sum_exp_Z2  # Softmax 概率，形状为 (batch_size, num_classes)

        # 构造 one-hot 编码的真实标签
        I_y = np.zeros_like(probs)  # 初始化为零矩阵
        I_y[np.arange(batch_size), y_batch] = 1  # 将真实标签位置设为 1

        # Step 3: 计算梯度
        G2 = probs - I_y  # 第二层的梯度，形状为
        G1 = (Z1 > 0) * (G2 @ W2.T)  # 第一层的梯度，形状为 (batch_size, hidden_dim)

        # 更新权重
        W2 -= lr * (A1.T @ G2) / batch_size  # 更新第二层权重
        W1 -= lr * (X_batch.T @ G1) / batch_size  # 更新第一层权重
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=True):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
