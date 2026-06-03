
import numpy as np 

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split



def load_mnist():

    print("MNIST er dataset load hocche.")



    mnist=fetch_openml("mnist_784", version=1, as_frame=False)



    x=mnist.data

    y=mnist.target.astype(int)



    x=x/255.0



    x_train, x_test, y_train, y_test=train_test_split(

        x,

        y,

        test_size=10000,

        random_state=42,

        stratify=y

    )

    return x_train, x_test, y_train, y_test



def one_hot_encode(y, num_classes=10):

    one_hot=np.zeros((y.shape[0], num_classes))

    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot