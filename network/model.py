import numpy as np 



class NeuralNetwork:

    def __init__(self, input_size=784, hidden_size=64, output_size=10, seed=42):

        np.random.seed(seed)



        self.w1=np.random.randn(input_size, hidden_size)*np.sqrt(2/input_size)



        self.b1=np.zeros((1,hidden_size))



        self.w2=np.random.randn(hidden_size, output_size)*np.sqrt(2/hidden_size)



        self.b2=np.zeros((1,output_size))



    def sigmoid(self,z):

            return 1/(1+np.exp(-z))



    def sofmax(self, z):

        exp_z=np.exp(z-np.max(z, axis=1, keepdims=True))

        return exp_z/np.sum(exp_z, axis=1, keepdims=True)

    

    def forward(self, x):

        self.z1=np.dot(x, self.w1)+self.b1

        self.a1=self.sigmoid(self.z1)



        self.z2=np.dot(self.a1, self.w2)+self.b2

        self.a2=self.sofmax(self.z2)



        return self.a2 



    def compute_loss(self, y_true, y_pred):

        epsilon=1e-15

        y_pred=np.clip(y_pred, epsilon, 1-epsilon)

        loss=-np.mean(np.sum(y_true*np.log(y_pred), axis=1))

        return loss 



    def print_shapes(self):



        print("W1 shape:", self.w1.shape)

        print("b1 shape:", self.b1.shape)

        print("W2 shape:", self.w2.shape)

        print("b2 shape:", self.b2.shape)