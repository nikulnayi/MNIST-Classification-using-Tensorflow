import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense,  MaxPool2D, BatchNormalization, GlobalAvgPool2D


#Functinal Approch 
def functional_model():
    myInput = Input(shape=(28, 28, 1))

    x = Conv2D(32,(3,3), activation='relu')(myInput)
    x = Conv2D(64,(3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10,activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=myInput,outputs = x)
    return model

#Tensorflow.Keras.Model : Inherit from the class
class MyCustomeModel(tensorflow.keras.Model):
    def __init__(self):
        super().__init__()    
        self.conv1 = Conv2D(32,(3,3), activation='relu')
        self.conv2 = Conv2D(64,(3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()
        self.conv3 = Conv2D(128,(3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()
        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10,activation='softmax')

    def call(self, MyInput):

        x = self.conv1(MyInput)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
