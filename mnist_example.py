import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense,  MaxPool2D, BatchNormalization, GlobalAvgPool2D
#Tensorflow.Keras.Model
seq_model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32,(3,3), activation='relu'),
        Conv2D(64,(3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128,(3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10,activation='softmax')
    ]
)   

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

def display_some_example(examples, labels):

    plt.figure(figsize=(10,10))
    for i in range(25):
        idx = np.random.randint(0 , examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]
        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    print("x_train_shape = ",x_train.shape)
    print("y_train_shape = ",y_train.shape)
    print("x_test_shape = ",x_test.shape)    
    print("x_test_shape = ",y_test.shape)
    if False:
        display_some_example(x_train,y_train)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)

    y_train = tensorflow.keras.utils.to_categorical(y_train,10)
    y_test = tensorflow.keras.utils.to_categorical(y_test,10)

    # model = functional_model()
    model = MyCustomeModel()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

    #Model Training
    model.fit(x_train,y_train,batch_size=64,epochs=3,validation_split=0.2)

    #Model Evaluation on Test Set
    model.evaluate(x_test,y_test,batch_size=64)

