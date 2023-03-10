    get_ipython().system('pip install tensorflow')

    import tensorflow as tf
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import cv2
    from PIL import Image
    import numpy as np

    img = cv2.imread('train.jp2')
    mask = Image.open('image.png')

    mask = np.array(mask)

    np.save('train_images.npy', img)
    np.save('train_masks.npy', mask)



    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=0
    )



    input_shape = (256, 256, 3)
    num_classes = 2
    learning_rate = 1e-4

    inputs = Input(shape=input_shape)
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([drop3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)

    model = Model(inputs=inputs, outputs=outputs)



    train_images = np.load('train_images.npy')
    train_masks = np.load('train_masks.npy')


    train_images, test_images, train_masks, test_masks = train_test_split(train_images, train_masks, test_size=0.2)

    
    train_images = np.expand_dims(train_images, axis=0)
    train_masks = np.expand_dims(train_masks, axis=0)
    test_images = np.expand_dims(test_images, axis=0)
    test_masks = np.expand_dims(test_masks, axis=0)

    
    train_data_gen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
    test_data_gen = ImageDataGenerator()


   
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data_gen.flow(train_images, train_masks, batch_size=16), epochs=50, validation_data=(test_images, test_masks))


    


    score = model.evaluate(test_images, test_masks, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    




