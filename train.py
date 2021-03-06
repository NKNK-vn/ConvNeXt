import os
from argparse import ArgumentParser

#import tensorflow
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from model import ConvNeXt
import matplotlib.pyplot as plt 
# from optimizer import CustomLearningRate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--model-folder", default='./model/', type=str)
    parser.add_argument("--train-folder", default='./data/train', type=str)
    parser.add_argument("--valid-folder", default='./data/validation', type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=2, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ConvNeXt-------------------')
    print('Github: https://github.com/NKNK-vn/ConvNeXt')
    print('Email: khoi.nkn12@gmail.com') #Update later
    print('---------------------------------------------------------------------')
    print('Training MobileNet model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    train_folder = args.train_folder
    valid_folder = args.valid_folder
    batch_size =  args.batch_size
    image_size = args.image_size
    num_classes = args.num_classes

    #Use ImageDataGenerator for augmentation
    train_datagen = ImageDataGenerator(rotation_range=15,
                                        rescale=1./255,
                                        shear_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    #Load train set
    train_ds = train_datagen.flow_from_directory(
        train_folder,
        target_size=(int(image_size), int(image_size)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
    )
    #Load test set
    val_ds = val_datagen.flow_from_directory(
        valid_folder,
        target_size=(int(image_size), int(image_size)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
    )
    print('Train label: {}'.format(train_ds.class_indices))
    print('Val label: {}'.format(val_ds.class_indices))

    # assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'
    assert image_size > 32, 'Unfortunately, model accepts jpg images size higher than 32'

    # Load model
    convnext = ConvNeXt(image_size)
    model = convnext.build_model()

    # # Create custom Optimizer
    # lrate = CustomLearningRate(512)
    #Callback
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    checkpoint = ModelCheckpoint(filepath=args.model_folder + 'model.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1)
    callbacks = [learning_rate_reduction, checkpoint]                    
    
    #Train model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, epochs = args.epochs, callbacks=callbacks, validation_data = val_ds)
    
    #Show Model Train Loss History
    plt.plot(history.history['loss'])
    plt.title('model training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.savefig("train_loss.jpg")
    plt.show()

    #Show Model Train Accuracy History
    plt.plot(history.history['accuracy'])
    plt.title('model training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy'], loc='upper left')
    plt.savefig("train_acc.jpg")
    plt.show()

    #Show Model Val Loss History
    plt.plot(history.history['val_loss'])
    plt.title('model validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['val_loss'], loc='upper left')
    plt.savefig("val_loss.jpg")
    plt.show()

    #Show Model Val Accuracy History
    plt.plot(history.history['val_accuracy'])
    plt.title('model validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['val_accuracy'], loc='upper left')
    plt.savefig("val_acc.jpg")
    plt.show()
    # Do Prediction