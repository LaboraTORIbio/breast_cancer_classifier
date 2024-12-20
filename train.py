import argparse
import numpy as np
import splitfolders
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


def split_and_load_dataset(input_dataset):
    dir_splitted = './data/splitted'
    dir_train = './data/splitted/train'
    dir_val = './data/splitted/val'
    dir_test = './data/splitted/test'
    # Splitting datasets:
    splitfolders.ratio(input_dataset, output=dir_splitted, seed=42,
                       ratio=(.8, .1, .1), group_prefix=None, move=False) 
    # Loading datasets:
    print('Training:')
    train_ds = image_dataset_from_directory(dir_train, image_size=(224, 224))
    print('Validation:')
    val_ds = image_dataset_from_directory(dir_val, image_size=(224, 224))
    print('Test:')
    test_ds = image_dataset_from_directory(dir_test, image_size=(224, 224))
    return train_ds, val_ds, test_ds


def calculate_class_weights(train_ds):
    class_labels = np.concatenate([y for x, y in train_ds], axis=0)
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict


def one_hot_encode(x, y):
    y = tf.keras.utils.to_categorical(y, num_classes=3)
    return x, y


def process_dataset(train_ds, val_ds, test_ds):
    # One-hot encoding class labels:
    train_ds = train_ds.map(one_hot_encode)
    val_ds = val_ds.map(one_hot_encode)
    test_ds = test_ds.map(one_hot_encode)
    # Prefetching datasets:
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE).cache()
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE).cache()
    test_ds = test_ds.prefetch(tf_data.AUTOTUNE).cache()
    return train_ds, val_ds, test_ds


def fine_tune_model(train_ds, val_ds, class_weights_dict):
    checkpoint = ModelCheckpoint(
        'model_{epoch:02d}_{val_accuracy:.3f}.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    # Define the CNN base model:
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model:
    base_model.trainable = False
    
    # Adding layers on top of the base model:
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)  # base model is in inference mode for BatchNorm
    x = GlobalAveragePooling2D()(x)  # pooling to reduce dimensionality
    for _ in range(3):
        x = Dense(50, activation='relu')(x)  # dense layer with ReLU activation
    outputs = Dense(3, activation='softmax')(x)  # output layer for 3 classes
    
    # Define and compile the model:
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model:
    print('\nTraining the dense layers...')
    model.fit(train_ds, epochs=5, class_weight=class_weights_dict, validation_data=val_ds)

    # Freeze all layers except the last 10:
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Compile the model with a low learning rate for fine-tuning:
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fine-tune the model:
    print('\nFine-tuning the model...')
    history = model.fit(train_ds, epochs=10, class_weight=class_weights_dict, validation_data=val_ds, callbacks=[checkpoint])
    
    return model, history


def training_workflow(input_dataset, output_model):
    print('\n1 - Splitting and loading dataset...')
    train_ds, val_ds, test_ds = split_and_load_dataset(input_dataset)
    print('\n2 - Processing dataset...')
    class_weights_dict = calculate_class_weights(train_ds)
    train_ds, val_ds, test_ds = process_dataset(train_ds, val_ds, test_ds)
    print('\n3 - Training model...')
    model, history = fine_tune_model(train_ds, val_ds, class_weights_dict)
    
    # Model metrics:
    train_accuracy = history.history['accuracy'][-1] * 100
    val_accuracy = history.history['val_accuracy'][-1] * 100
    print('\nEvaluating final model...')
    test_loss, test_accuracy = model.evaluate(test_ds, batch_size=32)
    test_accuracy = test_accuracy * 100
    print(f"\nTrain accuracy: {train_accuracy:.1f}%")
    print(f"Validation accuracy: {val_accuracy:.1f}%")
    print(f"Test accuracy: {test_accuracy:.1f}%\n")

    # Saving the TFLite model:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_model, 'wb') as f_out:
        f_out.write(tflite_model)
    print(f"\nModel saved as {output_model}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''Pipeline to train the Breast Cancer Classifier''')
    
    parser.add_argument('-i', action='store', help='Path to the input dataset', dest='INPUT',
                        required=False, default='./data/us_only')
    parser.add_argument('-o', action='store', help='Path to output the model', dest='OUTPUT',
                        required=False, default='breast_cancer_classifier.tflite')
    args = parser.parse_args()
    input_dataset = args.INPUT
    output_model = args.OUTPUT

    training_workflow(input_dataset, output_model)


if __name__ == '__main__':  
    main()
