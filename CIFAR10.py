import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mlflow
from urllib.parse import urlparse



def inception_module3(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3_1 = Conv2D(f2_out, (1, 3), padding='same', activation='relu')(conv3)
    conv3_2 = Conv2D(f2_out, (3, 1), padding='same', activation='relu')(conv3_1)
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3_1_1 = Conv2D(f3_out, (1, 3), padding='same', activation='relu')(conv5)
    conv3_1_2 = Conv2D(f3_out, (3, 1), padding='same', activation='relu')(conv3_1_1)
    conv3_2_1 = Conv2D(f3_out, (1, 3), padding='same', activation='relu')(conv3_1_2)
    conv3_2_2 = Conv2D(f3_out, (3, 1), padding='same', activation='relu')(conv3_2_1)
    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3_2, conv3_2_2, pool], axis=-1)
    return layer_out  # function for creating a projected inception module


def constructModel():
    inputs = keras.Input(shape=x_train.shape[1:])
    conv1 = Conv2D(32, kernel_size=7, activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv1)
    b1 = BatchNormalization()(pool1)
    conv2 = Conv2D(64, kernel_size=3, activation='relu')(b1)
    d0 = Dropout(0.2)(conv2)
    b2 = BatchNormalization()(d0)

    layer = inception_module3(b2, 32, 48, 64, 8, 16, 16)
    globalPooling1 = GlobalAveragePooling2D()(layer)
    d1 = Dropout(0.4)(globalPooling1)
    hidden2 = Dense(32, activation='linear')(d1)
    outputs = Dense(10, activation='softmax')(hidden2)
    model1 = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn_model")
    return model1


def loadDataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2, shuffle=True, random_state=42)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
    #                                                   test_size=0.2, shuffle=True, random_state=42)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # y_val = to_categorical(y_val)
    x_train, x_val, x_test, y_train, y_val, y_test = loadDataset()
    print(y_train.shape)
    mlflow.set_experiment('cifar10')
    with mlflow.start_run(run_name='BaseModel'):
        model = constructModel()
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            # optimizer=keras.optimizers.SGD(learning_rate=0.1, nesterov=True),
            optimizer=keras.optimizers.Adam(),
            # optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
            # optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

        # fit the keras model on the dataset
        my_callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='auto',
                                              factor=0.75, patience=5, min_lr=1e-5)
        ]
        mlflow.tensorflow.autolog(every_n_iter=5, log_models=True, disable=False, exclusive=False,
                                  disable_for_unsupported_versions=False, silent=False)
        history = model.fit(x_train[:100], y_train[:100], epochs=10, callbacks=my_callbacks,
                            batch_size=64, validation_data=(x_val, y_val), shuffle=True)
        mlflow.log_param("epoch", 10)
        mlflow.log_param("batch_size", 64)
        for a in history.history['accuracy']:
            mlflow.log_metric("Accuracy", a)
        for ll in history.history['loss']:
            mlflow.log_metric("loss", ll)
        for va in history.history['val_accuracy']:
            mlflow.log_metric("ValidationAccuracy", va)
        for vl in history.history['val_loss']:
            mlflow.log_metric("ValidationLoss", vl)

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        #
        # # Model registry does not work with file store
        # if tracking_url_type_store != "file":
        #
        #     # Register the model
        #     mlflow.tensorflow.log_model(model, registered_model_name="Cifar10_Base_Model")
        # else:
        #     mlflow.tensorflow.log_model(model)
