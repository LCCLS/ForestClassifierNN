import pandas as pd
import numpy as np
import tensorflow.python.keras.losses

from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler

from plotting import plot_history, plot_heatmap
from utils import step_decay
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/cover_data.csv')

features = dataset.drop(['class'], axis=1)
labels = dataset['class']

quantitative_cols = ['Elevation', 'Aspect', 'Slope',
                     'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology',
                     'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                     'Horizontal_Distance_To_Fire_Points']

categorical_columns = features.loc[:, 'Wilderness_Area1':].columns.to_list()
cat_indices = [features.columns.get_loc(x) for x in categorical_columns]

# print(dataset.head(30).to_string())
# This shows that there is in fact quite an unbalance in the dataset so we should resample the data a bit
# print(labels.value_counts().to_frame().T)

# sm = ADASYN()
# features_resampled, labels_resampled = sm.fit_resample(features, labels)
# print(labels_resampled.value_counts().to_frame().T)

# Check which columns are quantitative
quant_columns = dataset.select_dtypes(include=['object']).columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=labels)

ct = ColumnTransformer([('standardized', StandardScaler(), ['Elevation', 'Aspect', 'Slope',
                                                            'Horizontal_Distance_To_Hydrology',
                                                            'Vertical_Distance_To_Hydrology',
                                                            'Horizontal_Distance_To_Roadways',
                                                            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                                                            'Horizontal_Distance_To_Fire_Points'])],
                       remainder='passthrough')

x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

model = Sequential(
    [
        layers.InputLayer(input_shape=(features.shape[1],)),
        layers.Dense(64, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=5,
    restore_best_weights=True,
)

# lrate = LearningRateScheduler(step_decay)

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=1000,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    callbacks=[early_stopping])

# Plotting the performance

# plot_history(history, 'acc')
# plot_history(history, 'loss')

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
class_names = ['Spruce/Fir', 'Lodgepole Pine',
               'Ponderosa Pine', 'Cottonwood/Willow',
               'Aspen', 'Douglas-fir', 'Krummholz']

print(classification_report(y_test, y_pred, target_names=class_names))
plot_heatmap(class_names, y_pred, y_test)

