# train_gesture.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

df = pd.read_csv('gesture_data.csv', header=None)
X = df.iloc[:,1:].values.astype(np.float32)
le = LabelEncoder()
y = tf.keras.utils.to_categorical(le.fit_transform(df.iloc[:,0]))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(81,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=50, batch_size=32,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

model.save('gesture_model.keras')
np.save('label_classes.npy', le.classes_)
print("Classes:", le.classes_)
print(f"Val accuracy: {model.evaluate(X_val, y_val, verbose=0)[1]:.2%}")
