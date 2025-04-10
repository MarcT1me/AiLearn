import keras
from image import AIImage, Path
import numpy as np
from typing import Iterable


class AIModel:
    def __init__(self, size: int | Iterable = 28*28, *args, **kwargs):
        if isinstance(size, Iterable):
            size: int = sum(size)
        
        super().__init__(*args, **kwargs)
        self.model = keras.Sequential()
        print('construct model')
        
        # вход - ЧБ пиксели
        self.model.add(keras.layers.Dense(size, activation='relu'))  # 784
        
        # скрытые слои
        self.model.add(keras.layers.Dense(size//2, activation='relu'))  # 392
        self.model.add(keras.layers.Dense(size//4, activation='relu'))  # 196
        self.model.add(keras.layers.Dense(size//8, activation='relu'))  # 98
        self.model.add(keras.layers.Dense(size//14, activation='relu'))  # 56
        
        # выход - 10 категорий
        
        # выход классификации
        self.model.add(keras.layers.Dense(10, activation='softmax'))  # 10
        
        print('compile model')
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def learn(self, *, path: str | Path) -> list:
        dir_size = len(AIImage.search(path))
        
        # создание списков
        data: list[list[int]] = []
        labels: list[int] = []
        
        # заполнение
        i = 0
        for img in AIImage.iter_img(path):
            # определяем следующее изображение
            data.append(img.pixels)
            labels.append(img.category)
            del img
            i += 1
            print(f'\rloading images:  {round(i/dir_size*100, 2)}%', end='')
        print('\t - complete')
        
        # преобразование
        print('compile data')
        data_array = np.array(data)
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
        
        # обучение
        print('start learn')
        return self.model.fit(data_array, one_hot_labels, epochs=50, batch_size=60).history['loss']


def main() -> None:
    import matplotlib.pyplot as plt
    path = 'F:\\project\\dataset\\MNIST\\train\\'
    
    model = AIModel()
    loss = model.learn(path=path)
    
    plt.plot(loss)
    plt.grid(True)
    plt.show()
    
    name = 'MNIST_model_1.h5'
    print(f'save model with name: {name}')
    keras.saving.save_model(model.model, filepath=fr'F:\\project\\Python\\FirstAI\\{name}', overwrite=True)


if __name__ == '__main__': main()
