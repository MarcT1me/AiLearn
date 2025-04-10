from pathlib import Path
import os
from PIL import Image as PILImage
import numpy as np
from typing import Self


class AIImage:
    cache = dict()
    
    def __init__(self, _path: str | Path):
        # путь и связанные с именем файла строки
        self.fullpath: Path = _path
        self.basename: str = os.path.basename(_path)
        self.name: str = os.path.splitext(self.basename)[0]
        self.extension = os.path.splitext(self.basename)[-1]
        
        try:
            self.category = int(self.name[-1])
        except ValueError:
            self.category = 0
        # открытие файла изображений
        with PILImage.open(_path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.width, self.height = image.size
            
            # преобразую в ЧБ и сохраняю простой uint8
            self.pixels: list[int] = [(R + G + B)/3 for (R, G, B) in image.getdata()]
            self.pixel_data = np.array(self.pixels, np.uint8)
    
    def get_pix_index(self, x, y) -> int:
        return x*self.width + y
    
    def in_console(self):
        for p in range(self.width*self.height):
            if p%self.width == 0:
                print()
            print(self.pixel_data[p], end='\t')
        print()
    
    @staticmethod
    def search(_path: str | Path) -> list[str]:
        for _, _, n in os.walk(_path):
            # имена из первой попавшейся директории
            return n
    
    @classmethod
    def iter_img(cls, _path: str | Path) -> Self:
        # нахожу все имена
        _names = cls.search(_path)
        for name in _names:
            # каждый раз создаю изображение по пути
            yield cls(os.path.join(_path, name))


def main() -> None:
    import time
    
    path = 'f:\\project\\dataset\\MNIST\\debug\\'
    names = AIImage.search(path)
    
    for name in names:
        p = path + name
        print(p)
        
        img = AIImage(p)
        
        img.in_console()
        print(img.width, img.height)
        print(img.category)
        time.sleep(0.5)


if __name__ == '__main__': main()
