from pathlib import Path
import os
from PIL import Image as PILImage
from typing import Self


class AIImage:
    cache = dict()
    
    def __init__(self, *,
                 path: str | Path = None,
                 category: int = 0,
                 width: int = None,
                 height: int = None,
                 pix_data: list[int] = None
                 ):
        # путь и связанные с именем файла строки
        self.fullpath: Path = path
        
        if path is not None:
            basename: str = os.path.basename(path)
            name: str = os.path.splitext(basename)[0]
            
            try:
                self.category = int(name[-1])
            except ValueError:
                self.category = category
            
            # открытие файла изображений
            with PILImage.open(path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                self.width, self.height = image.size
                
                # преобразую в ЧБ и сохраняю простой uint8
                self.pixels: list[int] = [(R + G + B)//3 for (R, G, B) in image.getdata()]
        else:
            self.category = category
            self.width = width
            self.height = height
            self.pixels = pix_data
    
    def get_pix_index(self, x, y) -> int:
        return x*self.width + y
    
    def in_console(self):
        for p in range(self.width*self.height):
            if p%self.width == 0:
                print()
            print(self.pixels[p], end='\t')
        print()
    
    @staticmethod
    def search(_path: str | Path) -> list[str]:
        for _, _, n in os.walk(_path):
            # имена из первой попавшейся директории
            return n
    
    @classmethod
    def iter_img(cls, path: str | Path) -> Self:
        # нахожу все имена
        _names = cls.search(path)
        for name in _names:
            # каждый раз создаю изображение по пути
            yield cls(path=os.path.join(path, name))


def main() -> None:
    import time
    
    path = 'f:\\project\\dataset\\MNIST\\debug\\'
    names = AIImage.search(path)
    
    for name in names:
        p = path + name
        print(p)
        
        img = AIImage(path=p)
        
        img.in_console()
        print(img.width, img.height)
        print(img.category)
        time.sleep(0.5)


if __name__ == '__main__': main()
