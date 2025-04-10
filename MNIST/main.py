import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import pygame

from image import AIImage, np


def main() -> None:
    images_path = "F:\\project\\dataset\\MNIST\\test\\"
    names = AIImage.search(images_path)

    model_path = "F:\\project\\Python\\FirstAI"
    model: keras.Sequential = keras.models.load_model(f'{model_path}\\MNIST_model_1.h5')

    pygame.init()

    win = pygame.display.set_mode((520, 250))
    clock = pygame.time.Clock()

    index: str = ''
    old_index: str = ''
    name: str = None
    local = False

    def get_from_aiimg(_img: AIImage) -> pygame.Surface:
        surface = pygame.Surface((_img.width, _img.height), depth=32)

        for x in range(0, _img.width):
            for y in range(0, _img.height):
                r, g, b = [_img.pixel_data[_img.get_pix_index(x, y)]] * 3
                surface.set_at((y, x), (r, g, b))

        return pygame.transform.scale_by(surface, 5)

    def sum_index(s, i) -> str:
        try:
            return str(int(s) + i)
        except ValueError:
            return 0

    index_font = pygame.font.SysFont('Arial', 50, bold=True)
    img_surf = pygame.Surface((28 * 5, 28 * 5))
    img_surf.fill('yellow')
    img_category_font = pygame.font.SysFont('Arial', 100, bold=True)
    fin_img_category = img_category_font.render('None', True, 'yellow')
    category_font = pygame.font.SysFont('Arial', 20, bold=True)
    fin_category1 = category_font.render('None', True, 'yellow')
    fin_category2 = category_font.render('', True, 'white')

    running = True
    while running:
        """ events """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    index = index[:-1]
                elif event.key == pygame.K_DELETE:
                    index = ''
                elif event.key == 61:
                    index = sum_index(index, 1)
                elif event.key == 45:
                    index = sum_index(index, -1)
                elif event.unicode in '0123456789':
                    index += event.unicode
                elif event.key == pygame.K_RETURN:
                    if name is not None:
                        img = AIImage(
                            images_path + name if not local else rf'F:\\project\\Python\\FirstAI\\MNIST\\prompt_image.png'
                        )
                        img_surf = get_from_aiimg(img)
                        ret = np.array([round(i, 4) for i in model.predict(np.array([img.pixels]), verbose=0)[0]])
                        fin_category1 = category_font.render(str(ret[:5]), True, 'white')
                        fin_category2 = category_font.render(str(ret[5:]), True, 'white')
                        ai_ret = ret.argmax()
                        fin_img_category = img_category_font.render(
                            str(ai_ret),
                            True,
                            'green' if ai_ret == img.category else 'red'
                        )
                        name = None
                        old_index = index
                elif event.key == pygame.K_SPACE:
                    local = not local
        try:
            name = names[int(index)]
            fin_index_font = index_font.render(
                f'image with index: {index}',
                True,
                'cyan' if old_index != index else 'green'
            )
        except IndexError:
            fin_index_font = index_font.render(f'image: "{index}" not in test dir', True, 'red')
        except ValueError:
            fin_index_font = index_font.render(f'index: "{index}" is not Integer', True, 'red')
        """ render """
        win.fill('black')

        win.blit(fin_index_font, (5, 5))
        win.blit(img_surf, (5, fin_index_font.get_height() + 5))
        win.blit(fin_img_category, (28 * 5 + 10, fin_index_font.get_height() + 5))
        win.blit(fin_category1, (10, img_surf.get_height() + fin_index_font.get_height() + 10))
        win.blit(
            fin_category2,
            (10, img_surf.get_height() + fin_category1.get_height() + fin_index_font.get_height() + 10)
        )

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__': main()
