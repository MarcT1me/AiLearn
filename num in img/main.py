from inspect import stack
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import pygame
import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

from image import AIImage


def get_from_aiimg(_img: AIImage) -> pygame.Surface:
    surface = pygame.Surface((_img.width, _img.height), depth=32)
    
    for x in range(0, _img.width):
        for y in range(0, _img.height):
            r, g, b = [_img.pixels[_img.get_pix_index(x, y)]]*3
            surface.set_at((y, x), (r, g, b))
    
    return pygame.transform.scale_by(surface, 5)


def mouse_down_callback(_):
    global drag_active
    if dpg.get_mouse_pos()[1] <= 0:
        drag_active = not drag_active


def mouse_release_callback(_):
    global drag_active
    if drag_active:
        drag_active = not drag_active


def drag_handle(_, data):
    if drag_active:
        dpg.set_viewport_width(dpg.get_viewport_width())
        dpg.set_viewport_height(dpg.get_viewport_height())
        p = dpg.get_viewport_pos()
        dpg.set_viewport_pos((p[0] + data[1], p[1] + data[2]))


def exit_app():
    global running
    running = False


def generate(_):
    global generated_surface
    generated_surface.fill('black')
    # prompt
    prompt = int(dpg.get_value('prompt')%10)
    pr_array = [int(i == prompt) for i in range(10)]
    # predict
    ret_pixels = [int(255*i) for i in model.predict(np.array([pr_array]))[0]]
    # create image
    ai_img = AIImage(width=28, height=28, pix_data=ret_pixels)
    ai_img.in_console()
    generated_surface = pygame.transform.scale_by(get_from_aiimg(ai_img), 2)


def build() -> None:
    """ Create MAIN window """
    with dpg.window(label="Example Window", tag='main_win'):  # manu-bar
        with dpg.menu_bar(show=False, tag='main_bar'):
            # ico
            dpg.add_image('main_ico')
            # exit
            dpg.add_button(label='Exit', callback=exit_app, pos=(width - 60, 0), width=60)
            dpg.bind_item_theme(dpg.last_item(), 'exit_button_style')
        
        with dpg.group(horizontal=False, tag='main_group'):
            dpg.add_text("Generator int images 28x28 pixels")
            with dpg.group(horizontal=True, tag='input_group'):
                dpg.add_input_int(label='integer char - prompt', width=100, min_value=0, max_value=9, tag='prompt')
                dpg.add_button(label='generate', tag='generate', callback=generate)
            dpg.add_image(texture_tag='generated_img')


running = True
drag_active = False
w_size = width, height = 520, 250
model: keras.Sequential = keras.models.load_model(fr'F:\\project\\Python\\FirsTAI\\MNIST_generate_model_1.h5')
generated_surface = pygame.Surface((28*2, 28*8))
generated_surface.fill('white')
pygame.draw.circle(generated_surface, 'green', (28, 28), 10)
get_gen_surf = lambda: np.array(
    Image.frombytes('RGBA', generated_surface.get_size(), pygame.image.tostring(generated_surface, 'RGBA'))
)


def main() -> None:
    """ MAIN """
    app_path: str = os.path.dirname(
        os.path.abspath(stack()[0].filename)
    ).removesuffix('\\PyInstaller\\loader').removesuffix('\\_internal')
    print(app_path)
    
    """ Инициализация Dear PyGui """
    dpg.create_context()
    
    """ ОБРАБОТКА НАЖАТИЙ """
    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(callback=drag_handle)
        dpg.add_mouse_click_handler(callback=mouse_down_callback)
        dpg.add_mouse_release_handler(callback=mouse_release_callback)
    
    """ Загрузка SysFonts """
    with dpg.font_registry():
        dpg.add_font('C:\Windows\Fonts\ARIALN.TTF', 30, tag='arial_font_30')
    
    """ THEMES """
    with dpg.theme(tag='exit_button_style'):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (230, 75, 50), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (150, 150, 150), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 70, 70), category=dpg.mvThemeCat_Core)
    with dpg.theme(tag='comment_font_style'):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (120, 120, 120), category=dpg.mvThemeCat_Core)
    
    """ Загрузка TEXTURES """
    with dpg.texture_registry():
        tb_width, tb_height, _, tb_data = dpg.load_image(f"{app_path}/ico.png")
        dpg.add_static_texture(width=tb_width, height=tb_height, default_value=tb_data, tag="main_ico")
        
        dpg.add_dynamic_texture(width=28*2, height=28*2, default_value=get_gen_surf(), tag='generated_img')
    
    """ BUILD APP """
    build()
    
    dpg.bind_font('arial_font_30')
    """ START """
    dpg.create_viewport(
        title='Custom Title', width=width, height=height, decorated=False
    )
    dpg.set_primary_window('main_win', True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    """ MAIN-LOOP """
    while running:
        dpg.set_value('prompt', int(dpg.get_value('prompt')%10))
        dpg.set_value('generated_img', get_gen_surf())
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()


if __name__ == '__main__': main()
