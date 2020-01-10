import os
import gym_gvgai
from PIL import Image, ImageDraw, ImageFont

class LevelVisualizer:
    def __init__(self, env, tile_size=16):
        self.game = env
        self.version = 'v1'
        self.tile_size = tile_size
        self.dir = gym_gvgai.dir

        self.game_description = self.read_gamefile()
        self.sprite_paths = self.sprite_mapping()
        self.level_mapping = self.ascii_map()
        self.tiles = self.build_tiles()

    def read_gamefile(self):
        path = os.path.join(self.dir, 'envs','games', f'{self.game}_{self.version}', f'{self.game}.txt')
        with open(path, 'r') as game:
            gamefile = game.readlines()
        return gamefile

    def get_indent(self, string):
        string = string.replace('\t', '        ')
        return len(string) - len(string.lstrip())

    def remove_comment(self, string):
        if('#' in string):
            comment_idx = string.index('#')
            return string[:comment_idx]
        return string

    def remove_excess_objs(self, level_str):
        if('A' in level_str):
            avatar_idx = level_str.index('A')
            level_str = level_str.replace('A', '.')
            level_str = level_str[:avatar_idx] + 'A' + level_str[avatar_idx+1:]
        #Philip: Add method to read singelton after file read, then loop through those characters here
        if('+' in level_str):
            avatar_idx = level_str.index('+')
            level_str = level_str.replace('+', '.')
            level_str = level_str[:avatar_idx] + '+' + level_str[avatar_idx+1:]
        if('g' in level_str):
            avatar_idx = level_str.index('g')
            level_str = level_str.replace('g', '.')
            level_str = level_str[:avatar_idx] + 'g' + level_str[avatar_idx+1:]
        return level_str

    def sprite_mapping(self):
        sprite_set = False
        sprites = {}
        for l in self.game_description:
            line = l.split()
            if(len(line) == 0):
                pass
            elif(sprite_set):
                if(indent >= self.get_indent(l)):
                    sprite_set = False
                else:
                    img = [i for i in line if i.startswith('img=')]
                    if(len(img)==1):
                        key = line[0]
                        sprite = img[0][4:]
                        sprites[key] = sprite
            elif(line[0] == 'SpriteSet'):
                sprite_set = True
                indent = self.get_indent(l)
        return sprites

    def ascii_map(self):
        level_mapping = False
        mapping = {}
        for l in self.game_description:
            line = self.remove_comment(l).split()
            if(len(line) == 0):
                pass
            elif(level_mapping):
                if(indent >= self.get_indent(l)):
                    level_mapping = False
                elif(line[1] == '>'):
                    key = line[0]
                    tile = line[2:]
                    mapping[key] = tile
            elif(line[0] == 'LevelMapping'):
                level_mapping = True
                indent = self.get_indent(l)
        return mapping

    def get_sprite(self, name, alias=0):
        sprite = os.path.basename(self.sprite_paths[name])
        sprite_dir = os.path.dirname(self.sprite_paths[name])
        sprite_dir = os.path.join(self.dir,'envs','gvgai','sprites', sprite_dir)
        try:
            sprite_filename = min([i for i in os.listdir(sprite_dir) if i.startswith(sprite)])
            path = os.path.join(sprite_dir, sprite_filename)
            sprite = Image.open(path).convert('RGBA')
        except:
            sprite = Image.new('RGBA', (12,11), (0,0,0, 255)) #
            d = ImageDraw.Draw(sprite)
            d.text((3,0), alias)
        sprite = sprite.resize((self.tile_size, self.tile_size)) #, Image.ANTIALIAS) #Philip: remove antialias from library
        return sprite

    def _build_tile(self, name, sprite_list, background = -1):
        if(len(sprite_list) == 0):
            return background
        elif(background == -1):
            background = self.get_sprite(sprite_list[0], name)
            return self._build_tile(name, sprite_list[1:], background)
        else:
            foreground = self.get_sprite(sprite_list[0], name)
            background = Image.alpha_composite(background, foreground)
            return self._build_tile(name, sprite_list[1:], background)
    
    def build_tiles(self):
        lvl_tiles = {}
        for k in self.level_mapping:
            sprite_list = self.level_mapping[k]
            tile = self._build_tile(k, sprite_list)
            lvl_tiles[k] = tile
        return lvl_tiles

    def draw_level(self, level_str):
        level_str = self.remove_excess_objs(level_str)
        lvl_rows = level_str.split()
        w = len(lvl_rows[0])
        h = len(lvl_rows)
        ts = self.tile_size
        lvl_img = Image.new("RGB", (w*ts, h*ts))

        for y,r in enumerate(lvl_rows):
            for x,c in enumerate(r):
                img = self.tiles[c]
                lvl_img.paste(img, (x*ts, y*ts, (x+1)*ts, (y+1)*ts))
                
        return lvl_img
