from PIL import Image
import os
import math

input_folder = "image"
output_folder = "image_tiles"
tile_size = 512

os.makedirs(output_folder, exist_ok=True)

counter = 1

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        width, height = img.size

        cols = math.ceil(width / tile_size)
        rows = math.ceil(height / tile_size)

        for row in range(rows):
            for col in range(cols):
                left   = col * tile_size
                upper  = row * tile_size
                right  = min(left + tile_size, width)
                lower  = min(upper + tile_size, height)

                tile = img.crop((left, upper, right, lower))

                # 如果边缘不足512，用黑色填充补齐
                if tile.size != (tile_size, tile_size):
                    padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0))
                    tile = padded

                tile.save(os.path.join(output_folder, f"{counter}.jpg"), quality=95)
                counter += 1

print(f"完成！共生成 {counter - 1} 张切片，保存在 '{output_folder}' 文件夹中。")