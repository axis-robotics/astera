from base import read_cam, detect_flowers, transformation_matrix

img = read_cam()
flowers = [i['center'] for i in detect_flowers(img)]
flowers_transformed = list(map(transformation_matrix, flowers))

print(flowers)
print(flowers_transformed)
