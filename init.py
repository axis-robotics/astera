from base import read_cam, detect_flowers, transformation_matrix, release_cam

img = read_cam()
if img:

    flowers = [i['center'] for i in detect_flowers(img)]
    flowers_transformed = list(map(transformation_matrix, flowers))

    print(flowers)
    print(flowers_transformed)

release_cam()