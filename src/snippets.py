import cv2

img = cv2.imread("img/map.pgm", cv2.IMREAD_GRAYSCALE)

# 0 is obstacle, 205 is unknown, 254 is free


def fill_mask(img, mask, color):
    img[(mask!=0)] = color # != 0 to binarise the mask
    return img

def get_map_values(img):
    """Get unique pixel values in map."""
    out = set()
    for row in img:
        for px in row:
            out.add(px)
    return out

def get_tri_range(img):
    """Given three values, generate ranges using the values' midpoints."""
    values = list(get_map_values(img))
    values.sort()

    dividers = [int(values[i] + (values[i+1] - values[i]) / 2)
                for i in range(len(values) - 1)
                if i < len(values)]

    range_points = [values[0]]
    range_points.extend(dividers)
    range_points.append(values[-1])

    return list(zip(range_points, range_points[1:]))

def show_img(img):
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ranges = get_tri_range(img)

for range_ in ranges:
    mask = cv2.inRange(img, *[int(i) for i in range_])
    show_img(mask)

def img_masked_overlay(img, mask, back_img):
    """
    Apply image onto another image via mask.

    Note: Only works on images of the same size!
    """
    fg = cv2.bitwise_or(img, img, mask=mask)

    mask_inv = cv2.bitwise_not(mask)
    fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)

    return cv2.bitwise_or(fg, fg_back_inv)

show_img(img)

img_copy = img.copy()
img_copy = fill_mask(img.copy(), (mask!=0), 0)
show_img(mask)
show_img(img_copy)
