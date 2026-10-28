"""The 256 px page thumbnails the DINOv2 training vectors were computed from.

The same functions as metakat/page_type/engines/core/context/dino.py, which the deployed engine runs
(embeddings/image/dinov2/tests checks that they agree). Reconstruction of the original thumbnails was verified
pixel for pixel on 48 pages.
"""
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


THUMBNAIL_SIZES = (512, 256)
THUMBNAIL_JPEG_QUALITY = 90


def _shrink(image, max_size):
    height, width = image.shape[:2]
    longer_side = max(height, width)
    if longer_side <= max_size:
        return image
    scale = max_size / longer_side
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _jpeg_roundtrip(image):
    success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, THUMBNAIL_JPEG_QUALITY])
    if not success:
        raise ValueError('Cannot encode the thumbnail JPEG')
    return encoded


def original_thumbnail(rgb):
    """The 256 px thumbnail the training vectors were computed from.

    Full resolution -> 512 px longest edge -> JPEG Q90 -> 256 px longest edge -> JPEG Q90
    (INTER_AREA, Python round() for the shorter side). Images that already are at most 256 px
    are returned unchanged.
    """
    if max(rgb.shape[:2]) <= THUMBNAIL_SIZES[-1]:
        return rgb
    image = np.ascontiguousarray(rgb[..., ::-1])
    for size in THUMBNAIL_SIZES:
        encoded = _jpeg_roundtrip(_shrink(image, size))
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    with Image.open(BytesIO(encoded)) as thumbnail:
        return np.asarray(thumbnail.convert('RGB'), dtype=np.uint8)
