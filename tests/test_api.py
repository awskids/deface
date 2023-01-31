import imageio.v2 as imageio
import pytest

from deface.centerface import CenterFace
from deface.deface import image_object_detect, video_detect, factory


@pytest.mark.skip(reason="longer running")
def test__video_run():
    # ipath = "../frigate-with-ai/debug/cctv.mp4"   # 34 seconds
    ipath = "../frigate-with-ai/debug/first_clip.mp4"  # 14 seconds

    backend = "onnxrt"  # ["auto", "onnxrt", "opencv"]

    centerface = CenterFace(in_shape=None, backend=backend)

    video_detect(
        ipath=ipath,
        opath=None,
        centerface=centerface,
        threshold=0.2,
        cam=False,
        replacewith="blur",
        mask_scale=1.3,
        ellipse=True,
        draw_scores=True,
        enable_preview=False,
        nested=False,
        ffmpeg_config={"codec": "libx264"},
        replaceimg="../frigate-with-ai/debug/amsa-black.png"
    )


def test__image_object():
    cf = factory()
    ipath = "../frigate-with-ai/debug/IMG_0809.jpg"
    opath = "../frigate-with-ai/debug/defaced_image.jpg"
    frame = imageio.imread(ipath)
    mod_frame = image_object_detect(frame, cf)
    mod_frame = image_object_detect(frame, cf)
    mod_frame = image_object_detect(frame, cf)
    imageio.imsave(opath, mod_frame)

"""
Successfully installed 
PyWavelets-1.4.1 deface-0.0.1 imageio-2.25.0 
imageio-ffmpeg-0.4.8 networkx-3.0 opencv-python-4.7.0.68 
scikit-image-0.19.3 tifffile-2023.1.23.1 tqdm-4.64.1
"""