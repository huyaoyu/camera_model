
# Prepare the Python environment.
import os
import sys

_CF       = os.path.realpath(__file__)
_CD       = os.path.dirname(_CF)
_PKG_PATH = os.path.dirname(os.path.dirname(_CD))

sys.path.insert(0, _PKG_PATH)

# System packages.
import cv2
import numpy as np

# Tested packages.
from camera_model.double_sphere import DoubleSphere

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

def add_border(img, thickness=1):
    H, W = img.shape[0:2]

    img = np.copy(img)

    cv2.line( img, (   0,   0 ), ( W-1,   0 ), ( 0, 255, 0 ), thickness=thickness )
    cv2.line( img, ( W-1,   0 ), ( W-1, H-1 ), ( 0, 255, 0 ), thickness=thickness )
    cv2.line( img, ( W-1, H-1 ), (   0, H-1 ), ( 0, 255, 0 ), thickness=thickness )
    cv2.line( img, (   0, H-1 ), (   0,   0 ), ( 0, 255, 0 ), thickness=thickness )

    return img

def add_circles(img, xy, radius=1):
    img = np.copy(img)

    for i in range( xy.shape[1] ):
        cv2.circle( img, ( int(np.round(xy[0, i])), int(np.round(xy[1, i])) ), 
            radius=radius,
            color=(0, 255, 0),
            thickness=-1 )

    return img

def test_sample_pinhole_image():
    global _CF
    in_fisheye_img_fn = os.path.join( _CD, 'data', '16960_top0_img.png' )

    # Read the image.
    in_img = read_image(in_fisheye_img_fn)

    # Forward direction.
    fov_x = 60 # Degree.
    fov_y = 40 # Degree.

    fov_shape = ( 200, 300 ) # ( H, W )

    ax = np.linspace(-1, 1, fov_shape[1]).astype(np.float32) * fov_x / 2
    ay = np.linspace(-1, 1, fov_shape[0]).astype(np.float32) * fov_y / 2

    # From degree to rad.
    ax = ax / 180 * np.pi
    ay = ay / 180 * np.pi

    axx, ayy = np.meshgrid( ax, ay )

    x = np.tan( axx )
    y = np.tan( ayy )
    z = np.full( axx.shape, 1.0, dtype=x.dtype )

    points = np.stack( (x, y, z), axis=0 )

    rot_mat = np.zeros( (3, 3), dtype=x.dtype )
    rot_mat[ 0, 0 ] = 1
    rot_mat[ 1, 2 ] = 1
    rot_mat[ 2, 1 ] = -1

    points = rot_mat @ points.reshape( (3, -1) )
    points = points.reshape( ( 3, *fov_shape ) )

    import ipdb; ipdb.set_trace()

    # The camera model.
    camera_model = DoubleSphere(
        fx = 156.96507623,
        fy = 157.72873153,
        cx = 343,
        cy = 343,
        shape = (687, 687),
        xi = -0.17023409,
        alpha=0.59679147
    )

    sampled, mask = camera_model.sample_image( in_img, points )

    # Reshape.
    sampled = sampled.reshape( (*fov_shape, -1) )
    mask = mask.reshape( fov_shape )

    # Use gray as the invalid value.
    sampled[np.logical_not(mask), ...] = 128

    # Add image border.
    sampled_with_border = add_border( sampled, thickness=1 )

    # Write the sampled image.
    fn = os.path.join( _CD, 'output', 'sampled_pinhole.png' )
    cv2.imwrite( fn, sampled_with_border )

    # Write the mask.
    fn = os.path.join( _CD, 'output', 'sampled_pinhole_mask.png' )
    cv2.imwrite( fn, mask.astype(np.uint8) * 255 )

    # Get the sample border.
    border = np.concatenate( ( 
        points[:,  0,  :],
        points[:,  :, -1], 
        points[:, -1,  :],
        points[:,  :,  0]
     ), axis=1 )

    border_uxy, _ = camera_model.project_3d_2_image_plane(border)

    border_uxy = ( border_uxy + 1 ) / 2
    border_uxy[0, :] = border_uxy[0, :] * ( camera_model.shape[1] - 1 )
    border_uxy[1, :] = border_uxy[1, :] * ( camera_model.shape[0] - 1 )

    # Add circles to the input image.
    in_img_with_circles = add_circles( in_img, border_uxy )

    # Save the annotated input image.
    fn = os.path.join( _CD, 'output', 'annotated_input_fisheye.png' )
    cv2.imwrite(fn, in_img_with_circles)

if __name__ == '__main__':
    print( 'Hello, %s! ' % ( os.path.basename( __file__ ) ) )

    test_sample_pinhole_image()