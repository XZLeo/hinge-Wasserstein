import matplotlib.pyplot as plt
import torch


def plot_gnd(img:torch.Tensor, left_y, left_x, right_y, right_x, name:str):
    '''
    plot horizon on the original img
    param: img: original img
    '''
    w, h = img.size
    # center coordinae => pixel coordinate
    def transfer(x, y):
        pixel_x = x + w/2
        pixel_y = h/2 - y
        return pixel_x, pixel_y
    pxl, pyl = transfer(left_x, left_y)
    pxr, pyr =  transfer(right_x, right_y)
    plt.figure()
    plt.imshow(img)
    plt.plot((pxl, pxr), (pyl, pyr), linewidth=1, color='red')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.axis('off')
    plt.savefig(f'img/{name}.pdf',dpi=1600, bbox_inches='tight', pad_inches=0)
    return


def plot_centerCrop(center_img:torch.Tensor, left_y, left_x, right_y, right_x, name:str):
    '''
    plot horizon on the centerCropped window
    param: img: square img after max centerCrop 
    '''
    img = center_img.permute(1, 2, 0)
    crop_size = min(img.shape[:-1])
    # center coordinae => pixel coordinate
    def transfer(x, y):
        pixel_x = x + crop_size/2
        pixel_y = crop_size/2 - y
        return pixel_x, pixel_y
    pxl, pyl = transfer(left_x, left_y)
    pxr, pyr =  transfer(right_x, right_y)
    plt.figure()
    plt.imshow(img)
    plt.plot((pxl, pxr), (pyl, pyr), linewidth=1)
    plt.savefig(f'img/{name}.jpg',dpi=1600)
    return


def plot_centerCrop_gnd_output(center_img:torch.Tensor, pred_coordinate, gnd_coordinate, name:str):
    '''
    plot horizon on the centerCropped window
    param: img: square img after max centerCrop 
    '''
    pred_left_y, pred_left_x, pred_right_y, pred_right_x = pred_coordinate
    gnd_left_y, gnd_left_x, gnd_right_y, gnd_right_x = gnd_coordinate
    img = center_img.squeeze().permute(1, 2, 0)
    crop_size = min(img.shape[:-1])
    # center coordinae => pixel coordinate
    def transfer(x, y):
        pixel_x = x + crop_size/2
        pixel_y = crop_size/2 - y
        return pixel_x, pixel_y
    pxl, pyl = transfer(pred_left_x, pred_left_y)
    pxr, pyr =  transfer(pred_right_x, pred_right_y)
    gxl, gyl = transfer(gnd_left_x, gnd_left_y)
    gxr, gyr =  transfer(gnd_right_x, gnd_right_y)
    plt.figure()
    plt.imshow(img.cpu())
    plt.plot((pxl, pxr), (pyl, pyr), 'r', linewidth=1, label='prediction')
    plt.plot((gxl, gxr), (gyl, gyr), 'b', linewidth=1, label='gnd')
    plt.legend()
    plt.savefig(f'img/{name}.jpg',dpi=1600)
    return
