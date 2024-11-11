import colorcet as cc
import cv2
from PIL import Image
import numpy as np, torchvision
import torch, os, json
from skimage.color import label2rgb
from PIL import ImageColor

def get_colors(n_labels):
    return [cc.cm.glasbey_bw_minc_20(i) for i in range(n_labels)]

def get_colors_alpha(colors):
    return [[i[0],i[1],i[2],i[3]/2] for i in colors]

def get_category_colors(colors):
    return {colors[i][:3]: i for i in range(len(colors))}

def visualize_label(label: np.array, 
                    img: np.array, 
                    label_to_visualize: np.array, 
                    concat:bool = False, 
                    axis:int = 1
                   ) -> Image:
    """
    visualize certain labels from mask

    Parameters
    ----------
        label:  Mask in shape [classes (159), width, height]
        img:    Image in shape [classes (159), width, height]
        concat: Whether to display image and visualization side by side
        axis:   axis at which image and visualization are shown side by side
        
    Returns
    ----------
        visualization: Label visualization as PIL Image
        
    """
    
    colors = get_colors(label.shape[0])
    colors_alpha = get_colors_alpha(colors)
    category_colors = get_category_colors(colors)

    out_mask = np.zeros((img.shape[0],img.shape[1],3)).astype(np.uint8)

    for i in label_to_visualize:
        imgray = (label[i,:,:]*255).astype(np.uint8)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        x = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in x:
            if (contour is not None) and (len(contour) > 0) and  (len(contour[0])>2) and (type(contour) == type(())):
                cv2.fillPoly(
                             out_mask, 
                             contour, 
                             # [int(j*255) for j in colors[i]],
                            [colors_alpha[i][0]*255,colors_alpha[i][1]*255,colors_alpha[i][2]*255,colors_alpha[i][3]],
                            )
        
    out_contour = np.zeros((img.shape[0],img.shape[1],3)).astype(np.uint8)
    for i in label_to_visualize:
        imgray = (label[i,:,:]*255).astype(np.uint8)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        x = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in x:
            if (contour is not None) and (len(contour) > 0) and  (len(contour[0])>2)  and (type(contour) == type(())):
                cv2.drawContours(out_contour, contour, -1, [int(j*255) for j in colors[i]], 2)
    
    out = cv2.addWeighted(out_contour, 1, out_mask, 0.7, 0.0)
    out = cv2.addWeighted(img, 0.5, out, 0.7, 0.0)
    
    if concat:
        return Image.fromarray(np.concatenate([img, out],axis)).convert('RGB')
    else:
        return Image.fromarray(out).convert('RGB')

def visualize_mask(class_names: list, 
                   mask:np.array,
                   image:np.array,
                   img_size: int,
                   cat:bool=True,
                   axis:int=1,
                  ) -> Image:
    """
    Resize image and label to desired size and visualize certain labels

    Parameters
    ----------
        class_names: List of classes of interest
        mask:  Mask in shape [classes (159), width, height]
        image:    Image in shape [3, width, height]
        img_size: Desired image size
        cat: Whether to display image and visualization side by side
        axis:   axis at which image and visualization are shown side by side
        
    Returns
    ----------
        visualization: Label visualization as PIL Image in desired size
        
    """
    
    colors = get_colors(mask.shape[0])
    colors_alpha = get_colors_alpha(colors)
    category_colors = get_category_colors(colors)
    
    # Resize image and label to desired size
    img = torch.nn.functional.interpolate(
                torch.tensor(image).float().unsqueeze(0),
                img_size,
                mode='bilinear'
            ).byte().numpy()[0]
        
        # reshape to match cv2 shapes
    img = np.transpose(img, [1,2,0])
    
    label = torch.nn.functional.interpolate(
                torch.tensor(mask).float().unsqueeze(0),
                img_size,
                mode='nearest'
            ).bool().numpy()[0]

    
    if type(class_names) == list:
        pass
    else:
        class_names = [class_names]
    
    label_to_visualize = np.concatenate([np.array(label_mapper[n]).flatten() for n in class_names]).flatten()
    
    return visualize_label(label, img,  label_to_visualize, cat, axis)

def visualize_from_file(class_names: list, 
                         img_path: str, 
                         label_path: str, 
                         img_size: int, 
                         cat:bool = True, 
                         axis:int = 1, 
                         do_store:bool = False,
                         out_dir:str = '',
                        ) -> Image:
    """
    Load Image and label, resize image and label to desired size, and visualize certain labels

    Parameters
    ----------
        class_names: list of class names to visualize
        img_path: path of the image file to visualize
        label_path: path of the label file to visualize
        img_size: size at which the image should be visualized
        cat: show image and label side by side
        axis: axis at which image and label are shown side by side
        do_store: boolean indicating whether to store the visualization in the out_dir 
                            with the associated label path and class_name
        out_dir: path at which to store visualization
        
    Returns
    ----------
        visualization: Pillow Image with labels overlaying original image
        
    """

    # Load image and label files
    img = get_img(img_path)
    label = get_label(label_path)
    # visualize desired labels
    visualization = visualize_mask(class_names, label, img, img_size, cat, axis)
    if do_store:
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        visualization.save(os.path.join(
            out_dir,
            '{}_{}.png'.format(os.path.basename(label_path).split('.')[0], '_'.join(class_names))))
    return visualization