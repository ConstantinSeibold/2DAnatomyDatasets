import colorcet as cc
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageColor
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import cv2
import numpy as np, torchvision
import torch, os, json
from skimage.color import label2rgb

def get_colors(n_labels):
    return [cc.cm.glasbey_bw_minc_20(i) for i in range(n_labels)]

def get_colors_alpha(colors):
    return [(np.array([i[0],i[1],i[2],i[3]/2])*255).astype(np.uint8) for i in colors]

def get_category_colors(colors):
    return {i:(np.array(colors[i][:3])*255).astype(np.uint8) for i in range(len(colors))}

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

def visualize_coco_annotations_pil(image, annotations, coco, show_class_name=True, show_bbox=True):
    """
    Visualizes COCO mask annotations for a given image using PIL.

    Parameters:
        image (PIL.Image): The image to display the annotations on.
        annotations (list): List of annotations for the image (from COCO).
        coco (COCO): COCO object instance for loading annotations and categories.
        show_class_name (bool): If True, displays the class name for each annotation.
        show_bbox (bool): If True, displays the bounding box for each annotation.

    Returns:
        PIL.Image: Image with overlaid mask annotations.
    """
    # Convert PIL image to RGBA format for transparency handling
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))  # transparent overlay
    
    
    colors = get_colors(coco.loadCats(cat_id))
    colors_alpha = get_colors_alpha(colors)
    category_colors = get_category_colors(colors)
    
    for ann in annotations:
        # Get mask
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rle = maskUtils.frPyObjects(ann['segmentation'], image.height, image.width)
                mask = maskUtils.decode(rle)
            else:
                # RLE format
                mask = maskUtils.decode(ann['segmentation'])
        else:
            print("No segmentation found in annotation")
            continue

        mask = np.squeeze(mask)
        # import pdb;pdb.set_trace()
        # Draw mask with transparency
        color = colors_alpha[ann["category_id"]] #np.random.randint(0, 255, 3).tolist() + [128]  # Random color with 50% transparency
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        colored_mask = Image.new("RGBA", image.size, tuple(color))
        overlay.paste(colored_mask, (0, 0), mask_img)

        # Draw contours
    for ann in annotations:
        # Get mask
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rle = maskUtils.frPyObjects(ann['segmentation'], image.height, image.width)
                mask = maskUtils.decode(rle)
            else:
                # RLE format
                mask = maskUtils.decode(ann['segmentation'])
        else:
            print("No segmentation found in annotation")
            continue

        mask = np.squeeze(mask)
        
        color = category_colors[ann["category_id"]] 
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw = ImageDraw.Draw(overlay)
        for contour in contours:
            contour = contour[:, 0, :]  # Reshape for plotting
            contour_points = [(int(x), int(y)) for x, y in contour]
            draw.line(contour_points + [contour_points[0]], fill=tuple(color), width=2)

        # Show bounding box if requested
        if show_bbox and 'bbox' in ann:
            bbox = ann['bbox']
            x, y, w, h = bbox
            draw.rectangle([(x, y), (x + w, y + h)], outline=tuple(color), width=2)

        # Show class name if requested
        if show_class_name:
            cat_id = ann['category_id']
            category = coco.loadCats(cat_id)[0]['name']
            draw.text((x, y), category, fill=tuple(color))

    # Composite overlay with the original image
    annotated_image = Image.alpha_composite(image, overlay)
    return annotated_image.convert("RGB")  # Convert back to RGB for displaying without transparency issues

def visualize_multiclass(image, mask, label_dict):
    """
    Visualize multiclass segmentation by overlaying a segmentation mask on the input image.

    Args:
        image (np.ndarray or PIL.Image.Image): Input image to overlay the mask on.
        mask (np.ndarray): Segmentation mask (H x W) where each pixel value corresponds to a class index.
        label_dict (dict): Dictionary mapping class indices to class labels.

    Returns:
        np.ndarray: Image with segmentation mask overlay and optional class labels.
    """
    # Ensure the image is in RGB format
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Generate colors
    n_classes = len(label_dict)
    colors = get_colors(n_classes)
    colors_alpha = get_colors_alpha(colors)

    # Create an RGBA overlay for the mask
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for label, color in enumerate(colors_alpha):
        overlay[mask == label] = np.array(color) * 255  # Apply color to each class

    # Composite the overlay with the original image
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)  # Drop alpha for display
    combined = cv2.addWeighted(image, 0.6, overlay_rgb, 0.4, 0)

    return Image.fromarray(combined)