import os
import random
import skimage
import Model_MRCNN as modellib
import coco


def Model_AAMaskRCNN(Data, sol=None):
    if sol is None:
        sol = [5, 5, 300]
    # Root directory of the project
    # ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = "logs"

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        # GPU_COUNT = 0 for CPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object.
    model = modellib.MaskRCNN(sol, model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    # model.load_state_dict(torch.load(COCO_MODEL_PATH))

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl','Leaf', 'banana', 'Tomato','apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # Load a random image from the images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(Data, random.choice(class_names)))

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]
    return r
