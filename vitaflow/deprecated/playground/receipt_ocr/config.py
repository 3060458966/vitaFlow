import os

# user inputs
experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow")
path_to_tensorflow_model = os.path.join(experiment_root_directory, "ObjectDetection", "_model", "frozen_inference_graph.pb")
images_src = os.path.join(experiment_root_directory, "ObjectDetection", "test_images")

# user output
images_dest = os.path.join(experiment_root_directory, "ObjectDetection", "predictions")
text_out_dir = os.path.join(experiment_root_directory, "ObjectDetection", "extracted_text")
receipt_text_dir = os.path.join(experiment_root_directory, "ObjectDetection", "receipt_text_dir")
category_index = \
    {1: {'id': 1, 'name': 'merchant'},
     2: {'id': 2, 'name': 'receipt_number'},
     3: {'id': 3, 'name': 'date'},
     4: {'id': 4, 'name': 'line_items_and_value'},
     5: {'id': 5, 'name': 'total'},
     6: {'id': 6, 'name': 'tax'},
     7: {'id': 7, 'name': 'mode_of_payment'}}
