"""
"""
import logging
import gin
import tensorflow as tf
import os
from vitaflow.models.image.east.east_model_v0 import EASTIEstimatorModel
from vitaflow.datasets.image.icdar.icdar_data import ICDARTFDataset, get_images
from vitaflow.deprecated.iterators import CIDARIterator
from executor import Executor
from vitaflow.models.image.east.prediction import *


# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

tf.app.flags.DEFINE_bool('predict', False, 'run_prediction')

FLAGS = tf.app.flags.FLAGS

@gin.configurable
def run(save_checkpoints_steps=10,
        keep_checkpoint_max=5,
        save_summary_steps=5,
        log_step_count_steps=5,
        num_epochs=5,
        test_iterator=False,
        test_images_dir="",
        output_dir=gin.REQUIRED):
    """
    """

    model = EASTIEstimatorModel()
    data_iterator = CIDARIterator()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
    run_config.allow_soft_placement = True
    run_config.log_device_placement = False
    model_dir = model.model_dir

    run_config = tf.estimator.RunConfig(session_config=run_config,
                                        save_checkpoints_steps=save_checkpoints_steps,
                                        keep_checkpoint_max=keep_checkpoint_max,
                                        save_summary_steps=save_summary_steps,
                                        model_dir=model_dir,
                                        log_step_count_steps=log_step_count_steps)

    executor = Executor(model=model,
                        data_iterator=data_iterator,
                        config=run_config,
                        train_hooks=None,
                        eval_hooks=None,
                        session_config=None)

    if test_iterator:
        executor.test_iterator()

    num_samples = data_iterator._num_train_examples
    batch_size = data_iterator._batch_size

    if not FLAGS.predict:
        for current_epoch in tqdm(range(num_epochs), desc="Epoch"):
            current_max_steps = (num_samples // batch_size) * (current_epoch + 1)
            print("\n\n Training for epoch {} with steps {}\n\n".format(current_epoch, current_max_steps))
            # executor.train(max_steps=None)
            print("\n\n Evaluating for epoch\n\n", current_epoch)
            # executor.evaluate(steps=None)
            executor.train_and_evaluate()
            executor.export_model(model_dir+"/exported/")
    else:
        estimator = executor._estimator

        images = get_images(test_images_dir)
        for image_file_path in images:
            print("================> Text segmentation on :", image_file_path)
            im = cv2.imread(image_file_path)[:, :, ::-1]
            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = resize_image(im)
            im_resized = np.expand_dims(im_resized, axis=0).astype(np.float32)

            def get_dataset():
                dataset = tf.data.Dataset.from_tensor_slices(({"images": im_resized},
                                                              np.ones_like(im_resized)))
                dataset = dataset.batch(batch_size=1)
                print(dataset.output_shapes)
                return dataset
            start = time.time()
            timer = {'net': 0, 'restore': 0, 'nms': 0}
            predict_fn = estimator.predict(input_fn=lambda: get_dataset())

            for prediction in predict_fn:
                score = prediction["f_score"]
                geometry = prediction["f_geometry"]

            score = np.expand_dims(score, axis=0)
            geometry = np.expand_dims(geometry, axis=0)

            print("===============================")
            print(score.shape)
            print(geometry.shape)
            print("===============================")
            print(score)
            print(geometry)

            timer['net'] = time.time() - start
            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
            print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                image_file_path, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            duration = time.time() - start_time
            print('[timing] {}'.format(duration))

            # save to file
            if boxes is not None:
                res_file = os.path.join(
                    output_dir,
                    '{}.txt'.format(
                        os.path.basename(image_file_path).split('.')[0]))

                with open(res_file, 'w') as f:
                    for box in boxes:
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                        f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                        ))
                        cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            # if not FLAGS.no_write_images:
            img_path = os.path.join(output_dir, os.path.basename(image_file_path))
            cv2.imwrite(img_path, im[:, :, ::-1])


if __name__ == "__main__":
    gin.parse_config_file('east_config.gin')
    obj = ICDARTFDataset()
    obj.run()
    run()