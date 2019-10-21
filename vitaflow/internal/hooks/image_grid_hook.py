# Copyright 2019 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Image Grid Hook for GAN Models
"""
import os
from tensorflow.python.training import session_run_hook

from vitaflow.utils.image_utils import images_square_grid
from vitaflow.utils.print_helper import print_info


class ImageGridHook(session_run_hook.SessionRunHook):
    def __init__(self,
                 z_image,
                 d_loss,
                 g_loss,
                 global_Step,
                 store_interval_steps=3,
                 log_interval_steps=5,
                 path=None):
        """

        :param z_image: Generator _model tensor reference
        :param d_loss: Discriminator loss tensor
        :param g_loss: Generator loss tensor
        :param global_Step: Tenosrfloe global step refererence
        :param store_interval_steps: Number of steps interval to store the image as a grid
        :param log_interval_steps: Number of steps interval to log the losses on the terminal
        :param path: Path to store the image grids
        """
        self._z_image = z_image
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._global_Step = global_Step
        self._path = path
        self._store_interval_steps = store_interval_steps
        self._log_interval_steps = log_interval_steps


    def before_run(self, run_context):

        if self._path is None:
            self._path = os.path.join(os.path.expanduser("~"), "vitaFlow/runtime/GAN")

        global_step = run_context.session.process(self._global_Step)

        print_info("global_step {}".format(global_step))

        if global_step % self._store_interval_steps == 0: #store every n steps
            samples = run_context.session.process(self._z_image)
            channel = self._z_image.get_shape()[-1]

            if channel == 1:
                images_grid= images_square_grid(samples, "L")
            else:
                images_grid= images_square_grid(samples, "RGB")

            if not os.path.exists(self._path):
                os.makedirs(self._path)

            images_grid.save(os.path.join(self._path, 'step_{}.png'.format(global_step)))

        if global_step % self._log_interval_steps == 0:
            dloss, gloss = run_context.session.process([self._d_loss, self._g_loss])
            print_info("\nDiscriminator Loss: {:.4f}... Generator Loss: {:.4f}".format(dloss, gloss))