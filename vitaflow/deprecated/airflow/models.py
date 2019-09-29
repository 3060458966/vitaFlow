from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from vitaflow.pipeline.models.vf_east_model import east_flow_predictions


class EastModelOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder', 'model_dir')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 model_dir,
                 *args, **kwargs):
        super(EastModelOperator, self).__init__(*args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.model_dir = model_dir

    def execute(self, context):
        east_flow_predictions(input_dir=self.source_folder,
                              output_dir=self.destination_folder,
                              model_dir=self.model_dir)
