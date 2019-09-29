import gin


@gin.configurable
def get_experiment_root_directory(value):
    return value
