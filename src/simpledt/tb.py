from torch.utils.tensorboard import SummaryWriter


def log_dict_to_tensorboard(data, step, writer: SummaryWriter):
    for key, value in data.items():
        if isinstance(value, dict):
            writer.add_scalars(key, value, step)
        else:
            writer.add_scalar(key, value, step)
