from GRU.model import GRUModel
import config
import torch


def get_best_checkpoint_path():
    checkpoint_dir = ''
    checkpoint_dir = './GRU/model'
    return f'{checkpoint_dir}/best_GRU_centercrop.ckpt'


model = GRUModel.load_from_checkpoint(
            checkpoint_path=get_best_checkpoint_path(),
            input_dim=224,
            hidden_dim=1000,
            layer_dim=3,
            output_dim=3,
            learning_rate=0,
        )

# Save model
torch.save(model.state_dict(), f'{config.PROJECT_PATH}/GRU/model/GRU.pth')
