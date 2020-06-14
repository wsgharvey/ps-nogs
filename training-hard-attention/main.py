import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs
from hq_loader import get_train_celebhq_loader

def main(config, get_metadata=None):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # instantiate data loaders
    if config.dataset == 'celebhq':
        if config.mode == 'train':
            trainer, validator, _ = get_train_celebhq_loader(
                config.celebhq_image_dir, config.hq_attr_path,
                config.hq_sequences_dir, config.attr,
                config.n_optimal_seqs, config.fixed_attention_prop,
                config.anneal_epochs, config.celeba_crop_size,
                config.image_size, config.batch_size, config.mode,
                config.num_workers, config.valid_size, config.test_size,
                None, config.grayscale,
                config.celebhq_image_type)
            data_loader = (trainer, validator)
        else:
            _, _, data_loader = get_train_celebhq_loader(
                config.celebhq_image_dir, config.hq_attr_path,
                config.hq_sequences_dir, config.attr,
                config.n_optimal_seqs, config.fixed_attention_prop,
                config.anneal_epochs, config.celeba_crop_size,
                config.image_size, config.batch_size, config.mode,
                config.num_workers, config.valid_size, config.test_size,
                None, config.grayscale,
                config.celebhq_image_type)

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train, test or do plotting
    if config.test_seqs:
        assert config.mode == 'train'
        trainer.test_seqs()
    elif config.is_feedforward:
        assert config.mode == 'train', "Testing feedforward net not implemented yet"
        trainer.train_feedforward_classifier()
    elif config.kde or config.pde or config.plot_sequences:
        if config.kde:
            trainer.plot('kde')
        if config.pde:
            trainer.plot('pde')
        if config.plot_sequences:
            trainer.plot('sequences')
    elif config.mode == 'train':
        trainer.train()
    else:
        trainer.test()

if __name__ == '__main__':
    config = get_config()
    main(config)
