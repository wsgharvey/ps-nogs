import argparse
import os
import time
import pickle

import torch

from progress.bar import Bar
from torch.optim import Adam

from mea.variational_cnn.architectures import get_densenet, get_mobilenet
from mea.variational_cnn.loader import get_train_valid_loaders
from mea.config import CNN_SAVE_PATH, CNN_LOG_PATH, DEFAULT_CNN_TRAINING_BATCH_SIZE, \
        DEFAULT_CNN_TRAINING_EPOCHS, DEFAULT_CNN_TRAINING_N_WORKERS, DEFAULT_CNN_TRAINING_LR, \
        DEFAULT_CNN_ARCHITECTURE
from mea.utils import AverageMeter, Logger, set_random_seed

parser = argparse.ArgumentParser(description='Attention CNN')
parser.add_argument('name', type=str,
                    help='Name for saving log and checkpoints.')
parser.add_argument('--attr', type=int, default=-1,
                    help='Attribute to predict (otherwise predicts all).')
parser.add_argument('--resume', type=str, default='',
                    help='Optional pickled model to restart from.')
parser.add_argument('--lr', type=float, default=DEFAULT_CNN_TRAINING_LR,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=DEFAULT_CNN_TRAINING_EPOCHS,
                    help='Number of epochs to train for.')
parser.add_argument('--batch-size', type=int, default=DEFAULT_CNN_TRAINING_BATCH_SIZE,
                    help='Batch size.')
parser.add_argument('--architecture', type=str, default=DEFAULT_CNN_ARCHITECTURE,
                    choices=['densenet', 'mobilenet'],
                    help='Neural net architecture to use.')
parser.add_argument('--n-workers', type=int, default=DEFAULT_CNN_TRAINING_N_WORKERS,
                    help='Number of workers for data loader.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
args = parser.parse_args()

MODEL_NAME = f"{args.name}_seed_{args.seed}"
LOG_NAME = os.path.join(CNN_LOG_PATH, MODEL_NAME+".txt")

# fun hyperparameters
DO_ATTENTION = True
ALL_CLASSES = (args.attr == -1)
CLASS = args.attr  # only applicable when ALL_CLASSES = False

# in-between
N_EPOCHS = args.epochs
RESUME = args.resume

# boring hyperparameters
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.n_workers
LR = args.lr
SEED = args.seed

# set random seed-----------------------------------------------------------------------------------
set_random_seed(SEED)

# load model if necessary-----------------------------------------------------------------------------
if RESUME:
    model = pickle.load(open(RESUME, 'rb'))
else:
    if args.architecture == 'densenet':
        model = get_densenet(
            num_labels=(40 if ALL_CLASSES else 1)
        )
    elif args.architecture == 'mobilenet':
        model = get_mobilenet(
            num_labels=(40 if ALL_CLASSES else 1)
        )
    else:
        raise Exception
model = model.cuda()

# make loaders, define loss and optimizer etc.------------------------------------------------------
train_loader, val_loader = get_train_valid_loaders(BATCH_SIZE,
                                                   NUM_WORKERS)

optimizer = Adam(
    model.parameters(),
    lr=1e-4
)


def loss_criterion(probs, y):
    return -torch.distributions.Bernoulli(probs)\
        .log_prob(y.type(torch.cuda.FloatTensor)).sum()/probs.numel()


# training-----------------------------------------------------------------------------------------
def train_one_epoch(epoch_number):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time()-end)

        x, y = x.cuda(), y.cuda()

        output = model(x)

        if not ALL_CLASSES:
            output = output.view(-1)
            y = y[:, CLASS]

        loss = loss_criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_hat = output > 0.5
        correct = y_hat == y.type(torch.cuda.BoolTensor)
        accuracy.update(correct.sum().item() / correct.numel())

        losses.update(loss.item())

        batch_time.update(time.time()-end)
        end = time.time()

        bar.suffix  = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.9f} | Acc: {acc:.2f}%%".format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=accuracy.avg*100
        )
        bar.next()
    bar.finish()
    return losses.avg


def validate():
    with torch.no_grad():
        model.eval()
        avg_log_prob = AverageMeter()
        for i, (x, y) in enumerate(val_loader):
            x, y = x.cuda(), y.cuda()
            output = model(x)
            if not ALL_CLASSES:
                output = output.view(-1)
                y = y[:, CLASS]
            loss = loss_criterion(output, y)
            avg_log_prob.update(loss.item())
    return avg_log_prob.avg


# log hyperparameters
logger = Logger(LOG_NAME, locals())

for epoch in range(N_EPOCHS):
    # do training
    torch.cuda.empty_cache()
    train_loss = train_one_epoch(epoch)
    torch.cuda.empty_cache()
    valid_loss = validate()

    # training info to log
    logger.add_epoch(epoch, train_loss, valid_loss)
    best_valid_loss = logger.got_best_valid_loss()

    # save if validation_loss is best or e.g epoch % 10 == 0
    if epoch % 10 == 0 or best_valid_loss:
        checkpoint_name = f"{MODEL_NAME}_epoch_{epoch}" + \
                          ("_BEST" if best_valid_loss else "") + \
                          ".p"
        file_name = os.path.join(CNN_SAVE_PATH, checkpoint_name)
        model.cpu()
        pickle.dump(model, open(file_name, 'wb'))
        model.cuda()
        logger.log_checkpoint(file_name)
