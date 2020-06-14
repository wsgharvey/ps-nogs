import argparse
import os
import time
import pickle

import torch
import torch.nn as nn
from progress.bar import Bar

from mea.variational_cnn.birds.loader import get_train_valid_loaders
from mea.variational_cnn import get_densenet, get_mobilenet
from mea.config import BIRDS_N_ATTRIBUTES, \
    BIRDS_N_CLASSES, BIRDS_CNN_SAVE_PATH, BIRDS_CNN_LOG_PATH, \
    DEFAULT_CNN_TRAINING_BATCH_SIZE, DEFAULT_CNN_TRAINING_EPOCHS, \
    DEFAULT_CNN_TRAINING_N_WORKERS, DEFAULT_CNN_TRAINING_LR, \
    DEFAULT_CNN_ARCHITECTURE
from mea.utils import AverageMeter, Logger, set_random_seed


parser = argparse.ArgumentParser(description='Attention CNN')
parser.add_argument('name', type=str,
                    help='Name for saving log and checkpoints.')
parser.add_argument('--predict', type=str,
                    default='both',
                    choices=['attributes', 'class', 'both'],
                    help='What to predict.')
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
parser.add_argument('--optim', type=str, default='Adam',
                    choices=['SGD', 'Adam'],
                    help='Optimiser.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
args = parser.parse_args()


MODEL_NAME = f"{args.name}_seed_{args.seed}"
LOG_NAME_ATTR = os.path.join(BIRDS_CNN_LOG_PATH, f"{MODEL_NAME}_attr.txt")
LOG_NAME_CLASS = os.path.join(BIRDS_CNN_LOG_PATH, f"{MODEL_NAME}_class.txt")

PREDICT_ATTRS = args.predict in ['attributes', 'both']
PREDICT_CLASS = args.predict in ['class', 'both']
DO_ATTENTION = True
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.n_workers
LR = args.lr
OPTIMISER = args.optim
SEED = args.seed

# set random seed-----------------------------------------------------------------------------------
set_random_seed(SEED)

# initialise model - use 1 so we will get returned a dict if we are not using attr
n_attrs = BIRDS_N_ATTRIBUTES if PREDICT_ATTRS else 1
n_classes = BIRDS_N_CLASSES if PREDICT_CLASS else 1
if args.architecture == 'densenet':
    model = get_densenet(
        num_labels=n_attrs,
        num_classes=n_classes
    )
elif args.architecture == 'mobilenet':
    model = get_mobilenet(
        num_labels=n_attrs,
        num_classes=n_classes
    )
model = model.cuda()

# make loaders, define loss and optimizer etc.------------------------------------------------------
train_loader, valid_loader = get_train_valid_loaders('all',
                                                     BATCH_SIZE,
                                                     NUM_WORKERS)

optimiser = torch.optim.Adam if OPTIMISER == 'Adam' \
    else torch.optim.SGD
optimiser = optimiser(
    model.parameters(),
    lr=LR
)

def attr_loss_criterion(probs, y):
    return -torch.distributions.Bernoulli(probs)\
        .log_prob(y.type(torch.cuda.FloatTensor)).sum()/probs.numel()
def class_loss_criterion(logprobs, y):
    return nn.NLLLoss(reduction='mean')(logprobs, y)


# training-----------------------------------------------------------------------------------------
def train_one_epoch(epoch_number):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    attr_losses = AverageMeter()
    class_losses = AverageMeter()
    class_accuracy = AverageMeter()

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time()-end)
        x, y = x.cuda(), y.cuda()

        output = model(x)

        if PREDICT_ATTRS:
            pred = output['sigmoid']
            attr_y = y[:, :-1]
            attr_loss = attr_loss_criterion(pred, attr_y)
        else:
            attr_loss = torch.tensor(0.).cuda()
        if PREDICT_CLASS:
            pred = output['softmax']
            class_y = y[:, -1]
            class_loss = class_loss_criterion(pred, class_y)
            correct = torch.argmax(pred, dim=1) == class_y
        else:
            class_loss = torch.tensor(0.).cuda()
            correct = torch.tensor(0.)
        loss = attr_loss + class_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        class_accuracy.update(correct.sum().item() / correct.numel())
        attr_losses.update(attr_loss.item())
        class_losses.update(class_loss.item())

        batch_time.update(time.time()-end)
        end = time.time()

        bar.suffix  = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | AttrLoss: {al:.9f} | ClassLoss: {cl:.9f} | Acc: {acc:.2f}%%".format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            al=attr_losses.avg,
            cl=class_losses.avg,
            acc=class_accuracy.avg*100
        )
        bar.next()
    bar.finish()
    return attr_losses.avg, class_losses.avg


def validate():
    with torch.no_grad():
        model.eval()
        attr_log_prob = AverageMeter()
        class_log_prob = AverageMeter()
        for i, (x, y) in enumerate(valid_loader):
            x, y = x.cuda(), y.cuda()
            output = model(x)

            if PREDICT_ATTRS:
                pred = output['sigmoid']
                attr_y = y[:, :-1]
                attr_loss = attr_loss_criterion(pred, attr_y)
            else:
                attr_loss = torch.tensor(0.)
            if PREDICT_CLASS:
                pred = output['softmax']
                class_y = y[:, -1]
                class_loss = class_loss_criterion(pred, class_y)
            else:
                class_loss = torch.tensor(0.)
            attr_log_prob.update(attr_loss.item())
            class_log_prob.update(class_loss.item())
    return attr_log_prob.avg, class_log_prob.avg

# log hyperparameters
logger_attr = Logger(LOG_NAME_ATTR, locals())
logger_class= Logger(LOG_NAME_CLASS, locals())

for epoch in range(N_EPOCHS):
    # do training
    torch.cuda.empty_cache()
    attr_train_loss, class_train_loss = train_one_epoch(epoch)
    torch.cuda.empty_cache()
    attr_valid_loss, class_valid_loss = validate()

    # training info to log
    logger_attr.add_epoch(epoch, attr_train_loss, attr_valid_loss)
    logger_class.add_epoch(epoch, class_train_loss, class_valid_loss)
    best_valid_loss = logger_class.got_best_valid_loss()

    # save if validation_loss is best or e.g epoch % 10 == 0
    if epoch % 10 == 0 or best_valid_loss:
        checkpoint_name = f"{MODEL_NAME}_epoch_{epoch}" + \
                          ("_BEST" if best_valid_loss else "") + \
                          ".p"
        file_name = os.path.join(BIRDS_CNN_SAVE_PATH, checkpoint_name)
        model.cpu()
        pickle.dump(model, open(file_name, 'wb'))
        model.cuda()
        logger_class.log_checkpoint(file_name)
