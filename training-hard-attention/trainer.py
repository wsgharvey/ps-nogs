import torch
import torch.nn.functional as F

import torch.optim as optim

import os
from os.path import join, splitext
import time
import shutil
import itertools as it

from tqdm import tqdm
from nn import HardAttentionNetwork
from utils import ScoreMeter, AverageMeter, to_pil, \
    BatchScoreMeter, BestTracker, iters_to_minimum
from hq_loader import anneal_hq_loader
from location import Location

from experiment_utils import get_model_name, save_config

class Trainer(object):
    """
    Trains the hard attention network.
    """
    tracked_metrics = {'acc': 'greater',
                       'balanced_acc': 'greater',
                       'nll': 'lesser'}
    @staticmethod
    def get_metrics(score_meter, nll):
        return {'acc': score_meter.accuracy(),
                'balanced_acc': score_meter.balanced_accuracy(),
                'nll': nll}
    @staticmethod
    def tag_for_best_valid(metric):
        return 'best_valid_'+metric

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config
        self.best_tracker = BestTracker(self.tracked_metrics)  # if resuming, this will be initialised wrongly

        # glimpse network params
        self.glimpse_size = config.glimpse_size
        self.num_glimpses = config.num_glimpses

        # reinforce params
        self.M = 5

        # data params
        if config.mode == 'train':
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler)
            self.num_valid = len(self.valid_loader.sampler)
        else:
            self.test_loader = data_loader
        self.num_classes = config.num_classes
        self.num_channels = 1 if (config.grayscale or config.dataset == 'mnist') else 3

        # training params
        self.epochs = config.epochs
        self.attention_target_weight = config.attention_target_weight
        self.entropy_regularizer = config.entropy_regularizer
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.save_model_every = 100  # how often to save checkpoint (in iterations)
        self.valid_every = 100       # how often to validate (in iterations)

        # misc params
        self.logs_dir = config.logs_dir
        self.resume = config.resume
        self.image_size = config.image_size
        self.model_name = get_model_name(config)

        self.model_ckpt_dir = config.ckpt_dir + '/' +  self.model_name + '/'
        if not os.path.exists(self.model_ckpt_dir):
            os.makedirs(self.model_ckpt_dir)

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # construct file paths
        self.experiment_dir = join(self.logs_dir, self.model_name)
        self.train_results_path = join(self.experiment_dir, 'train.csv')
        self.valid_results_path = join(self.experiment_dir, 'valid.csv')

        # build neural net
        self.model = HardAttentionNetwork(config.num_glimpses,
                                          hidden_dim=config.hidden_size,
                                          conv_depth=config.conv_depth,
                                          conv_type=config.glimpse_net_type,
                                          classifier_size=config.classifier_size,
                                          num_classes=config.num_classes,
                                          glimpse_size=config.glimpse_size,
                                          loc_dist_type=config.dist_type,
                                          loc_mixture_components=config.loc_mixture_components,
                                          loc_std=config.std,
                                          heuristic_dist_path=config.data_dir)

        self.cuda = config.cuda
        if self.cuda:
            self.model.cuda()
            assert config.glimpse_net_type in available_baselines

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.init_lr,
        )

    @property
    def num_test(self):
        return len(self.test_loader.sampler)

    def train(self):
        print("\n[*] Train on {} samples, validate on {}".format(
                  len(self.train_loader.sampler),
                  len(self.valid_loader.sampler))
        )

        # set up save paths and iteration counter, load model if required
        if not self.resume:
            self.iter_count = 0
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            def make_empty_file(path):
                f = open(path, 'w')
                f.close()
            make_empty_file(self.train_results_path)
            make_empty_file(self.valid_results_path)
            # save params
            save_config(self.config, join(self.experiment_dir, 'config.txt'))
        else:
            self.load_checkpoint('recent')
            def truncate_file_to_iter(path, iters_at_first_write, iters_per_write):
                contents = open(path, 'r').readlines()
                min_num_lines = 1 + int((self.iter_count-iters_at_first_write) / iters_per_write)
                max_extra_lines = int(self.save_model_every / iters_per_write)
                max_num_lines = min_num_lines + max_extra_lines
                assert len(contents) >= min_num_lines and len(contents) <= max_num_lines
                with open(path, 'w') as f:
                    for line in contents[:min_num_lines]:
                        f.write(line)
            truncate_file_to_iter(self.train_results_path, 1, 1)
            truncate_file_to_iter(self.valid_results_path, 0, self.valid_every)
            print(f"Resuming at iteration {self.iter_count}.")

        for epoch in range(self.start_epoch, self.epochs):
            print(f'\nEpoch: {epoch+1}/{self.epochs} - LR: {self.lr:.6f}')
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        loss_actions = AverageMeter()
        score_meter = ScoreMeter()

        accs = AverageMeter()
        optimal_sequences_seen = 0

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, data_batch in enumerate(self.train_loader):
                if self.iter_count % self.save_model_every == 0:
                    # save recent copy of model (overwriting previous 'recent' copy)
                    self.save_checkpoint(
                        epoch=round(epoch + i/len(self.train_loader), 2),
                        tags=['recent'])
                if self.iter_count % self.valid_every == 0:
                    self.validate(epoch + i/len(self.train_loader))

                x, y, has_targets, attention_target_pixels = data_batch

                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    has_targets = has_targets.cuda()
                    attention_target_pixels = attention_target_pixels.cuda()

                attention_targets = Location(attention_target_pixels.cpu(), 'pixel')
                y = y.squeeze(1)
                optimal_sequences_seen += has_targets.sum().item()

                # run neural net
                outputs = self.model(["prediction_probs", "baselines",
                                      "selected_actions_prob", "entropies", ],
                                     x, replace_l=has_targets,
                                     replacement_l=attention_targets)

                log_pi = outputs["selected_actions_prob"]
                log_pi_targets = (log_pi * has_targets.view(-1, 1).float()).sum()

                predicted = torch.max(outputs["prediction_probs"][:, -1], 1)[1]

                # compute losses for differentiable modules

                # define loss_action as prediction loss after all time steps
                loss_action = F.nll_loss(outputs["prediction_probs"][:, -1], y)

                # calculate reward (accuracy) based on final prediction
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)  # B x T

                loss_baseline = F.mse_loss(outputs["baselines"], R)

                # loss entropy regularizer
                entropy = outputs["entropies"].sum(dim=1).mean(dim=0)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - outputs["baselines"].detach()
                loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce*(1-has_targets).float(), dim=0)
                # sum up into a hybrid loss
                loss = loss_action + \
                       loss_baseline + \
                       loss_reinforce + \
                       -self.attention_target_weight * log_pi_targets + \
                       -self.entropy_regularizer * entropy

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.cuda:
                    loss = loss.cpu()
                    loss_action = loss_action.cpu()
                    predicted = predicted.cpu()
                    y = y.cpu()

                # compute accuracy
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.size()[0])
                loss_actions.update(loss_action.data.item(), x.size()[0])
                score_meter.update(y, predicted)
                accs.update(acc.item(), x.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss.data.item(), acc.data.item())))
                batch_size = x.shape[0]
                pbar.update(batch_size)

                self.iter_count += 1

            # save the losses and accuracies
            self.save_losses("train", score_meter, losses.avg, loss_actions.avg)

            if self.config.dataset == 'celebhq':
                anneal_hq_loader(self.train_loader)
            print(f"Trained one epoch with {optimal_sequences_seen} optimal sequences.")

    def eval_on(self, data_loader):
        """
        Evaluate the model on a given data loader. Returns losses, nll, score_meter
        and a histogram of stop times for each stopping time as well as locations
        attended to
        """
        locs = Location(torch.zeros(0, self.config.num_glimpses).long(), "index")
        all_prediction_probs = []
        all_ys = []

        for i, (x, y, _, _) in enumerate(data_loader):
            with torch.no_grad():
                # duplicate to sample multiple trajectories
                x = x.repeat(self.M, 1, 1, 1)
                y = y.squeeze(1).repeat(self.M)
                # run neural net
                outputs = self.model(["prediction_probs",
                                      "locations",], x)
                locs = locs.concatenate(outputs["locations"], dim=0)
                all_prediction_probs.append(outputs["prediction_probs"])
                all_ys.append(y)

        all_prediction_probs = torch.cat(all_prediction_probs, dim=0)
        all_ys = torch.cat(all_ys, dim=0)

        stop_times = list(range(1, self.num_glimpses+1))
        for stop_type, \
            param in zip(it.repeat("fixed"), stop_times):

            assert stop_type == "fixed"
            prediction_probs = all_prediction_probs[:, param-1]
            stop_histogram = [0.]*self.num_glimpses
            stop_histogram[param-1] = 1.

            predicted = torch.max(prediction_probs, 1)[1]
            nll = F.nll_loss(prediction_probs, all_ys).data.item()

            score_meter = ScoreMeter()
            score_meter.update(all_ys, predicted)
            locs = locs.concatenate(outputs["locations"], dim=0)

            yield (stop_type, param), \
                (0., nll, score_meter, locs, stop_histogram)

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        score_meters = [None]*(self.num_glimpses+1)
        nlls = [None]*(self.num_glimpses+1)

        # save losses etc. to file
        for (stop_type, param),\
            (loss, nll, score_meter, _, _) in self.eval_on(self.valid_loader):

            if stop_type != 'fixed':
                continue

            assert stop_type == "fixed"
            score_meters[param] = score_meter
            nlls[param] = nll
            self.save_losses("valid", score_meter, loss, nll,
                             param=param)
            if param == self.num_glimpses:
                self.save_losses("valid", score_meter, loss, nll,
                                 param=None)
                final_score_meter, final_nll = score_meter, nll

        # update best_tracker and save appropriate checkpoints
        tags = []
        for metric, value in self.get_metrics(final_score_meter,
                                              final_nll).items():
            # update tracker, add tag and save metrics to file if new best
            if self.best_tracker.update(metric, value):
                tags.append(self.tag_for_best_valid(metric))
                self.save_metrics(final_score_meter, final_nll,
                                  self.tag_for_best_valid(metric)+".txt")
                for t in range(1, self.num_glimpses+1):
                    self.save_metrics(score_meters[t], nlls[t],
                                      self.tag_for_best_valid(metric)+f"_T_{t}.txt")
                if metric == 'nll':
                    print("Best NLL")
        self.save_checkpoint(
            epoch=round(epoch, 2),
            tags=tags
        )

        print(f"VALIDATION:   nll: {nll}   acc: {final_score_meter.accuracy()}" \
            + f"   balanced_acc: {final_score_meter.balanced_accuracy()}")



    def test(self):
        """
        Load and test checkpoints that scored highest on each metric.
        """
        print("Testing with ", len(self.test_loader.sampler), " examples")

        # for a single training run, run all of the saved tags on held out test data
        for metric in self.tracked_metrics:

            # load best checkpoint for this metric
            self.load_checkpoint(self.tag_for_best_valid(metric))

            # run on held out test data and save stuff in score meter
            for (stop_type, param), \
                (loss, nll, score_meter, locations, stop_histogram) \
                in self.eval_on(self.test_loader):

                # save results (for all metrics every time)
                param_name = 'laziness' if stop_type == 'adaptive' else 'T'
                self.save_metrics(score_meter, nll,
                                  f"test_best_{metric}_{param_name}_{param}.txt")

            # locations are returned the same each time so only save once per metric
            self.save_locs(locations,
                           f"test_best_{metric}_locs.pt")

    def save_checkpoint(self, epoch, tags):
        """
        saves copies of checkpoint for whatever tags are passed
         - e.g. 'epoch_9' or 'best_acc'
        """
        if len(tags) == 0:
            return
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'iter_count': self.iter_count,
                 'best_tracker': self.best_tracker}
        extension = '.pt'
        temp_name = 'temp' + extension
        temp_path = join(self.model_ckpt_dir, temp_name)
        torch.save(state, temp_path)
        for tag in tags:
            tagged_name = tag + extension
            tagged_path = join(self.model_ckpt_dir, tagged_name)
            shutil.copyfile(temp_path, tagged_path)

    def load_from_path(self, path):
        print("[*] Loading model {}".format(path))
        ckpt = torch.load(path)
        # load variables from checkpoint
        self.start_epoch = int(ckpt['epoch'])
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.iter_count = ckpt['iter_count']
        self.best_tracker = ckpt['best_tracker']
        print("[*] Loaded {} checkpoint @ epoch {}".format(
            path, ckpt['epoch']))

    def load_checkpoint(self, tag):
        filename = tag + '.pt'
        ckpt_path = join(self.model_ckpt_dir, filename)
        self.load_from_path(ckpt_path)

    def save_losses(self, mode, score_meter, total_loss, nll_loss,
                    param=None):
        """
        mode is 'valid' or 'train'
        """
        assert mode in ['train', 'valid']
        save_string = score_meter.to_csv(total_loss, nll_loss)
        filepath = self.valid_results_path if mode == 'valid' \
            else self.train_results_path
        if param is not None:
            path, ext = splitext(filepath)
            filepath = path + f"_{param}" + ext
        open(filepath, 'a').write(save_string)

    def save_metrics(self, score_meter, nll, fname):
        results_txt = "\n".join(f"{metric},{value}" for metric, value
                                in self.get_metrics(score_meter, nll).items())
        results_txt += "\niteration,"+str(self.iter_count)
        open(join(self.experiment_dir, fname), 'w').write(results_txt)

    def save_locs(self, locations, fname):
        path = join(self.experiment_dir, fname)
        locations.save(path)

    def save_stops(self, stop_histogram, fname):
        path = join(self.experiment_dir, fname)
        txt = '\n'.join(map(str, stop_histogram))
        open(path, 'w').write(txt)

    def visualise_dataset(self):
        """
        Sanity check.
        """
        if self.config.mode == 'train':
            dataset_names = ["train", "valid"]
            loaders = [self.train_loader, self.valid_loader]
        else:
            dataset_names = ["test"]
            loaders = [self.test_loader]

        for name, loader in zip(dataset_names, loaders):
            print(name)
            x, y, has_target, target = loader.dataset[loader.sampler.indices[0]]
            print("y:", y.item())
            print("attention target:", target.numpy() if has_target else "none")
            to_pil(x).save(f"{name}_0.png")

    def summarise(self):
        # print out test accuracy and training iterations
        test_acc_file = os.path.join(self.experiment_dir, "test_best_acc_T_5.txt")
        entry = open(test_acc_file, 'r').readline().strip()
        test_acc = float(entry.split(',')[1][1:-1])
        valid_file = os.path.join(self.experiment_dir, "valid.csv")

        score_meter = BatchScoreMeter()
        nlls = score_meter.load_from_csv(open(valid_file, 'r').read())[:, 1]
        iterations = iters_to_minimum(
            nlls, iters_per_value=self.valid_every, threshold=0.01
        )

        print(f"\n Test accuracy: {test_acc} \n Iterations until convergence: {iterations}")
