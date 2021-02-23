import os
from datetime import datetime

import torch
import torch.distributed as dist

from . import meters
from . import utils


class Trainer():
    def __init__(self, cfgs, model):
        # basic parameters
        self.distributed = cfgs.get('distributed')
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.make_metrics = lambda m=None, mode='moving': meters.StandardMetrics(m, mode)
        self.model = model(cfgs)
        # functional parameters
        self.joint_train = cfgs.get('joint_train', False)  # True: joint train on multiple images
        self.independent = cfgs.get('independent', True)  # True: each process has a different image
        self.reset_weight = cfgs.get('reset_weight', False)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.save_results = cfgs.get('save_results', False)
        # detailed parameters
        self.num_stage = cfgs.get('num_stage')
        self.stage_len_dict = cfgs.get('stage_len_dict')
        self.stage_len_dict2 = cfgs.get('stage_len_dict2', None)
        self.flip1_cfg = cfgs.get('flip1_cfg', [False, False, False])
        self.flip3_cfg = cfgs.get('flip3_cfg', [False, False, False])
        self.img_list_path = cfgs.get('img_list_path')
        self.img_root = cfgs.get('img_root', None)
        self.latent_root = cfgs.get('latent_root', None)
        self.mode_seq = ['step1', 'step2', 'step3']
        self.current_stage = 0
        self.count = 0

        self.prepare_img_list()
        self.setup_state()

        if self.save_results and self.rank == 0:
            img_save_path = self.checkpoint_dir + '/images'
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

    def prepare_img_list(self):
        img_list_file = open(self.img_list_path)
        self.img_list, self.depth_list, self.latent_list = [], [], []
        for line in img_list_file.readlines():
            img_name = line.split()[0]
            img_path = os.path.join(self.img_root, img_name)
            latent_path = os.path.join(self.latent_root, img_name.replace('.png', '.pt'))
            self.img_list.append(img_path)
            if self.load_gt_depth:
                self.depth_list.append(img_path.replace('image', 'depth'))
            self.latent_list.append(latent_path)
        if self.independent:
            assert len(self.img_list) % self.world_size == 0

    def setup_data(self, epoch):
        if self.joint_train:
            self.model.setup_target(self.img_list,
                                    self.depth_list if self.load_gt_depth else None,
                                    self.latent_list)
        elif self.independent:
            idx = epoch * self.world_size
            self.model.setup_target(self.img_list[idx:idx+self.world_size],
                                    self.depth_list[idx:idx+self.world_size] if self.load_gt_depth else None,
                                    self.latent_list[idx:idx+self.world_size])
        else:
            self.model.setup_target(self.img_list[epoch],
                                    self.depth_list[epoch] if self.load_gt_depth else None,
                                    self.latent_list[epoch])

    def setup_mode(self):
        stage_len_dict = self.stage_len_dict if self.current_stage == 0 else self.stage_len_dict2
        if self.count >= stage_len_dict[self.model.mode]:
            if (self.independent or self.rank == 0) and self.save_results:
                if self.model.mode == 'step3':
                    self.model.save_results(self.current_stage+1)
                elif self.model.mode == 'step2' and self.current_stage == 0:
                    self.model.save_results(self.current_stage)
            if self.model.mode == 'step1' and self.joint_train:
                # collect results in step1
                self.model.step1_collect()
            if self.model.mode == 'step2':
                # collect projected samples
                self.model.step2_collect()
            if self.model.mode == self.mode_seq[-1]:  # finished a stage
                self.current_stage += 1
                if self.current_stage >= self.num_stage:
                    return -1
                self.setup_state()
            idx = self.mode_seq.index(self.model.mode)
            next_mode = self.mode_seq[(idx + 1) % len(self.mode_seq)]
            self.model.mode = next_mode
            self.model.init_optimizers()
            self.metrics.reset()
            self.count = 0
        self.count += 1
        return 1

    def setup_state(self):
        self.model.flip1 = self.flip1_cfg[self.current_stage]
        self.model.flip3 = self.flip3_cfg[self.current_stage]

    def reset_state(self):
        self.current_stage = 0
        self.count = 0
        self.model.mode = 'step1'
        if self.reset_weight:
            self.model.reset_model_weight()
        self.model.init_optimizers()
        self.model.canon_mask = None
        self.setup_state()

    def save_checkpoint(self, iteration, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified iteration."""
        utils.xmkdir(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{iteration:05}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['iteration'] = iteration
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def train(self):
        self.model.init_optimizers()

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            self.logger = None
            if self.rank == 0:
                self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

        ## run
        if self.joint_train:
            num_epoch = 1
        elif self.independent:
            num_epoch = len(self.img_list) // self.world_size
        else:
            num_epoch = len(self.img_list)
        self.metrics = self.make_metrics(mode='moving')
        metrics_all = self.make_metrics(mode='total')
        iteration_all = 0
        for epoch in range(num_epoch):
            self.reset_state()
            self.setup_data(epoch)
            self.metrics.reset()
            i = 0
            while(True):
                state = self.setup_mode()
                if state < 0:
                    metrics_all.update(m, 1)
                    if self.rank == 0:
                        print(f"{'Epoch'}{epoch:05}/{metrics_all}")
                        self.save_checkpoint(iteration_all)
                    break
                m = self.model.forward()
                self.model.backward()
                if self.distributed:
                    for k, v in m.items():
                        if type(v) == torch.Tensor:
                            dist.all_reduce(v)
                            m[k] = v / dist.get_world_size()

                self.metrics.update(m, 1)
                if self.rank == 0:
                    print(f"{'E'}{epoch:04}/{'T'}{i:05}/{self.model.mode}/{self.metrics}")

                if self.use_logger:
                    if i % self.log_freq == 0:
                        self.model.visualize(self.logger, iteration=iteration_all)

                if (iteration_all+1) % self.save_checkpoint_freq == 0 and self.rank == 0:
                    self.save_checkpoint(iteration_all+1)

                if self.joint_train:
                    self.model.next_image()
                i += 1
                iteration_all += 1
