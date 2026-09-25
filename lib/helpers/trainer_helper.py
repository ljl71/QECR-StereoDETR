import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        self.amp_enabled = bool(cfg.get('amp_enabled', False))
        if self.amp_enabled and not torch.cuda.is_available():
            raise RuntimeError('trainer.amp_enabled requires CUDA')
        amp_dtype_name = str(cfg.get('amp_dtype', 'float16')).lower()
        if amp_dtype_name not in ('float16', 'fp16'):
            raise ValueError(
                'trainer.amp_dtype currently supports only float16/fp16'
            )
        self.amp_dtype = torch.float16
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled
        )
        self.zero_grad_set_to_none = bool(
            cfg.get('zero_grad_set_to_none', False)
        )
        self.max_train_batches = cfg.get('max_train_batches')
        if self.max_train_batches is not None:
            self.max_train_batches = int(self.max_train_batches)
            if self.max_train_batches <= 0:
                raise ValueError('trainer.max_train_batches must be positive')
        self.logger.info(
            'Training precision: %s',
            'CUDA AMP float16' if self.amp_enabled else 'FP32',
        )

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger,
                            strict=bool(cfg.get('pretrain_strict', True)),
                            allowed_missing_prefixes=cfg.get(
                                'pretrain_allowed_missing_prefixes'
                            ),
                            allow_unexpected=bool(
                                cfg.get('pretrain_allow_unexpected', True)
                            ))

        resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
        resume_requested = bool(cfg.get('resume_model', None))
        resume_if_exists = bool(cfg.get('resume_if_exists', False))
        if resume_requested or (resume_if_exists and os.path.exists(resume_model_path)):
            if not os.path.exists(resume_model_path):
                raise FileNotFoundError(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger,
                amp_scaler=self.grad_scaler if self.amp_enabled else None)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        elif resume_if_exists:
            self.logger.info(
                "No rolling checkpoint found; starting the fixed-epoch run "
                "from its configured initialization."
            )
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            if hasattr(self.detr_loss, "set_training_epoch"):
                self.detr_loss.set_training_epoch(epoch)
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    get_checkpoint_state(
                        self.model,
                        self.optimizer,
                        self.epoch,
                        best_result,
                        best_epoch,
                        amp_scaler=(
                            self.grad_scaler if self.amp_enabled else None
                        ),
                    ),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(
                                self.model,
                                self.optimizer,
                                self.epoch,
                                best_result,
                                best_epoch,
                                amp_scaler=(
                                    self.grad_scaler
                                    if self.amp_enabled else None
                                ),
                            ),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        # KITTI test labels are hidden, so a trainval run must not select a
        # checkpoint by test performance.  When explicitly requested, retain
        # the state after the preregistered number of epochs under an
        # unambiguous name.  Existing validation-based experiments keep their
        # historical behaviour because this switch is disabled by default.
        if bool(self.cfg.get('save_final_checkpoint', False)):
            os.makedirs(self.output_dir, exist_ok=True)
            final_checkpoint = os.path.join(
                self.output_dir,
                'checkpoint_final',
            )
            save_checkpoint(
                get_checkpoint_state(
                    self.model,
                    self.optimizer,
                    self.epoch,
                    best_result,
                    best_epoch,
                    amp_scaler=(
                        self.grad_scaler if self.amp_enabled else None
                    ),
                ),
                final_checkpoint,
            )
            self.logger.info(
                "Saved fixed-epoch final checkpoint: %s.pth (epoch=%d)",
                final_checkpoint,
                self.epoch,
            )

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets.keys():
                if key not in ["img_id"]:
                    targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size_croped']
            img_sizes_ori = info['img_size_original'].to(self.device)
            img_sizes_upper = info['upper'].to(self.device)
            targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
            ###
            # train one batch
            self.optimizer.zero_grad(
                set_to_none=self.zero_grad_set_to_none
            )
            with torch.cuda.amp.autocast(
                enabled=self.amp_enabled,
                dtype=self.amp_dtype,
            ):
                outputs = self.model(
                    inputs,
                    calibs,
                    targets,
                    img_sizes,
                    img_sizes_ori,
                    img_sizes_upper,
                    dn_args=dn_args,
                )
                mask_dict=None
                #ipdb.set_trace()
                detr_losses_dict = self.detr_loss(
                    outputs,
                    targets,
                    mask_dict,
                )

                weight_dict = self.detr_loss.weight_dict
                detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
                detr_losses = sum(detr_losses_dict_weighted)

            if not torch.isfinite(detr_losses).all():
                raise FloatingPointError(
                    'non-finite detector loss at epoch {} batch {}'.format(
                        epoch, batch_idx
                    )
                )

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")
                print("")
                print("")

            if self.amp_enabled:
                self.grad_scaler.scale(detr_losses).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                detr_losses.backward()
                self.optimizer.step()

            progress_bar.update()
            if (
                self.max_train_batches is not None
                and batch_idx + 1 >= self.max_train_batches
            ):
                self.logger.info(
                    'Stopped epoch after %d batches as configured.',
                    self.max_train_batches,
                )
                break
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin',
                    'heading_res', 'boxes_3d', 'sample_points',
                    'disp', "random_flip_flag", "random_switch_flag",
                    "teacher_depth", "teacher_confidence"]
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    if key in ['disp', "random_flip_flag", "random_switch_flag",
                               "teacher_depth", "teacher_confidence"]:
                        target_dict[key] = val[bz]
                    else:
                        target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list
